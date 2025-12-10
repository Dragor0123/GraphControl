import torch
import json
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np


from utils.random import reset_random_seed
from utils.args import Arguments
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes, precompute_khop_conditions_pure, precompute_khop_conditions_cumulative
from models import load_model
from datasets import NodeDataset
from optimizers import create_optimizer


def preprocess(config, dataset_obj, device):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 4, 'persistent_workers': True, 'pin_memory': True}
    
    print('generating subgraphs....')
    
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()
    
    train_graphs = collect_subgraphs(train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)

    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in train_graphs]
    [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in test_graphs]
    
        
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)

    return train_loader, test_loader


def finetune(config, model, train_loader, device, full_x_sim, test_loader):
    # freeze the pre-trained encoder (left branch)
    for k, v in model.named_parameters():
        if 'encoder' in k:
            v.requires_grad = False
            
    model.reset_classifier()
    eval_steps = 3
    patience = 15
    count = 0
    best_acc = 0
    log_interval = 5
    num_layers = len(model.encoder.gnn.layers)
    norm_logs = []

    params  = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    process_bar = tqdm(range(config.epochs))
    log_interval = config.log_interval if getattr(config, "log_norms", False) else None
    norm_logs = [] if log_interval is not None else None

    for epoch in process_bar:
        epoch_accum = None
        if log_interval is not None and epoch % log_interval == 0:
            epoch_accum = [
                {
                    "h_frozen_norm_sum": 0.0,
                    "h_ctrl_norm_sum": 0.0,
                    "cond_hidden_norm_sum": 0.0,
                    "cond_input_norm_sum": 0.0,
                    "zero_out_norm_sum": 0.0,
                    "ratio_zero_to_frozen_sum": 0.0,
                    "count": 0,
                    "cond_input_count": 0
                }
                for _ in range(num_layers)
            ]

        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)

            log_accumulator = {"layers": epoch_accum} if epoch_accum is not None else None

            # Compute condition based on model type
            if config.model in ['GCC_GraphControl_KHopPure', 'GCC_GraphControl_KHopCumulative']:
                # Use precomputed k-hop conditions (indexed by original_idx)
                x_sim_list = [x_sim_k[data.original_idx] for x_sim_k in full_x_sim]
                preds = model.forward_subgraph(x, x_sim_list, data.edge_index, data.batch, data.root_n_id, frozen=True, log_accumulator=log_accumulator)
            else:
                x_sim = full_x_sim[data.original_idx]
                preds = model.forward_subgraph(x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True, log_accumulator=log_accumulator)
                
            loss = criterion(preds, data.y)
            loss.backward()
            optimizer.step()

        if epoch_accum is not None:
            epoch_entry = {"epoch": epoch, "layers": []}
            for layer_data in epoch_accum:
                cnt = layer_data["count"] if layer_data["count"] > 0 else 1
                cond_input_cnt = layer_data["cond_input_count"] if layer_data["cond_input_count"] > 0 else None
                epoch_entry["layers"].append({
                    "h_frozen_norm": layer_data["h_frozen_norm_sum"] / cnt,
                    "h_ctrl_norm": layer_data["h_ctrl_norm_sum"] / cnt,
                    "cond_hidden_norm": layer_data["cond_hidden_norm_sum"] / cnt,
                    "cond_input_norm": (layer_data["cond_input_norm_sum"] / cond_input_cnt) if cond_input_cnt else None,
                    "zero_out_norm": layer_data["zero_out_norm_sum"] / cnt,
                    "ratio_zero_to_frozen": layer_data["ratio_zero_to_frozen_sum"] / cnt,
                    "batch_count": cnt
                })
            if norm_logs is not None:
                norm_logs.append(epoch_entry)
    
        if epoch % eval_steps == 0:
            acc = eval_subgraph(config, model, test_loader, device, full_x_sim)
            process_bar.set_postfix({"Epoch": epoch, "Accuracy": f"{acc:.4f}"})
            if best_acc < acc:
                best_acc = acc
                count = 0
            else:
                count += 1

        if count == patience:
            break

    return best_acc, norm_logs


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    # For large graph, we use cpu to preprocess it rather than gpu because of OOM problem.
    if dataset_obj.num_nodes < 30000:
        dataset_obj.to(device)

    # Precompute conditions based on model type
    if config.model == 'GCC_GraphControl_KHopPure':
        print('Precomputing k-hop conditions (Pure)...')
        x_sim = [x.to(device) for x in precompute_khop_conditions_pure(
            dataset_obj.data, num_layers=5, threshold=config.threshold, num_dim=config.num_dim
        )]
    elif config.model == 'GCC_GraphControl_KHopCumulative':
        print('Precomputing k-hop conditions (Cumulative)...')
        x_sim = [x.to(device) for x in precompute_khop_conditions_cumulative(
            dataset_obj.data, num_layers=5, threshold=config.threshold, num_dim=config.num_dim
        )]
    else:
        x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold).to(device)

    dataset_obj.to('cpu') # Otherwise the deepcopy will raise an error
    num_node_features = config.num_dim

    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    acc_list = []

    for i, seed in enumerate(config.seeds):
        reset_random_seed(seed)
        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        elif dataset_obj.data.train_mask.dim() > 1:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        
        train_loader, test_loader = preprocess(config, dataset_obj, device)
        
        model = load_model(num_node_features, dataset_obj.num_classes, config)
        model = model.to(device)

        # finetuning model
        best_acc, norm_logs = finetune(config, model, train_loader, device, x_sim, test_loader)
        
        acc_list.append(best_acc)
        print(f'Seed: {seed}, Accuracy: {best_acc:.4f}')
        if getattr(config, "log_norms", False) and norm_logs is not None:
            log_dir = os.path.join('results', 'norm_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{config.dataset}_{config.model}_seed{seed}.json")
            with open(log_path, 'w') as f:
                json.dump(
                    {
                        "dataset": config.dataset,
                        "model": config.model,
                        "seed": seed,
                        "log_interval": config.log_interval,
                        "entries": norm_logs
                    },
                    f,
                    indent=2
                )

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


def eval_subgraph(config, model, test_loader, device, full_x_sim):
    model.eval()

    correct = 0
    total_num = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, 'root_n_id'):
            batch.root_n_id = batch.root_n_index

        # Compute condition based on model type
        if config.model in ['GCC_GraphControl_KHopPure', 'GCC_GraphControl_KHopCumulative']:
            # Use precomputed k-hop conditions (indexed by original_idx)
            x_sim_list = [x_sim_k[batch.original_idx] for x_sim_k in full_x_sim]
            preds = model.forward_subgraph(batch.x, x_sim_list, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)
        else:
            x_sim = full_x_sim[batch.original_idx]
            preds = model.forward_subgraph(batch.x, x_sim, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)

        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

if __name__ == '__main__':
    config = Arguments().parse_args()
    main(config)
