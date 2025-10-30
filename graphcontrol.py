import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np


from utils.random import reset_random_seed
from utils.args import Arguments
from utils.sampling import collect_subgraphs
from utils.transforms import process_attributes, obtain_attributes
from models import load_model
from models.disenlink_condition import DisenLinkConditionGenerator, get_dataset_specific_config
from datasets import NodeDataset
from optimizers import create_optimizer


def preprocess(config, dataset_obj, device):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 4, 'persistent_workers': True, 'pin_memory': True}
    
    print('generating subgraphs....')

    # Handle both 1D and 2D masks (for datasets with multiple preset splits)
    train_mask = dataset_obj.data.train_mask
    test_mask = dataset_obj.data.test_mask

    if train_mask.dim() > 1:
        # For datasets with multiple splits, use the first split
        train_idx = train_mask[:, 0].nonzero().squeeze()
        test_idx = test_mask[:, 0].nonzero().squeeze()
    else:
        # For datasets with single split
        train_idx = train_mask.nonzero().squeeze()
        test_idx = test_mask.nonzero().squeeze()
    
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

    params  = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_optimizer(name=config.optimizer, parameters=params, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    process_bar = tqdm(range(config.epochs))

    for epoch in process_bar:
        for data in train_loader:
            optimizer.zero_grad()
            model.train()

            data = data.to(device)
            
            if not hasattr(data, 'root_n_id'):
                data.root_n_id = data.root_n_index

            sign_flip = torch.rand(data.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x = data.x * sign_flip.unsqueeze(0)
            
            x_sim = full_x_sim[data.original_idx]
            preds = model.forward_subgraph(x, x_sim, data.edge_index, data.batch, data.root_n_id, frozen=True)
                
            loss = criterion(preds, data.y)
            loss.backward()
            optimizer.step()
    
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

    return best_acc


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()

    # Initialize DisenLink condition generator if enabled
    disenlink_generator = None
    if hasattr(config, 'use_disenlink') and config.use_disenlink:
        print(f"\n=== Using DisenLink for condition generation ===")

        # Get dataset-specific config
        disenlink_config = get_dataset_specific_config(config.dataset)

        # Override with user config if provided (only if not None)
        if hasattr(config, 'disenlink_factors') and config.disenlink_factors is not None:
            disenlink_config['num_factors'] = config.disenlink_factors
        if hasattr(config, 'disenlink_alpha') and config.disenlink_alpha is not None:
            disenlink_config['hybrid_alpha'] = config.disenlink_alpha
        if hasattr(config, 'disenlink_epochs') and config.disenlink_epochs is not None:
            disenlink_config['disenlink_epochs'] = config.disenlink_epochs

        print(f"DisenLink config: {disenlink_config}")

        disenlink_generator = DisenLinkConditionGenerator(
            num_factors=disenlink_config['num_factors'],
            hidden_dim=disenlink_config['hidden_dim'],
            embed_dim=disenlink_config['embed_dim'],
            use_hybrid=disenlink_config['use_hybrid'],
            hybrid_alpha=disenlink_config['hybrid_alpha'],
            beta=disenlink_config['beta'],
            device=device,
            train_epochs=disenlink_config.get('disenlink_epochs', 100)
        )

    # Generate condition matrix (either DisenLink or feature similarity)
    if disenlink_generator is not None:
        # Keep data on CPU for subgraph collection
        train_mask = dataset_obj.data.train_mask
        if train_mask.dim() > 1:
            train_idx = train_mask[:, 0].nonzero().squeeze()
        else:
            train_idx = train_mask.nonzero().squeeze()

        # Sample fewer subgraphs for DisenLink training (faster)
        sample_size = min(100, len(train_idx))
        sample_idx = train_idx[torch.randperm(len(train_idx))[:sample_size]]

        print(f"Collecting {sample_size} subgraphs for DisenLink training...")
        disenlink_subgraphs = collect_subgraphs(
            sample_idx,
            dataset_obj.data,
            walk_steps=config.walk_steps,
            restart_ratio=config.restart
        )

        x_sim = obtain_attributes(
            dataset_obj.data,
            use_adj=False,
            threshold=config.threshold,
            use_disenlink=True,
            disenlink_generator=disenlink_generator,
            subgraphs=disenlink_subgraphs
        ).to(device)

        # Now move dataset to device if needed
        if dataset_obj.num_nodes < 30000:
            dataset_obj.to(device)
    else:
        # For large graph, we use cpu to preprocess it rather than gpu because of OOM problem.
        if dataset_obj.num_nodes < 30000:
            dataset_obj.to(device)

        x_sim = obtain_attributes(
            dataset_obj.data,
            use_adj=False,
            threshold=config.threshold
        ).to(device)
    
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
        
        train_loader, test_loader = preprocess(config, dataset_obj, device)
        
        model = load_model(num_node_features, dataset_obj.num_classes, config)
        model = model.to(device)

        # finetuning model
        best_acc = finetune(config, model, train_loader, device, x_sim, test_loader)
        
        acc_list.append(best_acc)
        print(f'Seed: {seed}, Accuracy: {best_acc:.4f}')

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
        x_sim = full_x_sim[batch.original_idx]
        preds = model.forward_subgraph(batch.x, x_sim, batch.edge_index, batch.batch, batch.root_n_id, frozen=True).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

if __name__ == '__main__':
    config = Arguments().parse_args()
    
    main(config)