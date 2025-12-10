# GraphControl Execution Flow Chart

This document illustrates the execution flow of graphcontrol.py for two model variants:
- `--model GCC_GraphControl`
- `--model GCC_GraphControl_KHopPure`

---

## Main Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        START (main)                              │
│                   graphcontrol.py:177                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Parse Arguments                                              │
│     - Arguments().parse_args()                                   │
│     - config.model = 'GCC_GraphControl' or                      │
│                      'GCC_GraphControl_KHopPure'                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Initialize Dataset                                           │
│     - NodeDataset(config.dataset, n_seeds)                      │
│     - dataset_obj.print_statistics()                            │
│     - Move to device if num_nodes < 30000                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────┴─────────┐
                   │  Branch by Model  │
                   └─────────┬─────────┘
                             │
         ┌───────────────────┴──────────────────┐
         │                                      │
         ▼                                      ▼
┌────────────────────┐              ┌────────────────────────┐
│ GCC_GraphControl   │              │ GCC_GraphControl_      │
│                    │              │    KHopPure            │
└────────┬───────────┘              └───────────┬────────────┘
         │                                      │
         ▼                                      ▼
┌────────────────────┐              ┌────────────────────────┐
│ obtain_attributes()│              │ precompute_khop_       │
│                    │              │    conditions_pure()   │
│ Returns:           │              │                        │
│  x_sim: [N, 32]    │              │ Returns:               │
│  (single tensor)   │              │  x_sim: List[Tensor]   │
│                    │              │  len=5, each [N, 32]   │
└────────┬───────────┘              └───────────┬────────────┘
         │                                      │
         └───────────────────┬──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Move dataset to CPU                                          │
│     - dataset_obj.to('cpu')                                     │
│     - Save train_masks and test_masks                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Loop over Seeds                                              │
│     for i, seed in enumerate(config.seeds):                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │      Per-Seed Processing Loop          │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1 Reset Random Seed                          │
        │      reset_random_seed(seed)                    │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.2 Set Train/Test Masks                       │
        │      dataset_obj.data.train_mask = ...          │
        │      dataset_obj.data.test_mask = ...           │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.3 Preprocess (Generate Subgraphs)            │
        │      preprocess(config, dataset_obj, device)    │
        │                                                 │
        │      See detailed flow below ↓                  │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.4 Load Model                                 │
        │      model = load_model(...)                    │
        │      - GCC_GraphControl or                      │
        │        GCC_GraphControl_KHopPure                │
        │      model.to(device)                           │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.5 Fine-tune Model                            │
        │      finetune(config, model, train_loader,      │
        │               device, x_sim, test_loader)       │
        │                                                 │
        │      See detailed flow below ↓                  │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.6 Record Accuracy                            │
        │      acc_list.append(best_acc)                  │
        └────────────────────────┬────────────────────────┘
                                 │
                                 │
        └─────────────────────────┴─────────────────────┐
                                                        │
                                                        │ (Loop back for next seed)
                                                        │
┌───────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Compute Final Results                                        │
│     final_acc = mean(acc_list)                                  │
│     final_acc_std = std(acc_list)                               │
│     Print results                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                          [ END ]
```

---

## Detailed Flow: preprocess()

```
┌─────────────────────────────────────────────────────────────────┐
│                    preprocess(config, dataset_obj, device)       │
│                         graphcontrol.py:16                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Extract Train/Test Indices                                   │
│     train_idx = dataset_obj.data.train_mask.nonzero().squeeze() │
│     test_idx = dataset_obj.data.test_mask.nonzero().squeeze()   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Generate Subgraphs via Random Walk                           │
│     collect_subgraphs(train_idx, dataset_obj.data,              │
│                      walk_steps, restart_ratio)                 │
│                                                                  │
│     For each node in train_idx/test_idx:                        │
│     - Random walk with restart                                  │
│     - Extract subgraph from path                                │
│     - Create Data object with:                                  │
│       * x: node features                                        │
│       * edge_index: subgraph edges                              │
│       * y: node label                                           │
│       * original_idx: node index in full graph                  │
│       * root_n_id: target node in subgraph                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Process Attributes for Each Subgraph                         │
│     [process_attributes(g, use_adj, threshold, num_dim)         │
│      for g in train_graphs + test_graphs]                       │
│                                                                  │
│     For each subgraph:                                          │
│     - Compute similarity matrix from features                   │
│     - Discretize by threshold                                   │
│     - Compute Laplacian eigenvectors                            │
│     - Replace g.x with positional encoding [num_nodes, 32]      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Create DataLoaders                                           │
│     train_loader = DataLoader(train_graphs, shuffle=True, ...)  │
│     test_loader = DataLoader(test_graphs, ...)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        [ RETURN ]
                   train_loader, test_loader
```

---

## Detailed Flow: finetune()

```
┌─────────────────────────────────────────────────────────────────┐
│              finetune(config, model, train_loader,               │
│                       device, x_sim, test_loader)                │
│                         graphcontrol.py:37                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Freeze Encoder                                               │
│     for k, v in model.named_parameters():                       │
│         if 'encoder' in k:                                      │
│             v.requires_grad = False                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Reset Classifier                                             │
│     model.reset_classifier()                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Initialize Training Components                               │
│     - Create optimizer (only trainable params)                  │
│     - criterion = CrossEntropyLoss()                            │
│     - best_acc = 0, patience counter                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Training Loop: for epoch in range(config.epochs)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1 Batch Loop: for data in train_loader       │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1.1 Prepare Data                             │
        │        data.to(device)                          │
        │        Set data.root_n_id if needed             │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1.2 Apply Sign Flip Augmentation             │
        │        sign_flip = random {-1, 1}               │
        │        x = data.x * sign_flip                   │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
                       ┌─────────┴─────────┐
                       │  Branch by Model  │
                       └─────────┬─────────┘
                                 │
         ┌───────────────────────┴──────────────────┐
         │                                          │
         ▼                                          ▼
┌────────────────────────┐              ┌─────────────────────────┐
│ GCC_GraphControl       │              │ GCC_GraphControl_       │
│                        │              │    KHopPure             │
└────────┬───────────────┘              └───────────┬─────────────┘
         │                                          │
         ▼                                          ▼
┌────────────────────────┐              ┌─────────────────────────┐
│ Get single condition:  │              │ Get condition list:     │
│                        │              │                         │
│ x_sim_batch =          │              │ x_sim_list = [          │
│   full_x_sim[          │              │   x_sim_k[              │
│     data.original_idx] │              │     data.original_idx]  │
│                        │              │   for x_sim_k           │
│ Shape: [B, 32]         │              │     in full_x_sim]      │
│                        │              │                         │
│                        │              │ Shape: List of [B, 32]  │
│                        │              │        (5 tensors)      │
└────────┬───────────────┘              └───────────┬─────────────┘
         │                                          │
         ▼                                          ▼
┌────────────────────────┐              ┌─────────────────────────┐
│ model.forward_subgraph │              │ model.forward_subgraph  │
│  (x, x_sim,            │              │  (x, x_sim_list,        │
│   edge_index, batch,   │              │   edge_index, batch,    │
│   root_n_id,           │              │   root_n_id,            │
│   frozen=True)         │              │   frozen=True)          │
│                        │              │                         │
│ See flow below ↓       │              │ See flow below ↓        │
└────────┬───────────────┘              └───────────┬─────────────┘
         │                                          │
         └───────────────────┬──────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1.3 Compute Loss & Update                    │
        │        loss = criterion(preds, data.y)          │
        │        loss.backward()                          │
        │        optimizer.step()                         │
        └────────────────────────┬────────────────────────┘
                                 │
                                 │ (Next batch)
                                 │
        └────────────────────────┴────────────────────────┐
                                                           │
┌──────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Evaluation (every eval_steps epochs)                         │
│     acc = eval_subgraph(config, model, test_loader,             │
│                         device, x_sim)                          │
│     Update best_acc and patience counter                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Early Stopping Check                                         │
│     if count == patience: break                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        [ RETURN ]
                         best_acc
```

---

## Detailed Flow: forward_subgraph() for GCC_GraphControl

```
┌─────────────────────────────────────────────────────────────────┐
│    model.forward_subgraph(x, x_sim, edge_index,                 │
│                           batch, root_n_id, frozen=True)         │
│                   gcc_graphcontrol.py:47                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Prepare Initial Features (Frozen Encoder)                    │
│     with torch.no_grad():                                       │
│         h_frozen = self.encoder.prepare_node_features(          │
│                      x, edge_index, root_n_id)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Prepare Control Features (Trainable Copy)                    │
│     h_ctrl = self.trainable_copy.prepare_node_features(         │
│                x, edge_index, root_n_id)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Project Condition (SINGLE condition for all layers)          │
│     cond_hidden = self.cond_proj(x_sim)                         │
│     cond_first_layer = self.cond_input_adapter(cond_hidden)     │
│                                                                  │
│     x_sim shape: [batch_size, 32]                               │
│     cond_hidden shape: [batch_size, hidden_size]                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Initialize Hidden States                                     │
│     hidden_states = [h_frozen]                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Layer-wise Processing Loop                                   │
│     for layer_idx, (layer_frozen, layer_ctrl, zero_layer)       │
│         in enumerate(zip(encoder_layers, ctrl_layers,           │
│                          zero_layers)):                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  5.1 Forward Frozen Branch                      │
        │      with torch.no_grad():                      │
        │          h_frozen = layer_frozen(h_frozen,      │
        │                                  edge_index)    │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  5.2 Add Condition to Control Branch            │
        │      if layer_idx == 0:                         │
        │          ctrl_input = h_ctrl + cond_first_layer │
        │      else:                                      │
        │          ctrl_input = h_ctrl + cond_hidden      │
        │                                                 │
        │      *** SAME condition for ALL layers ***      │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  5.3 Forward Control Branch                     │
        │      h_ctrl = layer_ctrl(ctrl_input, edge_index)│
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  5.4 Merge Branches via Zero-initialized Layer  │
        │      zero_out = zero_layer(h_ctrl)              │
        │      h_frozen = h_frozen + 0.01 * zero_out      │
        │      hidden_states.append(h_frozen)             │
        └────────────────────────┬────────────────────────┘
                                 │
                                 │ (Next layer)
                                 │
        └────────────────────────┴────────────────────────┐
                                                           │
┌──────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Graph Readout                                                │
│     out, _ = self.encoder.gnn.graph_readout(hidden_states,      │
│                                              batch)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Normalize (if enabled)                                       │
│     if self.encoder.norm:                                       │
│         out = F.normalize(out, p=2, dim=-1)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. Classification                                               │
│     x = self.linear_classifier(out)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        [ RETURN ]
                            x
```

---

## Detailed Flow: forward_subgraph() for GCC_GraphControl_KHopPure

```
┌─────────────────────────────────────────────────────────────────┐
│    model.forward_subgraph(x, x_sim_list, edge_index,            │
│                           batch, root_n_id, frozen=True)         │
│                   gcc_graphcontrol.py:137                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Prepare Initial Features (Frozen Encoder)                    │
│     with torch.no_grad():                                       │
│         h_frozen = self.encoder.prepare_node_features(          │
│                      x, edge_index, root_n_id)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Prepare Control Features (Trainable Copy)                    │
│     h_ctrl = self.trainable_copy.prepare_node_features(         │
│                x, edge_index, root_n_id)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Initialize Hidden States                                     │
│     hidden_states = [h_frozen]                                  │
│                                                                  │
│     *** NO condition projection yet ***                         │
│     *** Each layer uses different condition ***                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Layer-wise Processing Loop                                   │
│     for layer_idx, (layer_frozen, layer_ctrl, zero_layer)       │
│         in enumerate(zip(encoder_layers, ctrl_layers,           │
│                          zero_layers)):                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  4.1 Forward Frozen Branch                      │
        │      with torch.no_grad():                      │
        │          h_frozen = layer_frozen(h_frozen,      │
        │                                  edge_index)    │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.2 Select k-hop Specific Condition            │
        │      x_sim_k = x_sim_list[layer_idx]            │
        │                                                 │
        │      *** LAYER-SPECIFIC condition ***           │
        │      Layer 0: X @ X.T (no masking)              │
        │      Layer k: A^k ⊙ (X @ X.T) (k-hop only)      │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.3 Project Condition                          │
        │      cond_hidden = self.cond_proj(x_sim_k)      │
        │                                                 │
        │      x_sim_k shape: [batch_size, 32]            │
        │      cond_hidden: [batch_size, hidden_size]     │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.4 Add Condition to Control Branch            │
        │      if layer_idx == 0:                         │
        │          cond_first_layer =                     │
        │            self.cond_input_adapter(cond_hidden) │
        │          ctrl_input = h_ctrl + cond_first_layer │
        │      else:                                      │
        │          ctrl_input = h_ctrl + cond_hidden      │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.5 Forward Control Branch                     │
        │      h_ctrl = layer_ctrl(ctrl_input, edge_index)│
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  4.6 Merge Branches via Zero-initialized Layer  │
        │      zero_out = zero_layer(h_ctrl)              │
        │      h_frozen = h_frozen + 0.01 * zero_out      │
        │      hidden_states.append(h_frozen)             │
        └────────────────────────┬────────────────────────┘
                                 │
                                 │ (Next layer with next k-hop)
                                 │
        └────────────────────────┴────────────────────────┐
                                                           │
┌──────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Graph Readout                                                │
│     out, _ = self.encoder.gnn.graph_readout(hidden_states,      │
│                                              batch)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Normalize (if enabled)                                       │
│     if self.encoder.norm:                                       │
│         out = F.normalize(out, p=2, dim=-1)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Classification                                               │
│     x = self.linear_classifier(out)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        [ RETURN ]
                            x
```

---

## Key Differences Between the Two Models

### GCC_GraphControl
```
┌──────────────────────────────────────────────────────────────┐
│ Condition Computation (Once at Start):                       │
│                                                              │
│  obtain_attributes(data, threshold, num_dim)                │
│    ├─ Compute: S = X @ X.T (feature similarity)             │
│    ├─ Discretize: S_binary = (S > threshold ? 1 : 0)        │
│    ├─ Laplacian: L = D - S_binary                           │
│    └─ Eigenvectors: x_sim = V[:, :32]                       │
│                                                              │
│  Result: x_sim [N, 32] - SINGLE condition tensor             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Layer Processing:                                            │
│                                                              │
│  Layer 0: h_ctrl + cond_input_adapter(cond_proj(x_sim))      │
│  Layer 1: h_ctrl + cond_proj(x_sim)                          │
│  Layer 2: h_ctrl + cond_proj(x_sim)  ◄─ SAME condition       │
│  Layer 3: h_ctrl + cond_proj(x_sim)  ◄─ SAME condition       │
│  Layer 4: h_ctrl + cond_proj(x_sim)  ◄─ SAME condition       │
│                                                              │
│  All layers use IDENTICAL condition from global similarity   │
└──────────────────────────────────────────────────────────────┘
```

### GCC_GraphControl_KHopPure
```
┌──────────────────────────────────────────────────────────────┐
│ Condition Computation (Once at Start):                       │
│                                                              │
│  precompute_khop_conditions_pure(data, num_layers=5, ...)   │
│                                                              │
│  For k in [0, 1, 2, 3, 4]:                                   │
│    ├─ Compute: S = X @ X.T                                  │
│    ├─ If k > 0: compute A^k                                 │
│    ├─ Mask: S_masked = A^k ⊙ S (only k-hop neighbors)       │
│    ├─ Discretize: S_binary = (S_masked > threshold ? 1 : 0) │
│    ├─ Laplacian: L = D - S_binary                           │
│    └─ Eigenvectors: x_sim_k = V[:, :32]                     │
│                                                              │
│  Result: [x_sim_0, x_sim_1, ..., x_sim_4]                   │
│          Each [N, 32] - LIST of 5 condition tensors          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Layer Processing:                                            │
│                                                              │
│  Layer 0: h_ctrl + cond_input_adapter(cond_proj(x_sim_0))    │
│           └─ x_sim_0 from: X @ X.T (0-hop = self)            │
│                                                              │
│  Layer 1: h_ctrl + cond_proj(x_sim_1)                        │
│           └─ x_sim_1 from: A ⊙ (X @ X.T) (1-hop neighbors)   │
│                                                              │
│  Layer 2: h_ctrl + cond_proj(x_sim_2)                        │
│           └─ x_sim_2 from: A² ⊙ (X @ X.T) (2-hop neighbors)  │
│                                                              │
│  Layer 3: h_ctrl + cond_proj(x_sim_3)                        │
│           └─ x_sim_3 from: A³ ⊙ (X @ X.T) (3-hop neighbors)  │
│                                                              │
│  Layer 4: h_ctrl + cond_proj(x_sim_4)                        │
│           └─ x_sim_4 from: A⁴ ⊙ (X @ X.T) (4-hop neighbors)  │
│                                                              │
│  Each layer uses DIFFERENT condition based on hop distance   │
└──────────────────────────────────────────────────────────────┘
```

### Summary Table

| Aspect                    | GCC_GraphControl               | GCC_GraphControl_KHopPure        |
|---------------------------|--------------------------------|----------------------------------|
| Condition Type            | Single global condition        | List of k-hop conditions         |
| Precomputation Function   | `obtain_attributes()`          | `precompute_khop_conditions_pure()` |
| Condition Shape           | `[N, 32]`                      | `List[[N, 32], ...]` (len=5)     |
| Layer 0 Condition         | X @ X.T                        | X @ X.T (0-hop)                  |
| Layer k Condition         | X @ X.T (same as layer 0)      | A^k ⊙ (X @ X.T) (k-hop only)     |
| Condition Varies by Layer | ❌ No                          | ✅ Yes                           |
| Residual Scale (zero_out) | `h_frozen += 0.01 * zero_out`  | `h_frozen += 0.01 * zero_out`    |
| Control Branch Init       | cond_proj/cond_input/zero_layer zero-init | Same zero-init + fixed scale 0.01 |
| Forward Call              | `(x, x_sim, ...)`              | `(x, x_sim_list, ...)`           |

---

## Evaluation Flow: eval_subgraph()

```
┌─────────────────────────────────────────────────────────────────┐
│    eval_subgraph(config, model, test_loader,                    │
│                  device, full_x_sim)                            │
│                   graphcontrol.py:152                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Set Model to Eval Mode                                       │
│     model.eval()                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Initialize Counters                                          │
│     correct = 0                                                 │
│     total_num = 0                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Batch Loop: for batch in test_loader                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  3.1 Prepare Batch                              │
        │      batch.to(device)                           │
        │      Set batch.root_n_id if needed              │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
                       ┌─────────┴─────────┐
                       │  Branch by Model  │
                       └─────────┬─────────┘
                                 │
         ┌───────────────────────┴──────────────────┐
         │                                          │
         ▼                                          ▼
┌────────────────────────┐              ┌─────────────────────────┐
│ GCC_GraphControl       │              │ GCC_GraphControl_       │
│                        │              │    KHopPure             │
└────────┬───────────────┘              └───────────┬─────────────┘
         │                                          │
         ▼                                          ▼
┌────────────────────────┐              ┌─────────────────────────┐
│ x_sim =                │              │ x_sim_list = [          │
│   full_x_sim[          │              │   x_sim_k[              │
│     batch.original_idx]│              │     batch.original_idx] │
│                        │              │   for x_sim_k           │
│ preds = model.         │              │     in full_x_sim]      │
│   forward_subgraph(    │              │                         │
│     batch.x, x_sim,    │              │ preds = model.          │
│     ...).argmax(dim=1) │              │   forward_subgraph(     │
│                        │              │     batch.x, x_sim_list,│
│                        │              │     ...).argmax(dim=1)  │
└────────┬───────────────┘              └───────────┬─────────────┘
         │                                          │
         └───────────────────┬──────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  3.2 Update Counters                            │
        │      correct += (preds == batch.y).sum().item() │
        │      total_num += batch.y.shape[0]              │
        └────────────────────────┬────────────────────────┘
                                 │
                                 │ (Next batch)
                                 │
        └────────────────────────┴────────────────────────┐
                                                           │
┌──────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Compute Accuracy                                             │
│     acc = correct / total_num                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        [ RETURN ]
                           acc
```

---

## Condition Computation Details

### obtain_attributes() - For GCC_GraphControl

```
Input: data.x [N, d], threshold=0.15, num_dim=32
        │
        ▼
┌─────────────────────────────────┐
│ 1. Compute Similarity Matrix    │
│    S = X @ X.T                  │
│    Shape: [N, N]                │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 2. Discretize                   │
│    S_binary = (S > 0.15 ? 1:0)  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 3. Compute Laplacian            │
│    L = D - S_binary             │
│    (D is degree matrix)         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 4. Eigendecomposition           │
│    L, V = eigh(L)               │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 5. Extract First k Eigenvectors │
│    x = V[:, :32]                │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 6. L2 Normalization             │
│    x = normalize(x, norm="l2") │
└────────────┬────────────────────┘
             │
             ▼
        Output: [N, 32]
```

### precompute_khop_conditions_pure() - For GCC_GraphControl_KHopPure

```
Input: data (x, edge_index), num_layers=5, threshold=0.15, num_dim=32
        │
        ▼
┌─────────────────────────────────┐
│ Compute S = X @ X.T once        │
│ Precompute adj matrix           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Loop: for k in [0, 1, 2, 3, 4]  │
└────────────┬────────────────────┘
             │
             ▼
     ┌───────┴───────┐
     │   k = 0 ?     │
     └───────┬───────┘
         Yes │   No
             │    │
             ▼    ▼
      ┌──────────────────┐    ┌─────────────────────┐
      │ S_masked = S     │    │ Compute A^k         │
      │ (no masking)     │    │ S_masked = A^k ⊙ S  │
      │                  │    │ (k-hop neighbors)   │
      └──────┬───────────┘    └─────────┬───────────┘
             │                          │
             └───────────┬──────────────┘
                         │
                         ▼
             ┌────────────────────────┐
             │ Discretize:            │
             │ S_discrete =           │
             │   (S_masked > 0.15?1:0)│
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │ Laplacian:             │
             │ L = D - S_discrete     │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │ Eigendecomposition:    │
             │ eigen_vals, eigen_vecs │
             │   = eigh(L)            │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │ Extract & Normalize:   │
             │ PE = eigen_vecs[:, :32]│
             │ PE = normalize(PE)     │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │ Append to conditions   │
             │ list                   │
             └────────────┬───────────┘
                          │
                          │ (loop back for next k)
                          │
             ┌────────────┴───────────┐
             │                        │
Output: [PE_0, PE_1, PE_2, PE_3, PE_4]
        Each PE_k: [N, 32]

Legend:
  PE_0: from X @ X.T (0-hop = self similarity)
  PE_1: from A ⊙ (X @ X.T) (1-hop neighbors)
  PE_2: from A² ⊙ (X @ X.T) (2-hop neighbors)
  PE_3: from A³ ⊙ (X @ X.T) (3-hop neighbors)
  PE_4: from A⁴ ⊙ (X @ X.T) (4-hop neighbors)
```

---

## Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║                    GCC_GraphControl Architecture              ║
╚═══════════════════════════════════════════════════════════════╝

Input: x [B, num_nodes, 32], x_sim [B, num_nodes, 32], edge_index

    ┌──────────────────────────────────────────────────────────┐
    │                     Dual-Branch Design                    │
    └──────────────────────────────────────────────────────────┘

    ┌─────────────────────┐          ┌──────────────────────────┐
    │  Frozen Encoder     │          │  Trainable Copy          │
    │  (Pre-trained GCC)  │          │  + Condition Injection   │
    └──────────┬──────────┘          └────────────┬─────────────┘
               │                                  │
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────────┐
    │  prepare_node_      │          │  prepare_node_features() │
    │    features()       │          │                          │
    │  [no_grad]          │          │  [trainable]             │
    └──────────┬──────────┘          └───────────┬──────────────┘
               │                                  │
               │                      ┌───────────▼──────────────┐
               │                      │  cond_proj(x_sim)        │
               │                      │  cond_input_adapter()    │
               │                      │  [SAME x_sim all layers] │
               │                      └───────────┬──────────────┘
               │                                  │
    ╔══════════▼══════════════════════════════════▼══════════════╗
    ║              Layer-wise Processing (5 layers)              ║
    ╚════════════════════════════════════════════════════════════╝
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────────┐
    │  layer_frozen(h)    │          │  h_ctrl + condition      │
    │  [no_grad]          │          │  layer_ctrl(h_ctrl)      │
    └──────────┬──────────┘          └───────────┬──────────────┘
               │                                  │
               │                      ┌───────────▼──────────────┐
               │                      │  zero_layer(h_ctrl)      │
               │                      │  [zero-initialized]      │
               │                      └───────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────────────────────────────▼──────────────┐
    │  h_frozen = h_frozen + zero_layer(h_ctrl)                  │
    │  [Merge branches]                                          │
    └──────────┬─────────────────────────────────────────────────┘
               │
               │ (repeat 5 times)
               │
    ┌──────────▼──────────┐
    │  graph_readout()    │
    │  [aggregate nodes]  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  linear_classifier  │
    │  [output logits]    │
    └──────────┬──────────┘
               │
               ▼
          Output [B, num_classes]


╔═══════════════════════════════════════════════════════════════╗
║              GCC_GraphControl_KHopPure Architecture           ║
╚═══════════════════════════════════════════════════════════════╝

Input: x [B, num_nodes, 32],
       x_sim_list = [x_sim_0, x_sim_1, ..., x_sim_4]
       each [B, num_nodes, 32]
       edge_index

    ┌──────────────────────────────────────────────────────────┐
    │                     Dual-Branch Design                    │
    │                 with Layer-wise Conditions                │
    └──────────────────────────────────────────────────────────┘

    ┌─────────────────────┐          ┌──────────────────────────┐
    │  Frozen Encoder     │          │  Trainable Copy          │
    │  (Pre-trained GCC)  │          │  + K-hop Condition       │
    └──────────┬──────────┘          └────────────┬─────────────┘
               │                                  │
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────────┐
    │  prepare_node_      │          │  prepare_node_features() │
    │    features()       │          │                          │
    │  [no_grad]          │          │  [trainable]             │
    └──────────┬──────────┘          └───────────┬──────────────┘
               │                                  │
    ╔══════════▼══════════════════════════════════▼══════════════╗
    ║        Layer-wise Processing with K-hop Conditions         ║
    ╚════════════════════════════════════════════════════════════╝
               │                                  │
               │                      ┌───────────▼──────────────┐
               │                      │ x_sim_k = x_sim_list[k]  │
               │                      │ [DIFFERENT per layer]    │
               │                      │                          │
               │                      │ Layer 0: x_sim_0 (0-hop) │
               │                      │ Layer 1: x_sim_1 (1-hop) │
               │                      │ Layer 2: x_sim_2 (2-hop) │
               │                      │ Layer 3: x_sim_3 (3-hop) │
               │                      │ Layer 4: x_sim_4 (4-hop) │
               │                      └───────────┬──────────────┘
               │                                  │
               │                      ┌───────────▼──────────────┐
               │                      │  cond_proj(x_sim_k)      │
               │                      │  cond_input_adapter()    │
               │                      │  [if k==0]               │
               │                      └───────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────────┐
    │  layer_frozen(h)    │          │  h_ctrl + cond_k         │
    │  [no_grad]          │          │  layer_ctrl(h_ctrl)      │
    └──────────┬──────────┘          └───────────┬──────────────┘
               │                                  │
               │                      ┌───────────▼──────────────┐
               │                      │  zero_layer(h_ctrl)      │
               │                      │  [zero-initialized]      │
               │                      └───────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────────────────────────────▼──────────────┐
    │  h_frozen = h_frozen + zero_layer(h_ctrl)                  │
    │  [Merge branches]                                          │
    └──────────┬─────────────────────────────────────────────────┘
               │
               │ (repeat 5 times with different k-hop conditions)
               │
    ┌──────────▼──────────┐
    │  graph_readout()    │
    │  [aggregate nodes]  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  linear_classifier  │
    │  [output logits]    │
    └──────────┬──────────┘
               │
               ▼
          Output [B, num_classes]
```

---

## File References

- Main entry: `graphcontrol.py:177`
- Main function: `graphcontrol.py:96`
- Preprocess: `graphcontrol.py:16`
- Finetune: `graphcontrol.py:37`
- Evaluation: `graphcontrol.py:152`
- Subgraph collection: `utils/sampling.py:120`
- Process attributes: `utils/transforms.py:59`
- Obtain attributes: `utils/transforms.py:37`
- K-hop conditions (Pure): `utils/transforms.py:127`
- GCC_GraphControl model: `models/gcc_graphcontrol.py:10`
- GCC_GraphControl_KHopPure model: `models/gcc_graphcontrol.py:94`
