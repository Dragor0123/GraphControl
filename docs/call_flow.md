================================================================================
  GCC_GraphControl 실행 흐름 (함수 호출 순서)
  ================================================================================

  1. 프로그램 시작
     ├─ graphcontrol.py:main()
     │
     ├─ [1-1] 데이터셋 로드
     │  └─ NodeDataset(config.dataset, n_seeds=config.seeds)
     │     └─ datasets/node_dataset.py
     │
     ├─ [1-2] Condition 생성 (전체 그래프, 한 번만)
     │  └─ obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold)
     │     └─ utils/transforms.py:obtain_attributes()
     │        ├─ similarity(data.x, data.x)  # X @ X.T
     │        │  └─ utils/normalize.py:similarity()
     │        ├─ torch.where(tmp > threshold, 1.0, 0.0)  # 이진화
     │        ├─ get_laplacian_matrix(tmp)
     │        │  └─ utils/normalize.py:get_laplacian_matrix()
     │        └─ torch.linalg.eigh(tmp)  # 고유값 분해
     │        └─ return V[:, :32]  # x_sim [N, 32]
     │
     ├─ [1-3] Train/Test 데이터 준비
     │  └─ preprocess(config, dataset_obj, device)
     │     └─ graphcontrol.py:preprocess()
     │        ├─ process_attributes(dataset_obj.data, ...)  # Laplacian PE
     │        │  └─ utils/transforms.py:process_attributes()
     │        │     └─ obtain_attributes(data, use_adj=True, ...)
     │        │        └─ (위와 동일, 하지만 adjacency 기반)
     │        │
     │        └─ collect_subgraphs(dataset_obj, ...)  # Random walk subgraphs
     │           └─ utils/sampling.py:collect_subgraphs()
     │              ├─ random_walk_with_restart(...)
     │              └─ return train_loader, test_loader
     │
     └─ [1-4] 각 시드별로 실험 반복
        └─ for i, seed in enumerate(config.seeds):


  2. 모델 생성 (각 시드마다)
     └─ load_model(num_node_features, dataset_obj.num_classes, config)
        └─ models/model_manager.py:load_model()
           ├─ torch.load('checkpoint/gcc.pth')  # Pretrained weights
           │
           ├─ GCC_GraphControl.__init__(**kwargs)
           │  └─ models/gcc_graphcontrol.py:GCC_GraphControl.__init__()
           │     ├─ self.encoder = GCC(**kwargs)
           │     │  └─ models/gcc.py:GCC.__init__()
           │     │     ├─ GraphEncoder(...) 초기화
           │     │     ├─ 5개 GIN layers 생성
           │     │     └─ graph_readout (JK-style)
           │     │
           │     ├─ self.trainable_copy = copy.deepcopy(self.encoder)
           │     │
           │     ├─ self.cond_proj = nn.Linear(32, 128)  # SHARED
           │     ├─ self.cond_input_adapter = nn.Linear(128, 32)
           │     ├─ self.zero_layers = nn.ModuleList([...])  # 5개
           │     └─ self.linear_classifier = nn.Linear(128, num_classes)
           │
           ├─ model.encoder.load_state_dict(params)  # Frozen
           └─ model.trainable_copy.load_state_dict(params)  # Trainable


  3. Fine-tuning
     └─ finetune(config, model, train_loader, device, x_sim, test_loader)
        └─ graphcontrol.py:finetune()
           │
           ├─ [3-1] Freeze encoder
           │  └─ for k, v in model.named_parameters():
           │     if 'encoder' in k: v.requires_grad = False
           │
           ├─ [3-2] Reset classifier
           │  └─ model.reset_classifier()
           │     └─ models/gcc_graphcontrol.py:reset_classifier()
           │
           ├─ [3-3] Training loop
           │  └─ for epoch in range(config.epochs):
           │     │
           │     ├─ [3-3-1] Training
           │     │  └─ for data in train_loader:
           │     │     ├─ Sign flip augmentation
           │     │     │  └─ x = data.x * sign_flip
           │     │     │
           │     │     ├─ Extract condition for batch
           │     │     │  └─ x_sim = full_x_sim[data.original_idx]
           │     │     │     # [N_full, 32] → [N_batch, 32]
           │     │     │
           │     │     ├─ Forward pass
           │     │     │  └─ model.forward_subgraph(x, x_sim, ...)
           │     │     │     │
           │     │     │     └─ models/gcc_graphcontrol.py:forward_subgraph()
           │     │     │        │
           │     │     │        ├─ [Step 1] Condition processing (ONCE)
           │     │     │        │  ├─ cond_hidden = self.cond_proj(x_sim)
           │     │     │        │  │  # [N, 32] → [N, 128]
           │     │     │        │  └─ cond_first_layer = self.cond_input_adapter(cond_hidden)
           │     │     │        │     # [N, 128] → [N, 32]
           │     │     │        │
           │     │     │        ├─ [Step 2] Prepare initial features
           │     │     │        │  ├─ self.encoder.prepare_node_features(x, ...)
           │     │     │        │  │  └─ models/gcc.py:prepare_node_features()
           │     │     │        │  │     └─ embedding = self.node_embedding(x)
           │     │     │        │  │        # [N, 32] → [N, 32]
           │     │     │        │  │
           │     │     │        │  └─ self.trainable_copy.prepare_node_features(x, ...)
           │     │     │        │     # 동일
           │     │     │        │
           │     │     │        ├─ [Step 3] Layer-wise forward (5 layers)
           │     │     │        │  └─ for layer_idx in range(5):
           │     │     │        │     │
           │     │     │        │     ├─ Frozen branch
           │     │     │        │     │  └─ h_frozen = layer_frozen(h_frozen, edge_index)
           │     │     │        │     │     └─ models/gcc.py:GINConv.forward()
           │     │     │        │     │        ├─ MLP(x)
           │     │     │        │     │        └─ aggregate neighbors
           │     │     │        │     │
           │     │     │        │     ├─ Trainable branch
           │     │     │        │     │  ├─ if layer_idx == 0:
           │     │     │        │     │  │  └─ ctrl_input = h_ctrl + cond_first_layer
           │     │     │        │     │  ├─ else:
           │     │     │        │     │  │  └─ ctrl_input = h_ctrl + cond_hidden
           │     │     │        │     │  │
           │     │     │        │     │  └─ h_ctrl = layer_ctrl(ctrl_input, edge_index)
           │     │     │        │     │     └─ models/gcc.py:GINConv.forward()
           │     │     │        │     │
           │     │     │        │     └─ Inject into frozen
           │     │     │        │        └─ h_frozen = h_frozen + zero_layer(h_ctrl)
           │     │     │        │           └─ nn.Linear (zero initialized)
           │     │     │        │
           │     │     │        ├─ [Step 4] Graph readout
           │     │     │        │  └─ out, _ = self.encoder.gnn.graph_readout(hidden_states, batch)
           │     │     │        │     └─ models/gcc.py:GraphEncoder.graph_readout()
           │     │     │        │        ├─ Concatenate all layer outputs
           │     │     │        │        └─ Global mean pooling per graph
           │     │     │        │           # [N, 128] → [B, 128]
           │     │     │        │
           │     │     │        ├─ [Step 5] Normalization
           │     │     │        │  └─ out = F.normalize(out, p=2, dim=-1)
           │     │     │        │
           │     │     │        └─ [Step 6] Classification
           │     │     │           └─ x = self.linear_classifier(out)
           │     │     │              # [B, 128] → [B, num_classes]
           │     │     │
           │     │     ├─ Loss computation
           │     │     │  └─ loss = criterion(preds, data.y)
           │     │     │     └─ nn.CrossEntropyLoss
           │     │     │
           │     │     └─ Backward & Update
           │     │        ├─ loss.backward()
           │     │        └─ optimizer.step()
           │     │
           │     └─ [3-3-2] Evaluation (every 3 epochs)
           │        └─ if epoch % 3 == 0:
           │           └─ eval_subgraph(config, model, test_loader, device, x_sim)


  4. Evaluation
     └─ eval_subgraph(config, model, test_loader, device, full_x_sim)
        └─ graphcontrol.py:eval_subgraph()
           └─ for batch in test_loader:
              ├─ x_sim = full_x_sim[batch.original_idx]
              ├─ preds = model.forward_subgraph(batch.x, x_sim, ...)
              │  └─ (위의 forward_subgraph와 동일)
              └─ correct += (preds == batch.y).sum()


  5. 최종 결과
     └─ for i, seed in enumerate(config.seeds):
        ├─ best_acc = finetune(...)
        ├─ acc_list.append(best_acc)
        └─ print(f'Seed: {seed}, Accuracy: {best_acc}')
     
     └─ print(f"# final_acc: {mean}±{std}")


  ================================================================================
  코드 읽기 순서 추천
  ================================================================================

  📖 **초보자용 순서 (개념 이해 우선):**

  1. graphcontrol.py:main()
     └─ 전체 흐름 파악

  2. utils/transforms.py:obtain_attributes()
     └─ Condition이 무엇인지 이해

  3. models/gcc.py:GCC
     └─ Pretrained encoder 구조 이해

  4. models/gcc_graphcontrol.py:GCC_GraphControl.__init__()
     └─ ControlNet 아키텍처 이해

  5. models/gcc_graphcontrol.py:forward_subgraph()
     └─ 핵심 로직 (frozen + trainable + injection)

  6. graphcontrol.py:finetune()
     └─ 학습 과정


  📖 **디버깅용 순서 (실행 흐름 추적):**

  1. graphcontrol.py:main() [line 112]
     ↓
  2. utils/transforms.py:obtain_attributes() [line 58]
     ↓
  3. graphcontrol.py:preprocess() [line 16]
     ↓
  4. models/model_manager.py:load_model() [line 6]
     ↓
  5. models/gcc_graphcontrol.py:GCC_GraphControl.__init__() [line 12]
     ↓
  6. graphcontrol.py:finetune() [line 37]
     ↓
  7. models/gcc_graphcontrol.py:forward_subgraph() [line 74]
     ↓
  8. models/gcc.py:prepare_node_features() [line X]
     ↓
  9. models/gcc.py:GINConv.forward() [line X]
     ↓
  10. models/gcc.py:graph_readout() [line X]


  📖 **핵심만 빠르게:**

  1. models/gcc_graphcontrol.py:forward_subgraph() [line 74]
     └─ 여기가 모든 핵심!

  2. utils/transforms.py:obtain_attributes() [line 58]
     └─ Condition 생성

  3. graphcontrol.py:finetune() [line 37]
     └─ 학습 루프


  ================================================================================

  핵심 포인트:

  1. Condition은 한 번만 계산: main()에서 전체 그래프로 x_sim 계산 → 모든 epoch/batch에서 재사용
  2. Shared projection: cond_proj는 모든 레이어가 공유 (한 번만 계산)
  3. Layer-wise injection: zero_layers[0~4]를 통해 각 레이어마다 다른 zero convolution
  4. Frozen vs Trainable: Encoder는 frozen, trainable_copy + zero_layers + classifier만 학습
