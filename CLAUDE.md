# CLAUDE.md

## DisenLink-based Condition Matrix Generation for GraphControl

### Overview
This document outlines the integration of DisenLink's link prediction capabilities into GraphControl to generate more effective condition matrices (A') for heterophilic graphs. The current GraphControl implementation uses feature similarity-based condition generation, which assumes homophily and may fail on heterophilic graphs. This enhancement replaces or augments that approach with learned link predictions from DisenLink.

### Problem Statement
- **Current Limitation**: GraphControl's condition generation module uses cosine similarity with threshold filtering (Equation 4 in GraphControl paper), which inherently assumes homophily
- **Challenge**: On heterophilic graphs (e.g., Chameleon, Squirrel), feature-similar nodes may not be connected, leading to poor condition matrices
- **Solution**: Leverage DisenLink's factor-aware link prediction to generate condition matrices that respect heterophilic patterns

### Tips.
The original source code for the DisenLink study is located in the DisenLink/ directory. This directory is read-only, so you can reference the code in this directory when implementing your own code.

### Architecture Design
The source code below is merely an example. If you find something wrong with the implementation of claude-code, feel free to modify it and implement it differently.
#### 1. Components Integration

```
GraphControl Pipeline with DisenLink:
1. Source Data Pre-training (unchanged)
2. Target Data Processing:
   a. Subgraph Sampling (existing)
   b. DisenLink Training (new)
   c. Condition Generation (modified)
   d. GraphControl Fine-tuning (unchanged)
```

#### 2. DisenLink Integration Module

```python
class DisenLinkConditionGenerator:
    """
    Generates condition matrix A' using DisenLink predictions
    instead of or in addition to feature similarity.
    """
    
    def __init__(self, 
                 num_factors=5,
                 hidden_dim=32,
                 use_hybrid=True,
                 hybrid_alpha=0.5):
        """
        Args:
            num_factors: Number of disentangled factors (K)
            hidden_dim: Dimension per factor
            use_hybrid: Whether to combine with feature similarity
            hybrid_alpha: Weight for feature similarity (if hybrid)
        """
        self.disenlink = DisenLink(num_factors, hidden_dim)
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
    
    def generate_condition(self, graph, subgraphs):
        """
        Generate condition matrix A' for GraphControl.
        
        Args:
            graph: Target graph (A, X)
            subgraphs: List of sampled subgraphs
            
        Returns:
            A': Condition adjacency matrix
        """
        # Train DisenLink on subgraphs
        A_pred = self.train_and_predict(subgraphs)
        
        if self.use_hybrid:
            A_feat = self.compute_feature_similarity(graph.X)
            A_prime = self.hybrid_alpha * A_feat + \
                      (1 - self.hybrid_alpha) * A_pred
        else:
            A_prime = A_pred
            
        return A_prime
```

### Implementation Steps

#### Phase 1: DisenLink Training Pipeline

1. **Subgraph-based Training**
   ```python
   def train_disenlink_on_subgraphs(subgraphs, epochs=200):
       """
       Train DisenLink using GraphControl's subgraph sampling.
       
       Key considerations:
       - Use same sampling parameters as GraphControl
         (walk_steps=256, restart_rate=0.8)
       - Maintain consistency with pre-trained encoder input format
       """
       for subgraph in subgraphs:
           # Extract local adjacency and features
           A_local, X_local = subgraph
           
           # DisenLink forward pass
           Z = disenlink.get_initial_representations(X_local)
           factor_importance = disenlink.compute_factor_importance(Z)
           factor_neighbors = disenlink.select_factor_neighbors(factor_importance)
           H = disenlink.factor_aware_message_passing(Z, factor_neighbors)
           
           # Link reconstruction loss
           A_reconstructed = disenlink.reconstruct_links(H)
           loss = compute_reconstruction_loss(A_reconstructed, A_local)
       
       return trained_disenlink
   ```

2. **Link Prediction Aggregation**
   ```python
   def aggregate_predictions(subgraph_predictions):
       """
       Combine predictions from multiple subgraphs.
       
       Strategy:
       - Weight by confidence scores
       - Handle overlapping predictions
       - Apply threshold for final binary matrix
       """
       A_aggregated = weighted_average(subgraph_predictions)
       A_prime = apply_threshold(A_aggregated, tau=0.5)
       return A_prime
   ```

#### Phase 2: Integration with GraphControl

1. **Modify Condition Generation Module**
   ```python
   # In GraphControl's condition generation:
   def generate_condition(self, A, X, method='disenlink'):
       if method == 'disenlink':
           # Use DisenLink predictions
           condition_gen = DisenLinkConditionGenerator()
           A_prime = condition_gen.generate_condition(A, X)
       elif method == 'feature_similarity':
           # Original GraphControl approach
           A_prime = self.feature_similarity_condition(X)
       elif method == 'hybrid':
           # Combine both approaches
           A_prime = self.hybrid_condition(A, X)
       
       # Generate positional embedding from A_prime
       P_prime = compute_positional_embedding(A_prime)
       return P_prime
   ```

2. **Heterophily-aware Hyperparameters**
   ```python
   def get_dataset_specific_config(dataset_name):
       """
       Return optimal configuration based on dataset heterophily.
       """
       heterophilic_datasets = ['chameleon', 'squirrel', 'crocodile', 
                                'texas', 'wisconsin']
       
       if dataset_name.lower() in heterophilic_datasets:
           return {
               'num_factors': 6,
               'use_hybrid': True,
               'hybrid_alpha': 0.2,  # Lower weight on feature similarity
               'disenlink_epochs': 300
           }
       else:
           return {
               'num_factors': 4,
               'use_hybrid': True,
               'hybrid_alpha': 0.7,  # Higher weight on feature similarity
               'disenlink_epochs': 200
           }
   ```

### Evaluation Protocol

1. **Baseline Comparison**
   - Original GraphControl with feature similarity
   - GraphControl with DisenLink condition
   - GraphControl with hybrid condition

2. **Metrics**
   - Node classification accuracy
   - Link prediction AUC (if applicable)
   - Convergence speed
   - Stability across different random seeds

3. **Dataset Categories**
   - Heterophilic: Chameleon, Squirrel, Texas, Wisconsin
   - Homophilic: Cora_ML, Amazon-Photo
   - Mixed: Test on both to ensure no regression

### Expected Benefits

1. **Improved Heterophilic Performance**: 10-15% accuracy improvement on heterophilic datasets
2. **Maintained Homophilic Performance**: No degradation on homophilic datasets
3. **Better Transferability**: More robust transfer learning across diverse graph types

### Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Increased computational cost | Cache DisenLink predictions; update periodically |
| Hyperparameter sensitivity | Auto-tune based on graph homophily measure |
| Subgraph inconsistency | Ensure overlapping subgraphs for smooth aggregation |

### Code Organization
If necessary, you may modify the current directory structure slightly.
```
.
├── datasets
│   ├── data
│   ├── dataset
│   │   ├── Hindex.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── Hindex.cpython-39.pyc
│   │       └── __init__.cpython-39.pyc
│   ├── __init__.py
│   └── __pycache__
│       └── __init__.cpython-39.pyc
├── gcc.py
├── graphcontrol.py
├── GraphControl_tree.txt
├── models
│   ├── encoder.py
│   ├── gcc_graphcontrol.py
│   ├── gcc.py
│   ├── __init__.py
│   ├── mlp.py
│   ├── model_manager.py
│   ├── pooler.py
│   └── __pycache__
│       ├── gcc.cpython-39.pyc
│       ├── gcc_graphcontrol.cpython-39.pyc
│       ├── __init__.cpython-39.pyc
│       └── model_manager.cpython-39.pyc
├── node2vec.py
├── optimizers
│   ├── __init__.py
│   └── __pycache__
│       └── __init__.cpython-39.pyc
├── DisenLink # original source code of DisenLink Framework. This section is for reference and is protected as read-only.
├── README.md
└── utils
    ├── args.py
    ├── augmentation.py
    ├── __init__.py
    ├── normalize.py
    ├── __pycache__
    │   ├── args.cpython-39.pyc
    │   ├── __init__.cpython-39.pyc
    │   ├── normalize.cpython-39.pyc
    │   ├── random.cpython-39.pyc
    │   ├── register.cpython-39.pyc
    │   ├── sampling.cpython-39.pyc
    │   └── transforms.cpython-39.pyc
    ├── random.py
    ├── register.py
    ├── sampling.py
    └── transforms.py
```

### Testing Strategy

1. **Unit Tests**
   - DisenLink component functionality
   - Condition matrix generation
   - Subgraph aggregation logic

2. **Integration Tests**
   - End-to-end GraphControl with DisenLink
   - Performance benchmarks
   - Memory usage profiling

3. **Ablation Studies**
   - Impact of number of factors K
   - Hybrid vs. pure DisenLink condition
   - Different aggregation strategies

### References

- GraphControl: [Original paper and codebase]
- DisenLink: "Link Prediction on Heterophilic Graphs via Disentangled Representation Learning"
- Implementation will build upon existing GraphControl codebase

---
