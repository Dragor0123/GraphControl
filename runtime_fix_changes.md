# Runtime Error Fix Summary

## Context
- Dataset: Chameleon  
- Error: `RuntimeError: The size of tensor a (10920) must match the size of tensor b (21840)` raised inside `torch_sparse.sample` during subgraph collection.
- Root cause: The preset masks in the dataset are 2-D (multiple split columns). Passing the entire mask led to duplicated node indices, which then produced inconsistent tensor sizes during neighbor sampling.

## Modifications
1. **`graphcontrol.py`**
   - Collapse multi-column boolean masks onto the requested split column before preprocessing when the dataset does not use random splits.
   - Ensures `collect_subgraphs` receives a flat list of train/test node IDs.

2. **`utils/sampling.py`**
   - Symmetrize edges and add self-loops for isolated nodes prior to building the `SparseTensor`.
   - Guarantees every selected node has at least one neighbor so the random-walk sampler operates on consistent adjacency data.

These adjustments remove the mask duplication issue and prevent the shape mismatch error during random walk sampling.
