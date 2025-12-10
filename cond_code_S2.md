## Condition CODE 'S2': Node-wise Scalar Structural Condition (Degree + PageRank) for Layerwise GraphControl

---

## Goal

Introduce a **simple, feature-independent, node-wise structural condition** for GraphControl / Layerwise GraphControl based on:

* node degree
* PageRank

This condition should:

* be **purely structural (S-type)**, i.e., independent of node features and labels
* be easy to compute and inject
* modulate the **control branch** (gates/scales) without changing the base GNN architecture

The implementation must be generic (any dataset), but the target use case is **heterophilic datasets** with 5 trainable copy layers.

---

## High-Level Idea

For each node (v):

1. Compute:

   * degree: ( \text{deg}(v) )
   * PageRank: ( \text{PR}(v) )

2. Build a 2D structural condition vector:
   [
   c_v = \big[\ \tilde{d}_v,\ \tilde{pr}_v\ \big]
   ]
   where (\tilde{d}_v, \tilde{pr}_v) are normalized / scaled versions of degree and PageRank.

3. For each GraphControl layer (\ell):

   * Give the controller branch access to `c_v` (for all nodes)
   * e.g., concatenate `[h_v^{(\ell)}, c_v]` as controller input,
     or use `c_v` to compute FiLM-like gates/scales.

4. No change to the base GNN message passing.
   Only the **control branch** is conditioned on these structural scalars.

---

## API / Config Changes

1. **New cond type option**

Add a new CLI/arg option:

```bash
--cond_type s2_struct_scalar
```

When `cond_type == "s2_struct_scalar"`:

* Skip feature-similarity cond adjacency (`x_sim`) entirely.
* Compute **node-wise structural condition features** from adjacency.
* Pass these structural node features to the GraphControl controller branch in all trainable copy layers.

2. **Optional hyperparameters** (can start as constants)

* `--pr_alpha` (float, default: 0.85)
  teleport probability for PageRank
* `--deg_log` (bool, default: True)
  whether to use log-degree
* `--cond_struct_norm` (str, default: "standard")
  normalization type: `"none"`, `"standard"`, or `"minmax"` (optional, can be hardcoded to one scheme initially).

---

## Data / Preprocessing

Assume we already have:

* Sparse adjacency matrix `A` (same as base GNN, including self-loops if used).
* Number of nodes `N`.

### Step 1: Degree

Compute node degree vector:

```python
# If using scipy.sparse
deg = np.array(A.sum(axis=1)).flatten()  # shape [N]
```

Optionally apply log transform:

```python
if use_log_degree:
    deg = np.log1p(deg)  # log(1 + deg)
```

### Step 2: PageRank

Compute PageRank scores using adjacency `A`:

* Use a standard PageRank implementation (`networkx.pagerank`, custom power iteration, etc.)
* Make sure the same adjacency convention as the GNN is used (directed vs undirected, self-loops, etc.).

Pseudo:

```python
# Example using networkx (if allowed)
# G = nx.from_scipy_sparse_array(A)  # undirected
# pr_dict = nx.pagerank(G, alpha=pr_alpha)
# pr = np.array([pr_dict[i] for i in range(N)])

# Or implement custom power-iteration PageRank over A
```

Result: `pr` is a vector of length `N`.

### Step 3: Normalization

To avoid scale issues between deg and pr, normalize both:

Example: **standardization** (per-feature):

```python
def standardize(x):
    mean = x.mean()
    std = x.std() + 1e-12
    return (x - mean) / std

deg_norm = standardize(deg)
pr_norm  = standardize(pr)
```

Alternative: min-max or no-norm, but pick **one scheme and keep it consistent**.

### Step 4: Build structural cond tensor

Stack into a 2D node feature tensor:

```python
# shape [N, 2]
cond_struct = np.stack([deg_norm, pr_norm], axis=-1)
```

Convert to torch:

```python
cond_struct = torch.from_numpy(cond_struct).float().to(device)
```

This `cond_struct` should be accessible in the model for all control layers.

---

## Integration into GraphControl

Assume the current GraphControl / Layerwise GraphControl:

* has a controller branch per layer:

  * takes some input (usually node embeddings (h^{(\ell)}), possibly other cond info)
  * outputs gates/scales/residual mods for that layer

### Step 1: Model signature / fields

Update the GraphControl model to accept an optional `struct_cond` tensor:

```python
class GraphControl(...):
    def __init__(..., use_struct_cond=False, ...):
        self.use_struct_cond = use_struct_cond
        ...
    
    def forward(self, x, adj, ..., struct_cond=None):
        # x: [N, d_in]
        # struct_cond: [N, 2] or None
        ...
```

When `cond_type == "s2_struct_scalar"`, set `use_struct_cond=True` and pass `struct_cond` from the training script.

### Step 2: Layerwise controller input modification

Inside each layer `ℓ`’s controller:

Currently, controller input might be something like:

```python
ctrl_input = h_l  # [N, d_h]
```

Modify to:

```python
if self.use_struct_cond and struct_cond is not None:
    # Concatenate along feature dimension: [N, d_h + 2]
    ctrl_input = torch.cat([h_l, struct_cond], dim=-1)
else:
    ctrl_input = h_l
```

Then feed `ctrl_input` into the existing MLP / gating mechanism:

```python
gates = self.controller_mlp(ctrl_input)
# or whatever the current code does
```

No other logic change is required; the controller simply sees 2 extra structural features per node.

> Note: Keep the concatenation shape consistent for all layers
> (i.e., use the same `struct_cond` for each layer; it’s static per node).

### Step 3: Disable x_sim / cond adjacency for S2

When `cond_type == "s2_struct_scalar"`:

* Do **not** build or use `x_sim` or any cond adjacency.
* Only pass `struct_cond` into the model and let the controller modulate behavior based on it.

So in the training / model setup:

```python
if args.cond_type == "s2_struct_scalar":
    struct_cond = build_structural_cond(A)  # deg+pr
    model = GraphControl(..., use_struct_cond=True, ...)
else:
    struct_cond = None
    ...
```

and in the forward pass:

```python
logits = model(x, adj, ..., struct_cond=struct_cond)
```

---

## Helper Function

Implement a helper:

```python
def build_structural_cond(A, use_log_degree=True, pr_alpha=0.85, norm_type="standard"):
    """
    A: scipy.sparse adjacency matrix (NxN)
    Returns: torch.FloatTensor of shape [N, 2] (deg_norm, pr_norm)
    """
    # 1. degree
    deg = np.array(A.sum(axis=1)).flatten()
    if use_log_degree:
        deg = np.log1p(deg)

    # 2. pagerank
    pr = compute_pagerank(A, alpha=pr_alpha)  # implement separately

    # 3. normalization
    deg_norm = normalize_vector(deg, norm_type)
    pr_norm  = normalize_vector(pr,  norm_type)

    cond_struct = np.stack([deg_norm, pr_norm], axis=-1)
    cond_struct = torch.from_numpy(cond_struct).float()
    return cond_struct
```

`compute_pagerank` and `normalize_vector` can be small utility functions.

---

## Summary for Codex

1. Add a new `cond_type == "s2_struct_scalar"`.
2. Implement `build_structural_cond(A, ...)`:

   * compute degree (optional log) and PageRank
   * normalize both
   * return `[N, 2]` tensor (deg_norm, pr_norm).
3. Extend `GraphControl`:

   * add a `use_struct_cond` flag and a `struct_cond` argument in `forward`.
   * in each layer, if `use_struct_cond`:

     * concatenate `struct_cond` with node embeddings as controller input.
4. When `cond_type == "s2_struct_scalar"`:

   * **do not** construct `x_sim` or other cond adjacency.
   * always pass `struct_cond` to the model.
5. Keep the base GNN and the controller architecture intact; only the controller input is augmented with 2 scalar structural features per node.

This implements **S2: a minimal, purely structural, node-wise scalar condition (deg + PageRank)** for all trainable GraphControl layers.
