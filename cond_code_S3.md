## Condition CODE 'S3':: PPR-based Sparse Structural Condition Adjacency for Layerwise GraphControl

---

## Goal

Introduce a **PPR-based (Personalized PageRank) sparse condition adjacency** as an S-type (structure-only) alternative to `x_sim` for GraphControl / Layerwise GraphControl.

* Replace the current **feature-similarity adjacency `x_sim`** with a **PPR-derived sparse adjacency** `C_ppr`.
* Keep the rest of **Original GraphControl** and **Layerwise GraphControl** architecture unchanged.
* Target use: **heterophilic datasets**, but implementation should remain generic.

---

## High-Level Idea

* Start from the (normalized) base adjacency `A` used by the GNN.
* Compute an approximation of the Personalized PageRank matrix `Π` (N×N).
* For each node `i`, retain only the **top-k highest PPR neighbors** `j`, forming a sparse matrix `C_ppr`.
* Use `C_ppr` wherever `x_sim` (feature similarity adjacency) is currently used as the **condition graph** in GraphControl.

This gives a **global, role-aware structural cond graph** that does **not** depend on node features.

---

## API / Config Changes

1. **New cond type option**

Add a new CLI/arg option:

```bash
--cond_type s3_ppr
```

When `cond_type == "s3_ppr"`:

* Ignore feature-based similarity (`x_sim`).
* Construct `C_ppr` from adjacency and use it as the condition adjacency.

2. **Hyperparameters** (start with reasonable defaults)

* `--ppr_alpha` (float, default: 0.15)
  Teleport (restart) probability.
* `--ppr_topk` (int, default: 32)
  Top-k PPR neighbors per node to keep.
* `--ppr_symmetric` (bool, default: False)
  Whether to symmetrize `C_ppr` (optional).
* `--ppr_normalize` (bool, default: True)
  Whether to row-normalize `C_ppr` before usage.

---

## Data / Preprocessing

Assume we already have:

* Sparse adjacency matrix `A` (N×N), same convention as the base GNN (with self-loops if used).
* Number of nodes `N`.

### Step 1: Build transition matrix for random walk

We need a stochastic matrix `T` representing random walk transitions.

For undirected graphs:

```python
# scipy.sparse
deg = np.array(A.sum(axis=1)).flatten()
inv_deg = 1.0 / np.maximum(deg, 1e-12)
D_inv = scipy.sparse.diags(inv_deg)
T = D_inv @ A  # row-stochastic: T[i, j] = P(j | i)
```

If the base GNN uses a particular normalization, keep it consistent as much as possible.

### Step 2: Approximate Personalized PageRank

We want, for each node i, a PPR vector πᵢ satisfying:

[
\pi_i = \alpha e_i + (1 - \alpha) \pi_i T
]

Instead of computing the full dense matrix, we:

* either run a **push algorithm** / localized PPR for each node, or
* run a **power iteration** on all nodes and use top-k truncation.

For simplicity in the spec, assume a function:

```python
def compute_ppr_topk(T, alpha, topk):
    """
    T: scipy.sparse row-stochastic matrix (N x N)
    alpha: teleport probability
    topk: keep top-k entries per row
    Returns: sparse matrix PPR_topk (N x N)
             where each row i has at most top-k non-zero entries.
    """
    ...
```

Possible implementation outline:

* Initialize `P = alpha * I` (or uniform).
* For k=1..K iterations:

  * `P = alpha * I + (1 - alpha) * P @ T`
* After convergence/iterations:

  * For each row i, keep top-k values.

The implementation details can be optimized later; core requirement is:

> Output `PPR_topk` as a sparse matrix where row i contains the top-k PPR scores πᵢ(j).

### Step 3: Sparsify and post-process to build C_ppr

Once `PPR_topk` is computed:

```python
C_ppr = PPR_topk.copy()
```

Optionally:

* **Symmetrize** (if desired):

```python
if ppr_symmetric:
    C_ppr = 0.5 * (C_ppr + C_ppr.T)
```

* **Remove self-loops** in condition graph (optional):

```python
C_ppr.setdiag(0)
C_ppr.eliminate_zeros()
```

* **Normalize rows** (recommended for stability):

```python
if ppr_normalize:
    row_sum = np.array(C_ppr.sum(axis=1)).flatten()
    inv_row = 1.0 / np.maximum(row_sum, 1e-12)
    D_inv = scipy.sparse.diags(inv_row)
    C_ppr = D_inv @ C_ppr
```

Result: `C_ppr` is a **sparse N×N matrix** with at most `ppr_topk` non-zeros per row.

---

## Integration into GraphControl

We assume:

* Existing GraphControl code already accepts some `cond_adj` (e.g., `x_sim`) as adjacency for the control branch.
* It converts this adjacency into edge indices / normalized forms for use in message passing within the control branch.

### Step 1: Construction point

Locate the place where `x_sim` (or other cond adjacencies) are currently constructed:

Example:

```python
if cond_type == "feature_sim":
    cond_adj = build_feature_similarity_adj(features, ...)
elif cond_type == "khop_mixed":
    cond_adj = build_khop_mixed_adj(...)
...
```

Extend it:

```python
elif cond_type == "s3_ppr":
    cond_adj = build_ppr_cond_adj(A, ppr_alpha, ppr_topk, ppr_symmetric, ppr_normalize)
```

### Step 2: Implement `build_ppr_cond_adj`

Define a helper:

```python
def build_ppr_cond_adj(A, alpha=0.15, topk=32, symmetric=False, normalize=True):
    """
    A: scipy.sparse adjacency matrix (N x N), same as base GNN
    Returns: scipy.sparse matrix C_ppr (N x N), sparse PPR-based cond adjacency
    """
    # 1. build transition matrix
    deg = np.array(A.sum(axis=1)).flatten()
    inv_deg = 1.0 / np.maximum(deg, 1e-12)
    D_inv = scipy.sparse.diags(inv_deg)
    T = D_inv @ A  # row-stochastic

    # 2. approximate PPR with top-k per row
    PPR_topk = compute_ppr_topk(T, alpha, topk)

    C_ppr = PPR_topk.copy()

    # 3. optional symmetrization
    if symmetric:
        C_ppr = 0.5 * (C_ppr + C_ppr.T)

    # 4. remove self-loops
    C_ppr.setdiag(0)
    C_ppr.eliminate_zeros()

    # 5. optional row normalization
    if normalize:
        row_sum = np.array(C_ppr.sum(axis=1)).flatten()
        inv_row = 1.0 / np.maximum(row_sum, 1e-12)
        D_inv = scipy.sparse.diags(inv_row)
        C_ppr = D_inv @ C_ppr

    return C_ppr
```

`compute_ppr_topk` can initially be a simple batched power-iteration approximation; later it can be optimized.

### Step 3: Use C_ppr as cond adjacency

Once `cond_adj = C_ppr` is created, **no further architecture change** is needed:

* Whatever function currently transforms `cond_adj` to a PyTorch sparse tensor or `edge_index` can be reused:

```python
cond_edge_index, cond_edge_weight = sparse_to_edge_index(cond_adj)
```

* The control branch will now use PPR-based structure instead of feature similarity.

---

## Normalization & Stability Notes

* Row-normalization of `C_ppr` is recommended so that it behaves similarly to a transition matrix and avoids exploding magnitudes inside the controller.
* If GraphControl already applies its own normalization to cond adjacency, ensure you **do not double-normalize** in an inconsistent way. Best practice:

  * Either normalize in `build_ppr_cond_adj` and skip later,
  * Or pass raw `C_ppr` and keep existing normalization logic.

---

## Summary for Codex

1. Add `cond_type == "s3_ppr"` in the configuration / argument parsing.

2. Implement:

   ```python
   build_ppr_cond_adj(A, alpha, topk, symmetric, normalize)
   ```

   * Build row-stochastic `T` from `A`.
   * Approximate PPR with `compute_ppr_topk(T, alpha, topk)`.
   * Optional symmetrization, remove self-loops, row-normalize.
   * Return sparse `C_ppr` (N×N).

3. When `cond_type == "s3_ppr"`:

   * **Do not** construct `x_sim` (feature similarity).
   * Use `C_ppr` as `cond_adj` for the GraphControl controller branch.

4. Keep the rest of GraphControl / Layerwise GraphControl unchanged; only the source of the condition adjacency is swapped.

This implements **S3: a PPR-based sparse structural condition adjacency** as a drop-in replacement for `x_sim`, suitable for heterophilic GraphControl experiments.
