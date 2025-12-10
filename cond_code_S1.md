## Condition CODE 'S1': 2-hop Structure-based Condition for Layerwise GraphControl
---

## Goal

Replace the current **feature-similarity-based condition graph `x_sim`** with a **purely structural 2-hop condition graph** based on (A^2) (two-hop adjacency), while keeping the rest of **Original GraphControl** and **Layerwise GraphControl** unchanged.

This is for experiments on heterophilic datasets (Chameleon, Squirrel, Actor), but the implementation should be generic.

---

## High-Level Idea

* Let `A` be the original sparse adjacency matrix (including self-loops if the base model uses them).
* Build a new sparse matrix `C_2hop` from `A^2`, representing **2-hop connectivity**.
* Use `C_2hop` wherever `x_sim` (feature similarity adjacency) is currently used as the **condition graph** in GraphControl.
* Optionally sparsify / threshold `C_2hop` to avoid dense blow-up.

---

## API / Config Changes

1. **New cond type option**

   * Add a new CLI/arg option, for example:

     ```bash
     --cond_type s1_2hop
     ```
   * When `cond_type == "s1_2hop"`, the code should:

     * **ignore** feature-based similarity computation
     * construct the S1 2-hop condition graph from `A`.

2. **Optional hyperparameters** (with safe defaults)

   * `--two_hop_threshold` (float, default: 0.0)

     * if > 0, keep only entries in `A^2` with value ≥ threshold.
   * `--two_hop_topk` (int, default: 0)

     * if > 0, for each node keep only top-k largest 2-hop neighbors.

If you want to keep it minimal, implement these as constants in code first, then expose as flags later.

---

## Data / Preprocessing

Assume we already have:

* `A` as a sparse adjacency (e.g., `torch.sparse_coo_tensor` or `scipy.sparse.csr_matrix`)
* Number of nodes `N`.

### Step 1: Define A with self-loops consistent with base GNN

* If the base encoder adds self-loops (common), ensure `A` used here is **the same version**:

  ```python
  A = base_adj_with_self_loops  # same as GNN uses
  ```

### Step 2: Compute 2-hop adjacency

Conceptually:

* ( A_2 = A @ A )
* This counts the number of length-2 walks between nodes.

Implementation sketch (sparse):

* If using `scipy.sparse`:

  ```python
  A2 = A @ A  # still sparse
  ```
* If using PyTorch sparse, either:

  * convert to scipy for this product, or
  * use `torch.sparse.mm(A, A)` if `A` is in compatible format.

### Step 3: Remove 1-hop edges and self-loops (optional but recommended)

We want a **pure 2-hop view**, not a superset of 1-hop edges:

* Remove self-loops from `A2` if they exist.
* Optionally remove nodes that were already directly connected in `A`.

Sketch:

```python
# A, A2 are scipy.sparse CSR
A_bin = (A > 0).astype(int)
A2_bin = (A2 > 0).astype(int)

# remove 1-hop connections from A2
A2_bin = A2_bin.multiply((A_bin == 0))  # keep only pure 2-hop edges

# remove self-loops
A2_bin.setdiag(0)
A2_bin.eliminate_zeros()
```

If you **do want** to include 1-hop edges as well, skip the subtraction step and only remove self-loops.

### Step 4: Sparsify if needed

If graph is large / dense:

* Optional thresholding:

  ```python
  if two_hop_threshold > 0.0:
      A2_th = A2.multiply(A2 >= two_hop_threshold)
  else:
      A2_th = A2
  ```

* Optional top-k per row (if `two_hop_topk > 0`):

  * For each node `i`, select top-k neighbors `j` by A2 weight (or just arbitrarily if all equal).
  * Zero out others.

Result: a sparse matrix `C_2hop` with shape `[N, N]`.

---

## Integration into GraphControl

Assume current code:

* builds `x_sim` adjacency from features,
* passes it to GraphControl as some condition adjacency (e.g., `adj_sim`).

### Replace x_sim with C_2hop when `cond_type == "s1_2hop"`

1. **Construction point**

Locate the block that currently does:

```python
if cond_type == "feature_sim" or cond_type == "khop_mixed":
    x_sim = build_feature_similarity_adj(features, ...)
    cond_adj = x_sim
```

Modify to:

```python
if cond_type == "s1_2hop":
    C_2hop = build_two_hop_adj(A, two_hop_threshold, two_hop_topk)
    cond_adj = C_2hop
elif cond_type == "feature_sim":
    cond_adj = build_feature_similarity_adj(...)
...
```

2. **Downstream usage**

Wherever `cond_adj` (previously `x_sim`) is:

* used in the GraphControl controller branch,
* converted to edge index format,
* or normalized,

keep the logic identical, just with `cond_adj = C_2hop`.

No other changes in the GraphControl architecture are needed.

---

## Normalization

To keep behavior stable:

* Normalize `C_2hop` similarly to the base adjacency normalization (e.g., row-normalized):

```python
# e.g. row-normalization
row_sum = np.array(C_2hop.sum(axis=1)).flatten()
inv_row = 1.0 / np.maximum(row_sum, 1e-12)
D_inv = scipy.sparse.diags(inv_row)
C_norm = D_inv @ C_2hop
```

Then feed `C_norm` into GraphControl as the conditional adjacency.

---

## Summary for Codex

1. Add a new `cond_type == "s1_2hop"`.
2. Implement `build_two_hop_adj(A, two_hop_threshold, two_hop_topk)`:

   * sparse multiplication `A2 = A @ A`
   * remove self-loops (and optionally 1-hop edges)
   * optional threshold / top-k pruning
   * optional row-normalization
3. When `cond_type == "s1_2hop"`, use this `C_2hop` as `cond_adj` instead of `x_sim`.
4. Leave the rest of GraphControl / Layerwise GraphControl code unchanged.

This is the minimal S-type (2-hop) condition needed for the next experiments on heterophilic datasets.
