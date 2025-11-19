# Design Doc: Polynomial Multi-hop PE + Node-wise scalar per-hop gate for Layerwise GraphControl

## 0. Goal & Scope

**Goal:**
The existing **LayerwiseGraphControl (where-only)** is extended with a new version that considers **"what conditions"** to use.:

1. Use **multi-scale "polynomial" positional embeddings** derived from the condition Laplacian (Polynomial PE).
2. Add a **node-wise scalar per-hop gate (called G1 gate)** at each layer to mix these multi-scale PEs per node.
3. Feed the **gated multi-scale condition** into each layer's ControlNet branch, instead of a single static condition.

**Non-goals (for this iteration):**

* Do **not** change how the condition adjacency (A') is computed.
* Do **not** change the GCC backbone architecture.
* Do **not** introduce attention, channel-wise gates, or edge-wise gates (those come later if needed).

---

## 1. Preliminaries & Assumptions

Assume the current codebase already has:

1. A **condition adjacency** (A') (feature-similarity-based discretized adjacency) built exactly as in original GraphControl.
2. A function that computes a **Laplacian PE** from (A'), via eigen-decomposition:

   * Compute normalized Laplacian: $L' = I - D'^{-1/2} A' D'^{-1/2}$
   * Compute eigenpairs: $L' = U \Lambda U^\top$
   * Use top $d_{\text{pe}}$ eigenvectors as PE, where:

     * $U \in \mathbb{R}^{N \times d_{\text{pe}}}$
     * $\Lambda = \text{diag}(\lambda_1,\dots,\lambda_{d_{\text{pe}}}) \in \mathbb{R}^{d_{\text{pe}} \times d_{\text{pe}}}$
3. A **LayerwiseGraphControl** module that:

   * Has a frozen encoder and a trainable encoder, exposed as `.layers`.
   * Currently uses a **single** condition embedding (e.g., `pe_cond`) projected once and injected into each layer via `cond_proj` and `zero_layer[l]`.

We will **modify only the condition path**; the rest stays intact.

---

## 2. Polynomial Multi-hop PE Design

### 2.1. Notation

* $A'$: condition adjacency (feature similarity graph).
* $L' = I - D'^{-1/2} A' D'^{-1/2}$: normalized Laplacian.
* $U \in \mathbb{R}^{N \times d}$: matrix of top $d$ eigenvectors.
* $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$: vector of corresponding eigenvalues.
* $K$: number of **polynomial scales** (e.g., 2 or 3).

### 2.2. Polynomial filter choice

Use **polynomial filters** over eigenvalues:

$$
f_k(\lambda) = (1 - \lambda)^k, \quad k = 1, \dots, K
$$

Rationale:

* No new hyperparameters (only integer $k$).
* Simple to implement.
* Interpretable as repeated smoothing (higher $k$ = broader effective hop / more global structure).

### 2.3. Constructing K multi-scale PEs from one eigendecomposition

1. After eigendecomposition, we have:

   * $U[:, i]$ = $i$-th eigenvector, dimension $N$.
   * $\lambda_i$ = eigenvalue for component $i$.

2. For each $k \in \{1, \dots, K\}$:

   * Compute weights:
     $$
     w^{(k)}_i = f_k(\lambda_i) = (1 - \lambda_i)^k
     $$

   * Construct **polynomial PE at scale k**:
     $$
     \text{PE}^{(k)} = U \cdot \text{diag}(w^{(k)}_1, \dots, w^{(k)}_d)
     $$
     which is simply:

     * For each node $v$:
       $$
       \text{PE}^{(k)}_v[j] = U_{v,j} \cdot w^{(k)}_j, \quad j = 1,\dots,d
       $$
       So the shape is:

       * $\text{PE}^{(k)} \in \mathbb{R}^{N \times d_{\text{pe}}}$

3. Store all of them as a tensor:

   * Either shape $(K, N, d_{\text{pe}})$, or
   * $(N, K, d_{\text{pe}})$.

   Pick one convention and stick to it. For gating, it will be useful to have easy access to the **per-node set** of PEs:

   * For a node $v$, we want:
     $$
     \{\text{PE}^{(1)}_v,\dots,\text{PE}^{(K)}_v\} \in \mathbb{R}^{K \times d_{\text{pe}}}
     $$

**Summary of Step 2:**

* **Input:** $A'$
* **Existing output:** single PE: $U$
* **New output:** multi-scale polynomial PEs:
  $$
  \text{PE\_poly}(k, v, :) = \text{PE}^{(k)}_v \quad \text{for } k=1..K,\ v=1..N
  $$

---

## 3. G1 Gate: Node-wise Scalar Per-hop Gate

We now design the **G1 gate** that, for each layer $l$ and node $v$, outputs a **scalar gate per scale k**:

$$
g^{(l)}(v) = (g^{(l)}_1(v), \dots, g^{(l)}_K(v)) \in (0, 1)^K
$$

These are **node-wise, per-hop scalars**.

### 3.1. Inputs to the gate

At layer $l$, you have:

* Current node embedding:
  $$
  h^{(l)}_v \in \mathbb{R}^{d_{\text{hid}}}
  $$
* Polynomial PEs for node $v$:
  $$
  \text{PE}^{(k)}_v \in \mathbb{R}^{d_{\text{pe}}}, \quad k = 1,\dots,K
  $$

Choose a simple input representation for the gate:

* Compute a **summary of the multi-scale PEs** for node $v$. For example:
  $$
  \bar{p}_v = \frac{1}{K} \sum_{k=1}^K \text{PE}^{(k)}_v \in \mathbb{R}^{d_{\text{pe}}}
  $$
* Concatenate this with $h^{(l)}_v$:

  $$
  z^{(l)}_v = [h^{(l)}_v \,|\, \bar{p}_v] \in \mathbb{R}^{d_{\text{hid}} + d_{\text{pe}}}
  $$

This keeps the gate **simple** while giving it both:

* current feature state (from $h^{(l)}_v$), and
* structural context (from $\bar{p}_v$).

> Alternative: if you want the gate to see the full set $\{\text{PE}^{(k)}_v\}$, you can flatten or pool differently, but for the first version, the average is sufficient.

### 3.2. Gate network structure

For each layer $l$, define a small MLP:

* Input: $z^{(l)}_v \in \mathbb{R}^{d_{\text{hid}} + d_{\text{pe}}}$
* Output: $g^{(l)}(v) \in (0,1)^K$

Conceptually:

$$
g^{(l)}(v) = \sigma\big(W^{(l)}_2 \,\phi(W^{(l)}_1 z^{(l)}_v + b^{(l)}_1) + b^{(l)}_2\big)
$$

where:

* $W^{(l)}_1 \in \mathbb{R}^{d_{\text{gate}} \times (d_{\text{hid}} + d_{\text{pe}})}$
* $W^{(l)}_2 \in \mathbb{R}^{K \times d_{\text{gate}}}$
* $\phi$ is a nonlinearity such as ReLU.
* $\sigma$ is a sigmoid (element-wise) to keep outputs in $(0,1)$.

Design choices:

* **Per-layer vs shared gate parameters:**

  * Option A (more flexible): each layer $l$ has its own gate MLP (separate parameters).
  * Option B (cheaper): share a single gate MLP across all layers.
  * For the first experiment, **per-layer gates** align with "multi-layer ControlNet" idea, but you can also start shared if parameter budget is tight.

* **Initialization:**

  * Bias in the final layer can be initialized to a small **negative value** so that initial sigmoid outputs are near 0 (turning condition off at start), or near 0.5 for neutral mixing.
  * This ensures at initialization, the model behaves close to the current LayerwiseGraphControl (condition relatively weak).

---

## 4. Combining Gated Multi-scale PE into a Single Condition per Layer

For each layer $l$ and node $v$:

1. You have $K$ polynomial PEs:
   $$
   \text{PE}^{(1)}_v, \dots, \text{PE}^{(K)}_v \in \mathbb{R}^{d_{\text{pe}}}
   $$
2. You computed gates:
   $$
   g^{(l)}_1(v), \dots, g^{(l)}_K(v) \in (0,1)
   $$

Define the **gated combined condition PE**:

$$
\widetilde{\text{PE}}^{(l)}_v = \sum_{k=1}^K g^{(l)}_k(v) \, \text{PE}^{(k)}_v \in \mathbb{R}^{d_{\text{pe}}}
$$

Interpretation:

* This is a **node-wise mixture** of K scales.
* When $g^{(l)}_k(v)$ is high, scale k contributes more.
* The gate can learn e.g.:

  * node prefers local structure (small k),
  * prefers more global structure (large k),
  * or a mix.

> Implementation note: if PE tensor is $(N, K, d_{\text{pe}})$ and gate is $(N, K)$, you can think of this as a **weighted sum over the K axis**.

---

## 5. Integrating into LayerwiseGraphControl

We now describe how to plug the **gated condition** into the **existing layerwise ControlNet**.

### 5.1. Current behavior (recap)

Currently, LayerwiseGraphControl likely does something like:

1. Compute a **single condition PE**: `pe_cond ∈ ℝ^{N×d_pe}` from $A'$.
2. Project once:

   * `cond_h = cond_proj(pe_cond) ∈ ℝ^{N×d_hid}`.
3. For each layer $l$:

   * Frozen branch:
     $$
     h^{(l+1)}_{\text{frozen}} = \text{layer\_frozen}^{(l)}(h^{(l)}_{\text{frozen}}, \text{edge\_index})
     $$
   * Control branch:
     $$
     h^{(l+1)}_{\text{ctrl}} = \text{layer\_ctrl}^{(l)}(h^{(l)}_{\text{ctrl}} + \text{cond\_h}, \text{edge\_index})
     $$
   * Residual injection:
     $$
     h^{(l+1)}_{\text{frozen}} \mathrel{+}= \text{zero\_layer}^{(l)}(h^{(l+1)}_{\text{ctrl}})
     $$

where `zero_layer` output is initialized to zero.

### 5.2. New behavior

We will **replace the static `cond_h`** with a **per-layer, per-node dynamic conditioned vector** built from polynomial PE and gates.

Steps at each forward pass:

1. **Precompute polynomial PEs** (once per graph):

   * Compute $U, \Lambda$ from $A'$.
   * Compute $\text{PE}^{(k)}$ for all $k=1..K$.
   * Stack as `PE_poly` with shape `[N, K, d_pe]`.

2. **Initialize branch states**:

   * `h_frozen` and `h_ctrl` start from whatever you used before (e.g., original structural PE).

3. **For each layer $l$:**

   For each node $v$ (vectorized in code):

   1. Build gate input:

      * $h^{(l)}_v$ from frozen or control branch (you can choose one – simplest: use frozen branch).
      * $\bar{p}_v = \frac{1}{K} \sum_k \text{PE}^{(k)}_v$

      Then:
      $$
      z^{(l)}_v = [h^{(l)}_v \,|\, \bar{p}_v]
      $$

   2. Compute gates:
      $$
      g^{(l)}(v) = \text{GateMLP}^{(l)}(z^{(l)}_v) \in (0,1)^K
      $$

   3. Compute combined condition PE:
      $$
      \widetilde{\text{PE}}^{(l)}_v = \sum_{k=1}^K g^{(l)}_k(v) \, \text{PE}^{(k)}_v
      $$
      This gives $\widetilde{\text{PE}}^{(l)} \in \mathbb{R}^{N \times d_{\text{pe}}}$.

   4. Project combined PE into hidden dimension (reuse the idea of `cond_proj`, can be shared across layers):

      $$
      \text{cond\_h}^{(l)}_v = \text{cond\_proj}(\widetilde{\text{PE}}^{(l)}_v) \in \mathbb{R}^{d_{\text{hid}}}
      $$

      You can:

      * Use a **single shared `cond_proj`** for all layers, or
      * Have **per-layer `cond_proj^{(l)}`**; for the first version, shared is usually enough.

   5. Control branch update:

      * Use this per-layer condition as an additive input (similar to current code):

        $$
        h^{(l+1)}_{\text{frozen}} = \text{layer\_frozen}^{(l)}(h^{(l)}_{\text{frozen}}, \text{edge\_index})
        $$

        $$
        h^{(l+1)}_{\text{ctrl}} = \text{layer\_ctrl}^{(l)}(h^{(l)}_{\text{ctrl}} + \text{cond\_h}^{(l)}, \text{edge\_index})
        $$

      * Then residual injection unchanged:

        $$
        h^{(l+1)}_{\text{frozen}} \mathrel{+}= \text{zero\_layer}^{(l)}(h^{(l+1)}_{\text{ctrl}})
        $$

Initialization:

* `cond_proj` weights and biases: **zero-initialize** as in original GraphControl (so initially, condition has no effect).
* `zero_layer[l]` already zero-initialized.
* Gate MLP:

  * Initialize last-layer bias so that $g^{(l)}_k(v)$ starts near a small value (e.g., 0 or 0.5).
  * This ensures early training behaves close to "no extra condition" or a mild average.

---

## 6. Hyperparameters & Practical Choices

Recommended starting settings:

* **Number of polynomial scales ($K$)**:

  * Start with **K = 2 or 3**:

    * e.g., $k = 1, 2$ or $k = 1, 2, 3$.
* **Gate hidden dimension ($d_{\text{gate}}$)**:

  * Something small relative to $d_{\text{hid}}$, e.g. 32 or 64.
* **Which branch's $h^{(l)}$ to use as gate input**:

  * Use **frozen branch** $h^{(l)}_{\text{frozen}}$ for stability, or use `detach()` of control branch if needed.
* **Shared vs per-layer gate**:

  * For full 5.1 spirit, use **per-layer gate**.
  * If parameter count is a concern, start with a shared gate and later extend.

---

## 7. Sanity Checks

Before running full experiments, please note and check the following:

1. **Initialization sanity:**

   * With gate outputs forced to zero (or cond_proj zero-init), model output ≈ current LayerwiseGraphControl output.
   * With all gates = 1 and `cond_proj` behaving like before, you should roughly recover "single-scale where-only" behavior if you set all $\text{PE}^{(k)}$ equal or restrict to k=1.

2. **Ablation toggles:**

   * Disable gate (fix $g^{(l)}_k(v) = 1/K$ for all nodes/layers) → this tests whether multi-scale PEs alone help.
   * Vary K (1 vs 2 vs 3) to see effect of multi-scale vs single-scale.

3. **Monitoring:**

   * Log average gate values per layer and per hop over training.
   * Check if, on heterophilic datasets, deeper layers or some hops get higher weights.

---

## 8. Summary

* Modeling multi-hop polynomial structural signals
* Learning node-wise hop weighting per layer
* Supplying dynamic, context-aware condition embeddings