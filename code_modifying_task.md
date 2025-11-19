# Layerwise GraphControl Implementation Guide.

**Goal**

Implement a *layerwise* variant of GraphControl that:
- Keeps the **condition (“What”) exactly the same** as the original GraphControl (same feature-similarity-based PE).
- Changes only **“Where”** the condition is injected:  
  from a single global ControlNet → to **per-layer** control branches.
- Uses the **same condition PE at every layer** for the first experiment.

This is an *experimental* variant; do not modify how the condition is computed.

**Warning**
All Python code in this document is for illustrative purposes only and may differ in detail from the code written in the current project folder. Please note that these examples are for illustrative purposes only.

---

## 0. High-level Design

Original GraphControl (conceptual):

```python
# Conceptual baseline
H = f_frozen(PE_orig) + Z2( f_ctrl(PE_orig + Z1(PE_cond)) )
````

* `f_frozen`: frozen GCC encoder
* `f_ctrl`: trainable copy of encoder
* `PE_orig`: positional embedding from original adjacency `A`
* `PE_cond`: positional embedding from feature-similarity adjacency `A'`
* `Z1`, `Z2`: zero-initialized MLPs

New **Layerwise GraphControl** (conceptual):

* Expose the GCC encoder as a sequence of layers: `layers[0], ..., layers[L-1]`
* Keep *two* encoders:

  * `encoder_frozen.layers[l]`
  * `encoder_copy.layers[l]`
* Use a shared **projection of PE_cond** to hidden dimension (`cond_proj`)
* For each layer `l`, define a **zero-MLP** `zero_layer[l]`
* In the forward pass:

  * Step through layers synchronously
  * At each layer, feed the **same projected condition** into the control branch
  * Inject the control branch output into the frozen branch using `zero_layer[l]`

Conceptual forward:

```python
cond_h = cond_proj(PE_cond)

h_frozen = PE_orig
h_ctrl   = PE_orig   # or zeros_like(PE_orig), see notes

for l in layers:
    h_frozen = layer_frozen[l](h_frozen, edge_index)
    h_ctrl   = layer_ctrl[l](h_ctrl + cond_h, edge_index)
    h_frozen = h_frozen + zero_layer[l](h_ctrl)  # starts as 0, grows during training

H = h_frozen
```

**Important constraint:**
Do **not** change how `PE_cond` is computed. Only change *where and how* it is injected across layers.

---

## 1. Locate the Relevant Code

1. Find the **GCC encoder** used as the backbone. Likely filenames:

   * `models/gcc_encoder.py`
   * `models/backbone.py`
   * or similar.

2. Find the **GraphControl wrapper** that:

   * Holds the frozen encoder and the trainable copy.
   * Performs a single-shot call like:

     ```python
     h_frozen = encoder_frozen(pe_orig, edge_index)
     h_ctrl   = encoder_copy(pe_orig + Z1(pe_cond), edge_index)
     out      = h_frozen + Z2(h_ctrl)
     ```

   This is the main point to refactor.

3. Identify where **positional embeddings** are computed:

   * `PE_orig` from original adjacency `A`
   * `PE_cond` from feature-similarity adjacency `A'`
     **Do not change these computations.**

---

## 2. Step 1 – Make GCC Encoder Explicitly Layerwise

We need the encoder to expose its layers as a `ModuleList`.

If it’s not already structured this way, refactor it. The goal is to have something like:

```python
# Example: GCC backbone with explicit layers
class GCCBackbone(nn.Module):
    def __init__(self, conv_layers, readout=None):
        super().__init__()
        self.layers = nn.ModuleList(conv_layers)  # [layer0, layer1, ...]
        self.readout = readout                    # optional pooling / MLP head

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        if self.readout is not None:
            x = self.readout(x)
        return x
```

If the current backbone has all logic in a monolithic `forward` and does not expose `self.layers`, refactor it as follows:

* Extract the per-layer convolution / message-passing modules into `self.layers`.
* Keep any global pooling / readout as `self.readout`.

After that, ensure there are **two** instances:

```python
encoder_frozen = GCCBackbone(...)
encoder_copy   = GCCBackbone(...)

# Freeze one of them
for p in encoder_frozen.parameters():
    p.requires_grad = False
```

---

## 3. Step 2 – Define ZeroMLP (if not already present)

If there is already a `ZeroMLP` or `zero_module` used in GraphControl, reuse it.
Otherwise, define one in a shared utilities file (e.g., `models/utils.py`):

```python
class ZeroMLP(nn.Module):
    """
    Simple MLP (or linear layer stack) whose output is initialized to zero.
    This mimics the 'zero-conv' behavior from ControlNet.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # Single linear layer
            self.net = nn.Linear(in_dim, out_dim)
        else:
            # Two-layer MLP if needed
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
```

If there is already a project-specific zero-initialized block (e.g., `ZeroConv`, `ZeroModule`), use that instead of this definition.

---

## 4. Step 3 – Implement LayerwiseGraphControl

Create a new module, for example `models/layerwise_graphcontrol.py`:

```python
import torch
import torch.nn as nn

class LayerwiseGraphControl(nn.Module):
    """
    Layerwise GraphControl:
    - Uses the SAME condition PE (PE_cond) for all layers.
    - Only changes WHERE the condition is injected.
    - Does NOT change how PE_cond is computed.
    """
    def __init__(self, encoder_frozen, encoder_copy, hidden_dim, pe_dim, readout=None):
        super().__init__()
        self.encoder_frozen = encoder_frozen  # GCCBackbone, with .layers
        self.encoder_copy   = encoder_copy    # same architecture, trainable

        self.num_layers = len(self.encoder_frozen.layers)

        # Project condition PE to hidden dimension (shared across layers)
        self.cond_proj = ZeroMLP(pe_dim, hidden_dim)

        # Per-layer zero-MLPs for residual injection
        self.zero_layer = nn.ModuleList([
            ZeroMLP(hidden_dim, hidden_dim) for _ in range(self.num_layers)
        ])

        # Optional readout (if not already inside encoder_frozen)
        self.readout = readout

        # Freeze pretrained encoder parameters
        for p in self.encoder_frozen.parameters():
            p.requires_grad = False

    def forward(self, pe_orig, pe_cond, edge_index, batch=None):
        """
        pe_orig: [N, pe_dim] positional embedding from original adjacency A
        pe_cond: [N, pe_dim] positional embedding from feature-sim adjacency A' (unchanged!)
        edge_index: [2, E] graph edges
        batch: optional [N] batch indices for pooling
        """

        # 1. Compute shared condition representation
        cond_h = self.cond_proj(pe_cond)  # [N, hidden_dim]

        # 2. Initialize hidden states
        #    Option A: both branches start from pe_orig
        h_frozen = pe_orig
        h_ctrl   = pe_orig

        # Layerwise propagation
        for l, (layer_frozen, layer_ctrl) in enumerate(
            zip(self.encoder_frozen.layers, self.encoder_copy.layers)
        ):
            # Frozen branch: standard GCC layer
            h_frozen = layer_frozen(h_frozen, edge_index)

            # Control branch: inject same condition at each layer
            h_in_ctrl = h_ctrl + cond_h        # SAME cond_h for all layers
            h_ctrl    = layer_ctrl(h_in_ctrl, edge_index)

            # Layerwise residual injection:
            # - zero_layer[l] is zero-initialized
            # - initially, this adds ~0 to h_frozen (pure frozen behavior)
            h_frozen = h_frozen + self.zero_layer[l](h_ctrl)

        # Optional readout
        out = h_frozen
        if self.readout is not None:
            out = self.readout(out, batch=batch)

        return out
```

> Note: If the encoder’s readout is already inside `encoder_frozen`, you can remove `readout` here and simply return `h_frozen` (or whatever shape the classifier expects).

---

## 5. Step 4 – Wire It into the Training Script

Locate the part of the training code where GraphControl is currently instantiated.
This may look similar to:

```python
encoder_frozen = GCCBackbone(...)
encoder_copy   = GCCBackbone(...)
model = GraphControl(encoder_frozen, encoder_copy, hidden_dim, pe_dim, ...)
```

Replace this with:

```python
from models.layerwise_graphcontrol import LayerwiseGraphControl

encoder_frozen = GCCBackbone(...)
encoder_copy   = GCCBackbone(...)
model = LayerwiseGraphControl(
    encoder_frozen=encoder_frozen,
    encoder_copy=encoder_copy,
    hidden_dim=HIDDEN_DIM,  # must match encoder hidden size
    pe_dim=PE_DIM,          # dimensionality of positional embeddings
    readout=READOUT_MODULE  # if needed
)
```

Make sure:

* `HIDDEN_DIM` matches the hidden feature size of the GCC encoder layers.
* `PE_DIM` matches the dimension of `pe_orig` / `pe_cond`.
* Any classifier head (MLP for node classification) is wired on top of the output of `model`.

**Important:**
Wherever the original GraphControl `forward` was called with `(pe_orig, pe_cond, edge_index, ...)`, call the new model with the **same arguments** so that “What” stays identical:

```python
# Example
logits = model(pe_orig, pe_cond, edge_index, batch=batch)
loss   = criterion(logits[train_mask], labels[train_mask])
```

---

## 6. Initialization and Sanity Check

Before training, verify the following:

1. All `ZeroMLP` instances (`cond_proj` and each `zero_layer[l]`) have:

   * weights initialized to zero,
   * biases initialized to zero.

2. If you run a **forward pass before any training** with the same inputs as the original GCC encoder,
   the outputs of `LayerwiseGraphControl` should be **very close** to the outputs of the frozen GCC encoder alone
   (differences only from numerical noise).

A quick unit test (pseudo-code):

```python
with torch.no_grad():
    out_gcc      = encoder_frozen(pe_orig, edge_index)   # baseline
    out_lw_gc    = model(pe_orig, pe_cond, edge_index)   # layerwise GraphControl

    diff = (out_gcc - out_lw_gc).abs().mean().item()
    print("Mean abs difference:", diff)
```

Expected: `diff` should be ~0 at initialization.

---

## 7. Experiment Notes (for the human researcher)

This implementation changes only **Where** the condition is injected:

* Same `PE_cond` (feature-similarity adjacency → Laplacian PE).
* Same backbone encoder structure.
* Only difference: instead of one global `f_ctrl` + global zero-MLPs, we now:

  * use per-layer control branches,
  * inject the same `PE_cond` at each layer,
  * use per-layer zero-MLPs for residual integration.

After this is working and trained, the next steps (not for this document / not for Codex yet) can include:

* Trying different variants of:
  * where to add `cond_h` (only some layers, concat vs add, etc.),
  * whether the control branch initial state is `pe_orig` or zeros,
  * different readout positions.

-----