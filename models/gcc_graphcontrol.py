import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.register import register
from .gcc import GCC
from utils.transforms import compute_propagated_similarity_pe

@register.model_register
class GCC_GraphControl(nn.Module):

    def __init__(
        self,
        **kwargs
    ):
        super(GCC_GraphControl, self).__init__()
        positional_dim = kwargs['positional_embedding_size']
        hidden_size = kwargs['node_hidden_dim']
        output_dim = kwargs['num_classes']

        self.encoder = GCC(**kwargs)
        self.trainable_copy = copy.deepcopy(self.encoder)

        self.hidden_size = hidden_size
        self.node_input_dim = self.encoder.node_input_dim

        self.cond_proj = nn.Linear(positional_dim, hidden_size)
        self.cond_input_adapter = nn.Linear(hidden_size, self.node_input_dim)
        num_layers = len(self.encoder.gnn.layers)
        self.zero_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        )

        self.linear_classifier = nn.Linear(hidden_size, output_dim)

        self._zero_init_module(self.cond_proj)
        self._zero_init_module(self.cond_input_adapter)
        for layer in self.zero_layers:
            self._zero_init_module(layer)

    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')

    def reset_classifier(self):
        self.linear_classifier.reset_parameters()

    def forward_subgraph(self, x, x_sim, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        if not frozen:
            raise NotImplementedError('Please freeze pre-trained models')

        self.encoder.eval()

        with torch.no_grad():
            h_frozen = self.encoder.prepare_node_features(x, edge_index, root_n_id)

        h_ctrl = self.trainable_copy.prepare_node_features(x, edge_index, root_n_id)
        cond_hidden = self.cond_proj(x_sim)
        cond_first_layer = self.cond_input_adapter(cond_hidden)

        hidden_states = [h_frozen]
        encoder_layers = self.encoder.gnn.layers
        ctrl_layers = self.trainable_copy.gnn.layers

        for layer_idx, (layer_frozen, layer_ctrl, zero_layer) in enumerate(
            zip(encoder_layers, ctrl_layers, self.zero_layers)
        ):
            with torch.no_grad():
                h_frozen = layer_frozen(h_frozen, edge_index)

            if layer_idx == 0:
                ctrl_input = h_ctrl + cond_first_layer
            else:
                ctrl_input = h_ctrl + cond_hidden
            h_ctrl = layer_ctrl(ctrl_input, edge_index)

            h_frozen = h_frozen + zero_layer(h_ctrl)
            hidden_states.append(h_frozen)

        out, _ = self.encoder.gnn.graph_readout(hidden_states, batch)
        if self.encoder.norm:
            out = F.normalize(out, p=2, dim=-1, eps=1e-5)

        x = self.linear_classifier(out)
        return x

    @staticmethod
    def _zero_init_module(module):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@register.model_register
class GCC_GraphControl_Propagation(nn.Module):
    """
    Option D: Layerwise GraphControl with same condition, different projections per layer.
    All layers use the same base condition PE from X @ X.T, but with layer-specific projections.
    """

    def __init__(self, **kwargs):
        super(GCC_GraphControl_Propagation, self).__init__()
        positional_dim = kwargs.pop('positional_embedding_size')
        hidden_size = kwargs.pop('node_hidden_dim')
        output_dim = kwargs.pop('num_classes')
        # Remove threshold from kwargs (not needed anymore)
        kwargs.pop('threshold', None)

        self.encoder = GCC(
            positional_embedding_size=positional_dim,
            node_hidden_dim=hidden_size,
            num_classes=output_dim,
            **kwargs
        )
        self.trainable_copy = copy.deepcopy(self.encoder)

        self.hidden_size = hidden_size
        self.node_input_dim = self.encoder.node_input_dim
        self.positional_dim = positional_dim

        # Layer-wise condition projections (each layer gets same PE, different projection)
        num_layers = len(self.encoder.gnn.layers)
        self.num_layers = num_layers

        # Each layer has its own projection of the base condition
        self.cond_projs = nn.ModuleList([
            nn.Linear(positional_dim, hidden_size) for _ in range(num_layers)
        ])
        self.cond_input_adapters = nn.ModuleList([
            nn.Linear(hidden_size, self.node_input_dim if i == 0 else hidden_size)
            for i in range(num_layers)
        ])
        self.zero_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.linear_classifier = nn.Linear(hidden_size, output_dim)

        # Zero initialization
        for layer in self.cond_projs:
            self._zero_init_module(layer)
        for layer in self.cond_input_adapters:
            self._zero_init_module(layer)
        for layer in self.zero_layers:
            self._zero_init_module(layer)

    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')

    def reset_classifier(self):
        self.linear_classifier.reset_parameters()

    def forward_subgraph(self, x_pe, x_sim, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        """
        Args:
            x_pe: Positional embedding for frozen/copy encoder input
            x_sim: Single base condition PE (same for all layers, computed from X @ X.T)
            edge_index: Edge index
            batch: Batch assignment
            root_n_id: Root node IDs
        """
        if not frozen:
            raise NotImplementedError('Please freeze pre-trained models')

        self.encoder.eval()

        # Prepare initial features
        with torch.no_grad():
            h_frozen = self.encoder.prepare_node_features(x_pe, edge_index, root_n_id)

        h_ctrl = self.trainable_copy.prepare_node_features(x_pe, edge_index, root_n_id)

        hidden_states = [h_frozen]
        encoder_layers = self.encoder.gnn.layers
        ctrl_layers = self.trainable_copy.gnn.layers

        # Layer-wise forward: same condition, different projections
        for layer_idx, (layer_frozen, layer_ctrl, zero_layer) in enumerate(
            zip(encoder_layers, ctrl_layers, self.zero_layers)
        ):
            # Frozen branch
            with torch.no_grad():
                h_frozen = layer_frozen(h_frozen, edge_index)

            # Use same base condition but with layer-specific projection
            cond_k = self.cond_projs[layer_idx](x_sim)  # Same x_sim, different projection
            cond_k_adapted = self.cond_input_adapters[layer_idx](cond_k)

            # Control branch with layer-specific transformed condition
            ctrl_input = h_ctrl + cond_k_adapted
            h_ctrl = layer_ctrl(ctrl_input, edge_index)

            # Inject into frozen branch
            h_frozen = h_frozen + zero_layer(h_ctrl)
            hidden_states.append(h_frozen)

        # Readout
        out, _ = self.encoder.gnn.graph_readout(hidden_states, batch)
        if self.encoder.norm:
            out = F.normalize(out, p=2, dim=-1, eps=1e-5)

        x = self.linear_classifier(out)
        return x

    @staticmethod
    def _zero_init_module(module):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@register.model_register
class GCC_GraphControl_EdgeDropout(nn.Module):
    """
    Layerwise GraphControl with edge dropout augmentation.
    - Layer 0: receives clean condition (X @ X.T based)
    - Layer 1~k: receive dropout-augmented conditions (dropout rate: 0.2 during training)
    - Uses SHARED projection like baseline GCC_GraphControl
    """

    def __init__(self, **kwargs):
        super(GCC_GraphControl_EdgeDropout, self).__init__()
        positional_dim = kwargs.pop('positional_embedding_size')
        hidden_size = kwargs.pop('node_hidden_dim')
        output_dim = kwargs.pop('num_classes')
        self.cond_dropout_rate = kwargs.pop('cond_dropout_rate', 0.2)

        self.encoder = GCC(
            positional_embedding_size=positional_dim,
            node_hidden_dim=hidden_size,
            num_classes=output_dim,
            **kwargs
        )
        self.trainable_copy = copy.deepcopy(self.encoder)

        self.hidden_size = hidden_size
        self.node_input_dim = self.encoder.node_input_dim

        # SHARED projection (same as baseline GCC_GraphControl)
        self.cond_proj = nn.Linear(positional_dim, hidden_size)
        self.cond_input_adapter = nn.Linear(hidden_size, self.node_input_dim)

        # Layer-wise zero layers for injection
        num_layers = len(self.encoder.gnn.layers)
        self.zero_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.linear_classifier = nn.Linear(hidden_size, output_dim)

        # Zero initialization
        self._zero_init_module(self.cond_proj)
        self._zero_init_module(self.cond_input_adapter)
        for layer in self.zero_layers:
            self._zero_init_module(layer)

    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')

    def reset_classifier(self):
        self.linear_classifier.reset_parameters()

    def forward_subgraph(self, x, x_sim, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        """
        Args:
            x: Positional embedding for frozen/copy encoder input
            x_sim: Base condition (X @ X.T based)
            edge_index: Edge index
            batch: Batch assignment
            root_n_id: Root node IDs
        """
        if not frozen:
            raise NotImplementedError('Please freeze pre-trained models')

        self.encoder.eval()

        # Import dropout utility
        from utils.transforms import apply_edge_dropout_to_similarity

        # Prepare initial features
        with torch.no_grad():
            h_frozen = self.encoder.prepare_node_features(x, edge_index, root_n_id)

        h_ctrl = self.trainable_copy.prepare_node_features(x, edge_index, root_n_id)

        hidden_states = [h_frozen]
        encoder_layers = self.encoder.gnn.layers
        ctrl_layers = self.trainable_copy.gnn.layers

        # Process conditions for each layer (shared projection)
        for layer_idx, (layer_frozen, layer_ctrl, zero_layer) in enumerate(
            zip(encoder_layers, ctrl_layers, self.zero_layers)
        ):
            # Frozen branch
            with torch.no_grad():
                h_frozen = layer_frozen(h_frozen, edge_index)

            # Apply dropout to condition for layers > 0 during training
            if layer_idx == 0:
                # Layer 0: clean condition
                cond_input = x_sim
            else:
                # Other layers: augmented condition (only during training)
                if self.training:
                    cond_input = apply_edge_dropout_to_similarity(x_sim, self.cond_dropout_rate)
                else:
                    cond_input = x_sim

            # Use SHARED projection (same for all layers)
            cond_hidden = self.cond_proj(cond_input)

            # First layer needs adapter to node_input_dim, others use hidden directly
            if layer_idx == 0:
                cond_first_layer = self.cond_input_adapter(cond_hidden)
                ctrl_input = h_ctrl + cond_first_layer
            else:
                ctrl_input = h_ctrl + cond_hidden

            # Control branch
            h_ctrl = layer_ctrl(ctrl_input, edge_index)

            # Inject into frozen branch
            h_frozen = h_frozen + zero_layer(h_ctrl)
            hidden_states.append(h_frozen)

        # Readout
        out, _ = self.encoder.gnn.graph_readout(hidden_states, batch)
        if self.encoder.norm:
            out = F.normalize(out, p=2, dim=-1, eps=1e-5)

        x = self.linear_classifier(out)
        return x

    @staticmethod
    def _zero_init_module(module):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
