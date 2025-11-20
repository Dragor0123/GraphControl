import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.register import register
from .gcc import GCC
from utils.transforms import compute_khop_condition_pe_pure, compute_khop_condition_pe_cumulative

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
class GCC_GraphControl_KHop(nn.Module):
    """
    Layerwise GraphControl with k-hop condition masking.
    Choose between Option A (pure) and Option B (cumulative) by changing compute_fn.
    """

    def __init__(self, **kwargs):
        super(GCC_GraphControl_KHop, self).__init__()
        positional_dim = kwargs['positional_embedding_size']
        hidden_size = kwargs['node_hidden_dim']
        output_dim = kwargs['num_classes']

        # Choose condition function: 'pure' or 'cumulative'
        self.khop_mode = kwargs.pop('khop_mode', 'pure')  # Default: Option A
        self.threshold = kwargs.pop('threshold', 0.1)

        self.encoder = GCC(**kwargs)
        self.trainable_copy = copy.deepcopy(self.encoder)

        self.hidden_size = hidden_size
        self.node_input_dim = self.encoder.node_input_dim
        self.positional_dim = positional_dim

        # Layer-wise condition projections
        num_layers = len(self.encoder.gnn.layers)
        self.num_layers = num_layers

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

    def forward_subgraph(self, x_pe, x_original, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        """
        Args:
            x_pe: Positional embedding for frozen/copy encoder input
            x_original: Original node features for computing k-hop conditions
            edge_index: Edge index
            batch: Batch assignment
            root_n_id: Root node IDs
        """
        if not frozen:
            raise NotImplementedError('Please freeze pre-trained models')

        self.encoder.eval()

        # Get compute function based on mode
        if self.khop_mode == 'pure':
            compute_fn = compute_khop_condition_pe_pure
        elif self.khop_mode == 'cumulative':
            compute_fn = compute_khop_condition_pe_cumulative
        else:
            raise ValueError(f"Unknown khop_mode: {self.khop_mode}")

        # Prepare initial features
        with torch.no_grad():
            h_frozen = self.encoder.prepare_node_features(x_pe, edge_index, root_n_id)

        h_ctrl = self.trainable_copy.prepare_node_features(x_pe, edge_index, root_n_id)

        hidden_states = [h_frozen]
        encoder_layers = self.encoder.gnn.layers
        ctrl_layers = self.trainable_copy.gnn.layers

        num_nodes = x_pe.shape[0]

        # Layer-wise forward with k-hop conditions
        for layer_idx, (layer_frozen, layer_ctrl, zero_layer) in enumerate(
            zip(encoder_layers, ctrl_layers, self.zero_layers)
        ):
            # Frozen branch
            with torch.no_grad():
                h_frozen = layer_frozen(h_frozen, edge_index)

            # Compute k-hop condition for this layer
            PE_k = compute_fn(
                x=x_original,
                edge_index=edge_index,
                num_nodes=num_nodes,
                k=layer_idx,
                num_dim=self.positional_dim,
                threshold=self.threshold
            )

            # Project condition
            cond_k = self.cond_projs[layer_idx](PE_k)
            cond_k_adapted = self.cond_input_adapters[layer_idx](cond_k)

            # Control branch with layer-specific condition
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
