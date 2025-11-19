import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.register import register
from .gcc import GCC

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
