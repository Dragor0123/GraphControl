import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seeds", type=int, nargs="+", default=[0])
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='Cora_ML')

        # Model configuration
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=128)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.0)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu', 
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--model', type=str, help="model name", default='GCC_GraphControl',
                                 choices=['GCC', 'GCC_GraphControl', 'GCC_GraphControl_Propagation', 'GCC_GraphControl_EdgeDropout', 'GCC_GraphControl_KHopPure', 'GCC_GraphControl_KHopCumulative'])
    
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam', 
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=5e-4)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=100)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--finetune', action='store_true', help="Quickly find optim parameters")
        
        # Processing node attributes
        self.parser.add_argument('--use_adj', action='store_true', help="use eigen-vectors of adjacent matrix as node attributes")
        self.parser.add_argument('--threshold', type=float, help="the threshold for discreting similarity matrix", default=0.15)
        self.parser.add_argument('--num_dim', type=int, help="the number of replaced node attributes", default=32)
        self.parser.add_argument('--cond_dropout_rate', type=float, help="dropout rate for condition augmentation", default=0.2)     
        # self.parser.add_argument('--ad_aug', action='store_true', help="adversarial augmentation")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.8) #0.3
        self.parser.add_argument('--walk_steps', type=int, help="the number of random walk's steps", default=256)

        # Node2vec config
        self.parser.add_argument('--emb_dim', type=int, default=128, help="Embedding dim for node2vec")
        self.parser.add_argument('--walk_length', type=int, default=50, help="Walk length for node2vec")
        self.parser.add_argument('--context_size', type=int, default=10, help="Context size for node2vec")
        self.parser.add_argument('--walk_per_nodes', type=int, default=10, help="Walk per nodes for node2vec")

        # Logging
        self.parser.add_argument('--log_norms', action='store_true', help="Enable norm logging for h_frozen/h_ctrl/condition vectors")
        self.parser.add_argument('--log_interval', type=int, default=5, help="Epoch interval for norm logging")
        self.parser.add_argument(
            '--ctrl_norm_type',
            type=str,
            default='none',
            choices=['none', 'layernorm', 'layernorm_first'],
            help='Normalization type for GraphControl zero branch (none, layernorm, layernorm_first)'
        )
        # Condition selection
        self.parser.add_argument(
            '--cond_type',
            type=str,
            default='feature',
            choices=['feature', 's1_2hop', 's2_struct_scalar'],
            help='Condition graph type: feature similarity (default), S1 2-hop structural, or S2 degree+PageRank scalars'
        )
        self.parser.add_argument(
            '--two_hop_threshold',
            type=float,
            default=0.0,
            help='Threshold for S1 2-hop condition; keep entries >= threshold (0 to disable)'
        )
        self.parser.add_argument(
            '--two_hop_topk',
            type=int,
            default=0,
            help='Top-k neighbors per node for S1 2-hop condition (0 to disable)'
        )
        # S2 structural scalars
        self.parser.add_argument(
            '--pr_alpha',
            type=float,
            default=0.85,
            help='PageRank alpha for S2 structural condition'
        )
        self.parser.add_argument(
            '--deg_log',
            action='store_true',
            help='Use log(1+deg) for S2 structural condition'
        )
        self.parser.add_argument(
            '--cond_struct_norm',
            type=str,
            default='standard',
            choices=['none', 'standard', 'minmax'],
            help='Normalization for structural condition (degree, pagerank)'
        )
        
    def parse_args(self):
        return self.parser.parse_args()
