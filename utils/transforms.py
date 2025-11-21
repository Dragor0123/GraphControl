from typing import List, Union
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import torch.nn.functional as F
import torch
import scipy
import sklearn.preprocessing as preprocessing

from .normalize import similarity, get_laplacian_matrix

def apply_edge_dropout_to_similarity(x_sim, dropout_rate=0.2):
    """
    Apply edge dropout to similarity matrix for condition augmentation.

    Args:
        x_sim: [N, num_dim] positional embedding from similarity matrix
        dropout_rate: probability of dropping edges (default: 0.2)

    Returns:
        x_sim_aug: [N, num_dim] augmented positional embedding
    """
    # To apply edge dropout, we need to:
    # 1. Reconstruct approximate similarity from PE (just add random noise)
    # 2. Or directly apply dropout to PE dimensions

    # Simple approach: apply dropout to PE dimensions during training
    # This simulates the effect of edge dropout on the resulting PE
    if dropout_rate > 0:
        mask = torch.rand_like(x_sim) > dropout_rate
        x_sim_aug = x_sim * mask.float()
        # Renormalize to maintain similar scale
        x_sim_aug = x_sim_aug * (1.0 / (1.0 - dropout_rate))
    else:
        x_sim_aug = x_sim

    return x_sim_aug

def obtain_attributes(data, use_adj=False, threshold=0.1, num_dim=32):
    save_node_border = 30000
        
    if use_adj:
        # to undirected and remove self-loop
        edges = to_undirected(data.edge_index)
        edges, _ = remove_self_loops(edges)
        tmp = to_dense_adj(edges)[0]
    else:
        tmp = similarity(data.x, data.x)
        
        # discretize the similarity matrix by threshold
        tmp = torch.where(tmp>threshold, 1.0, 0.0)

    tmp = get_laplacian_matrix(tmp)
    if tmp.shape[0] > save_node_border:
        L, V = scipy.linalg.eigh(tmp)
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(tmp) # much faster than torch.linalg.eig
    
    x = V[:, :num_dim].float()
    x = preprocessing.normalize(x.cpu(), norm="l2")
    x = torch.tensor(x, dtype=torch.float32)

    return x
    

def process_attributes(data, use_adj=False, threshold=0.1, num_dim=32, soft=False, kernel=False):
    '''
    Replace the node attributes with positional encoding. Warning: this function will replace the node attributes!
    
    Args:
      data: a single graph contains x (if use_adj=False) and edge_index.
      use_adj: use the eigen-vectors of adjacent matrix or similarity matrix as node attributes.
      threshold: only work when use_adj=False, used for discretize the similarity matrix. 1 if Adj(i,j)>0.1 else 0
      soft: only work when use_adj=False, if soft=True, we will use soft similarity matrix.
      
    Returns:
      modified data.
    '''
    
    if use_adj:
        # to undirected and remove self-loop
        edges = to_undirected(data.edge_index)
        if edges.size(1) > 1:
            edges, _ = remove_self_loops(edges)
        else:
            edges = torch.tensor([[0],[0]]) # for isolated nodes
        Adj = to_dense_adj(edges)[0]
    else:
        
        if kernel:      
            # memory efficient
            XY = (data.x@data.x.T) # 2xy
            deg = torch.diag(XY)
            Y_norm = deg.repeat(XY.shape[0],1)
            X_norm = Y_norm.T
            Adj = X_norm - 2*XY + Y_norm # |X-Y|^2
            Adj = torch.exp(-0.05*Adj) # rbf kernel
        else:
            Adj = similarity(data.x, data.x) # equal to linear kernel
        if soft:
            L, V = torch.linalg.eigh(Adj)
            x = V[:, :num_dim].float()
            x = F.normalize(x, dim=1)
            data.x = x
            return data
        else:
            # discretize the similarity matrix by threshold
            Adj = torch.where(Adj>threshold, 1.0, 0.0)
    Lap = get_laplacian_matrix(Adj)
    
    L, V = torch.linalg.eigh(Lap) # much faster than torch.linalg.eig, if this line triggers bugs please refer to https://github.com/pytorch/pytorch/issues/70122#issuecomment-1232766638
    L_sort, _ = torch.sort(L, descending=False)
    hist = torch.histc(L, bins=32, min=0, max=2)
    hist = hist.unsqueeze(0)

    # Padding
    import sklearn.preprocessing as preprocessing
    if V.shape[0] < num_dim:
        V = preprocessing.normalize(V, norm="l2")
        V = torch.tensor(V, dtype=torch.float32)
        x = torch.nn.functional.pad(V, (0, num_dim-V.shape[0]))
        data.x = x.float()
        data.eigen_val = torch.nn.functional.pad(L_sort, (0, num_dim-L_sort.shape[0])).unsqueeze(0)
    else:
        x = V[:, 0:num_dim].float()
        x = preprocessing.normalize(x, norm="l2")
        x = torch.tensor(x, dtype=torch.float32)
        data.x = x.float()
        data.eigen_val = L_sort[:num_dim].unsqueeze(0)

    return data


def precompute_propagated_features(x, edge_index, num_nodes, num_layers):
    """
    Precompute feature propagation: X, AX, A²X, ..., A^(num_layers-1)X

    This computes propagated features once for the entire graph:
    - Layer 0: X
    - Layer 1: AX
    - Layer 2: A²X
    - Layer k: AᵏX

    Args:
        x: [N, d] original node features
        edge_index: [2, E] edge indices
        num_nodes: number of nodes
        num_layers: number of GNN layers

    Returns:
        propagated_features: list of [N, d] tensors, length = num_layers
    """
    device = x.device

    # Convert to sparse adjacency matrix
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp

    adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # Normalize: D^(-1/2) A D^(-1/2) for better propagation
    # Or use simple row normalization: D^(-1) A
    row_sum = torch.tensor(adj_scipy.sum(axis=1)).squeeze()
    row_sum[row_sum == 0] = 1  # Avoid division by zero
    d_inv = 1.0 / row_sum
    d_inv_mat = sp.diags(d_inv.cpu().numpy())
    adj_normalized = d_inv_mat @ adj_scipy

    # Convert to torch sparse tensor
    adj_coo = adj_normalized.tocoo()
    indices = torch.LongTensor([adj_coo.row, adj_coo.col]).to(device)
    values = torch.FloatTensor(adj_coo.data).to(device)
    adj_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    # Propagate features
    propagated = [x]  # Layer 0: original features
    x_current = x.clone()

    for k in range(1, num_layers):
        x_current = torch.sparse.mm(adj_sparse, x_current)
        propagated.append(x_current)

    return propagated


def compute_propagated_similarity_pe(x_propagated, threshold=0.1, num_dim=32):
    """
    Compute PE from propagated feature similarity: (AᵏX) @ (AᵏX).T

    Args:
        x_propagated: [N, d] propagated features for layer k
        threshold: discretization threshold
        num_dim: PE dimension

    Returns:
        PE: [N, num_dim] positional embedding
    """
    device = x_propagated.device

    # Compute similarity: (AᵏX) @ (AᵏX).T
    S = similarity(x_propagated, x_propagated)

    # Discretize
    S = torch.where(S > threshold, 1.0, 0.0)

    # Compute Laplacian PE
    L = get_laplacian_matrix(S)

    try:
        eigen_vals, eigen_vecs = torch.linalg.eigh(L)
    except RuntimeError:
        L_np = L.cpu().numpy()
        eigen_vals, eigen_vecs = scipy.linalg.eigh(L_np)
        eigen_vals = torch.from_numpy(eigen_vals).to(device)
        eigen_vecs = torch.from_numpy(eigen_vecs).to(device)

    # Extract and normalize eigenvectors
    if eigen_vecs.shape[1] < num_dim:
        PE = torch.nn.functional.pad(eigen_vecs, (0, num_dim - eigen_vecs.shape[1]))
    else:
        PE = eigen_vecs[:, :num_dim]

    PE = PE.float()
    PE_np = preprocessing.normalize(PE.cpu().numpy(), norm="l2")
    PE = torch.tensor(PE_np, dtype=torch.float32, device=device)

    return PE


