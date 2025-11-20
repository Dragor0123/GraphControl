from typing import List, Union
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import torch.nn.functional as F
import torch
import scipy
import sklearn.preprocessing as preprocessing

from .normalize import similarity, get_laplacian_matrix

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


def compute_khop_condition_pe_pure(x, edge_index, num_nodes, k, num_dim=32, threshold=0.1):
    """
    Option A: Pure k-hop masking
    S_k = A^k * (X @ X.T)

    Args:
        x: [num_nodes, feature_dim] original node features
        edge_index: [2, E] edge index
        num_nodes: number of nodes
        k: hop distance (layer index)
        num_dim: PE dimension
        threshold: discretization threshold

    Returns:
        PE_k: [num_nodes, num_dim] k-hop condition PE
    """
    device = x.device

    # Compute feature similarity: X @ X.T
    S = similarity(x, x)

    # Compute A^k
    if k == 0:
        A_k = torch.eye(num_nodes, dtype=torch.float32, device=device)
    else:
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        A_k = torch.linalg.matrix_power(A, k) if k > 1 else A

    # Apply masking: S_k = A^k * S
    S_k = A_k * S

    # Discretize
    S_k = torch.where(S_k > threshold, 1.0, 0.0)

    # Compute Laplacian PE
    L_k = get_laplacian_matrix(S_k)

    try:
        eigen_vals, eigen_vecs = torch.linalg.eigh(L_k)
    except RuntimeError:
        L_k_np = L_k.cpu().numpy()
        eigen_vals, eigen_vecs = scipy.linalg.eigh(L_k_np)
        eigen_vals = torch.from_numpy(eigen_vals).to(device)
        eigen_vecs = torch.from_numpy(eigen_vecs).to(device)

    # Extract and normalize eigenvectors
    if eigen_vecs.shape[1] < num_dim:
        PE_k = torch.nn.functional.pad(eigen_vecs, (0, num_dim - eigen_vecs.shape[1]))
    else:
        PE_k = eigen_vecs[:, :num_dim]

    PE_k = PE_k.float()
    PE_k_np = preprocessing.normalize(PE_k.cpu().numpy(), norm="l2")
    PE_k = torch.tensor(PE_k_np, dtype=torch.float32, device=device)

    return PE_k


def compute_khop_condition_pe_cumulative(x, edge_index, num_nodes, k, num_dim=32, threshold=0.1):
    """
    Option B: Cumulative k-hop masking
    S_k = (I + A + A^2 + ... + A^k) * (X @ X.T)

    Args:
        x: [num_nodes, feature_dim] original node features
        edge_index: [2, E] edge index
        num_nodes: number of nodes
        k: maximum hop distance (layer index)
        num_dim: PE dimension
        threshold: discretization threshold

    Returns:
        PE_k: [num_nodes, num_dim] cumulative k-hop condition PE
    """
    device = x.device

    # Compute feature similarity: X @ X.T
    S = similarity(x, x)

    # Compute I + A + A^2 + ... + A^k
    A_cumulative = torch.eye(num_nodes, dtype=torch.float32, device=device)

    if k > 0:
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        A_i = A.clone()

        for i in range(1, k + 1):
            if i > 1:
                A_i = torch.matmul(A_i, A)
            A_cumulative = A_cumulative + A_i

        # Binarize
        A_cumulative = (A_cumulative > 0).float()

    # Apply masking: S_k = A_cumulative * S
    S_k = A_cumulative * S

    # Discretize
    S_k = torch.where(S_k > threshold, 1.0, 0.0)

    # Compute Laplacian PE
    L_k = get_laplacian_matrix(S_k)

    try:
        eigen_vals, eigen_vecs = torch.linalg.eigh(L_k)
    except RuntimeError:
        L_k_np = L_k.cpu().numpy()
        eigen_vals, eigen_vecs = scipy.linalg.eigh(L_k_np)
        eigen_vals = torch.from_numpy(eigen_vals).to(device)
        eigen_vecs = torch.from_numpy(eigen_vecs).to(device)

    # Extract and normalize eigenvectors
    if eigen_vecs.shape[1] < num_dim:
        PE_k = torch.nn.functional.pad(eigen_vecs, (0, num_dim - eigen_vecs.shape[1]))
    else:
        PE_k = eigen_vecs[:, :num_dim]

    PE_k = PE_k.float()
    PE_k_np = preprocessing.normalize(PE_k.cpu().numpy(), norm="l2")
    PE_k = torch.tensor(PE_k_np, dtype=torch.float32, device=device)

    return PE_k