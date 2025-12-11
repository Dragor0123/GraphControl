from typing import List, Union
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj
import torch.nn.functional as F
import torch
import scipy
import sklearn.preprocessing as preprocessing
import numpy as np

from .normalize import similarity, get_laplacian_matrix
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix
import scipy.sparse as sp

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

def obtain_attributes(data, use_adj=False, threshold=0.1, num_dim=32, labels=None):
    save_node_border = 30000
        
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


def precompute_khop_conditions_pure(data, num_layers=5, threshold=0.1, num_dim=32):
    """
    Precompute all k-hop conditions for the entire graph (Option A: Pure)

    Args:
        data: graph data with x and edge_index
        num_layers: number of GNN layers (5 for GCC)
        threshold: discretization threshold
        num_dim: PE dimension

    Returns:
        List of [N, num_dim] tensors, one per layer
    """
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp

    device = data.x.device
    num_nodes = data.x.size(0)

    # Compute feature similarity once
    S = similarity(data.x, data.x)

    # Precompute adjacency matrix
    adj_scipy = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)

    conditions = []

    for k in range(num_layers):
        if k == 0:
            # Layer 0: just use feature similarity
            S_masked = S
        else:
            # Compute A^k
            adj_k = adj_scipy
            for _ in range(k - 1):
                adj_k = adj_k @ adj_scipy

            # Convert to dense
            adj_k_dense = torch.from_numpy(adj_k.toarray()).float().to(device)

            # Mask: A^k ⊙ S
            S_masked = adj_k_dense * S

        # Discretize
        S_discrete = torch.where(S_masked > threshold, 1.0, 0.0)

        # Compute Laplacian PE
        L = get_laplacian_matrix(S_discrete)

        if num_nodes > 30000:
            L_np = L.cpu().numpy()
            eigen_vals, eigen_vecs = scipy.linalg.eigh(L_np)
            eigen_vals = torch.from_numpy(eigen_vals).to(device)
            eigen_vecs = torch.from_numpy(eigen_vecs).to(device)
        else:
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

        conditions.append(PE)

    return conditions


def precompute_khop_conditions_cumulative(data, num_layers=5, threshold=0.1, num_dim=32):
    """
    Precompute all k-hop conditions for the entire graph (Option B: Cumulative)

    Args:
        data: graph data with x and edge_index
        num_layers: number of GNN layers (5 for GCC)
        threshold: discretization threshold
        num_dim: PE dimension

    Returns:
        List of [N, num_dim] tensors, one per layer
    """
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp

    device = data.x.device
    num_nodes = data.x.size(0)

    # Compute feature similarity once
    S = similarity(data.x, data.x)

    # Precompute adjacency matrix
    adj_scipy = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)

    conditions = []

    for k in range(num_layers):
        if k == 0:
            # Layer 0: just use feature similarity
            S_masked = S
        else:
            # Compute I + A + A² + ... + A^k
            cumulative_adj = sp.eye(num_nodes)
            adj_power = sp.eye(num_nodes)

            for i in range(1, k + 1):
                adj_power = adj_power @ adj_scipy
                cumulative_adj = cumulative_adj + adj_power

            # Convert to dense
            cumulative_adj_dense = torch.from_numpy(cumulative_adj.toarray()).float().to(device)

            # Mask: (I + A + ... + A^k) ⊙ S
            S_masked = (cumulative_adj_dense > 0).float() * S

        # Discretize
        S_discrete = torch.where(S_masked > threshold, 1.0, 0.0)

        # Compute Laplacian PE
        L = get_laplacian_matrix(S_discrete)

        if num_nodes > 30000:
            L_np = L.cpu().numpy()
            eigen_vals, eigen_vecs = scipy.linalg.eigh(L_np)
            eigen_vals = torch.from_numpy(eigen_vals).to(device)
            eigen_vecs = torch.from_numpy(eigen_vecs).to(device)
        else:
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

        conditions.append(PE)

    return conditions


def compute_khop_condition_pe_pure(x, edge_index, num_nodes, k, threshold=0.1, num_dim=32):
    """
    Option A: Pure k-hop masking
    Compute condition from A^k ⊙ (X @ X.T) - only k-hop neighbors

    Args:
        x: [N, d] node features
        edge_index: [2, E] edge indices
        num_nodes: number of nodes
        k: hop distance (0 for X@X.T, 1 for 1-hop, etc.)
        threshold: discretization threshold
        num_dim: PE dimension

    Returns:
        PE: [N, num_dim] positional embedding
    """
    device = x.device

    # Compute feature similarity
    S = similarity(x, x)

    if k == 0:
        # Layer 0: just use feature similarity
        S_masked = S
    else:
        # Compute A^k
        from torch_geometric.utils import to_scipy_sparse_matrix
        import scipy.sparse as sp

        adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

        # Compute A^k
        adj_k = adj_scipy
        for _ in range(k - 1):
            adj_k = adj_k @ adj_scipy

        # Convert to dense tensor
        adj_k_dense = torch.from_numpy(adj_k.toarray()).float().to(device)

        # Mask: A^k ⊙ S (only k-hop neighbors)
        S_masked = adj_k_dense * S

    # Discretize
    S_discrete = torch.where(S_masked > threshold, 1.0, 0.0)

    # Compute Laplacian PE
    L = get_laplacian_matrix(S_discrete)

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


def compute_khop_condition_pe_cumulative(x, edge_index, num_nodes, k, threshold=0.1, num_dim=32):
    """
    Option B: Cumulative k-hop masking
    Compute condition from (I + A + A² + ... + A^k) ⊙ (X @ X.T)

    Args:
        x: [N, d] node features
        edge_index: [2, E] edge indices
        num_nodes: number of nodes
        k: maximum hop distance
        threshold: discretization threshold
        num_dim: PE dimension

    Returns:
        PE: [N, num_dim] positional embedding
    """
    device = x.device

    # Compute feature similarity
    S = similarity(x, x)

    if k == 0:
        # Layer 0: just use feature similarity
        S_masked = S
    else:
        # Compute I + A + A² + ... + A^k
        from torch_geometric.utils import to_scipy_sparse_matrix
        import scipy.sparse as sp

        adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

        # Initialize with I (identity)
        cumulative_adj = sp.eye(num_nodes)

        # Add A, A², ..., A^k
        adj_power = sp.eye(num_nodes)
        for i in range(1, k + 1):
            adj_power = adj_power @ adj_scipy
            cumulative_adj = cumulative_adj + adj_power

        # Convert to dense tensor
        cumulative_adj_dense = torch.from_numpy(cumulative_adj.toarray()).float().to(device)

        # Mask: (I + A + ... + A^k) ⊙ S
        S_masked = (cumulative_adj_dense > 0).float() * S

    # Discretize
    S_discrete = torch.where(S_masked > threshold, 1.0, 0.0)

    # Compute Laplacian PE
    L = get_laplacian_matrix(S_discrete)

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


def build_two_hop_condition_pe(data, num_dim=32, two_hop_threshold=0.0, two_hop_topk=0, remove_one_hop=True):
    """
    Build positional embeddings from pure 2-hop structural adjacency (A^2) for S1 condition.

    Steps:
      1) Build sparse adjacency with self-loops (consistent with typical GNN usage).
      2) Compute A^2.
      3) Optionally remove 1-hop edges and self-loops to keep pure 2-hop structure.
      4) Optional thresholding and top-k pruning.
      5) Row-normalize and compute Laplacian eigenvectors.

    Returns:
      torch.Tensor [N, num_dim] positional embedding (L2-normalized).
    """
    from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
    import scipy.sparse as sp

    num_nodes = data.x.size(0)
    device = data.x.device

    # 1) Build adjacency with self-loops
    edge_index_with_loops, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
    A = to_scipy_sparse_matrix(edge_index_with_loops, num_nodes=num_nodes).tocsr()

    # 2) Compute A^2
    A2 = A @ A

    # Binary views for masking
    A_bin = (A > 0).astype(np.int64)
    A2_bin = (A2 > 0).astype(np.int64)

    # 3) Remove 1-hop edges (pure 2-hop) and self-loops
    if remove_one_hop:
        A2_bin = A2_bin.multiply((A_bin == 0))
    A2_bin.setdiag(0)
    A2_bin.eliminate_zeros()

    # Use weighted version based on counts for threshold/topk
    A2_val = A2.multiply(A2_bin)  # keep structure mask

    # 4) Thresholding
    if two_hop_threshold > 0.0:
        A2_val = A2_val.multiply(A2_val >= two_hop_threshold)
        A2_val.eliminate_zeros()

    # 4b) Top-k per row
    if two_hop_topk > 0:
        A2_val = _row_topk_csr(A2_val, k=two_hop_topk)

    # 5) Row-normalize
    row_sum = np.array(A2_val.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1.0
    inv_row = 1.0 / row_sum
    D_inv = sp.diags(inv_row)
    C_norm = D_inv @ A2_val  # row-normalized

    # Convert to torch dense adjacency
    C_dense = torch.from_numpy(C_norm.toarray()).float().to(device)
    L = get_laplacian_matrix(C_dense)

    # Eigen-decomposition
    try:
        eigen_vals, eigen_vecs = torch.linalg.eigh(L)
    except RuntimeError:
        L_np = L.cpu().numpy()
        eigen_vals, eigen_vecs = scipy.linalg.eigh(L_np)
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


def _row_topk_csr(mat_csr, k):
    """
    Keep top-k entries per row for a CSR sparse matrix.
    """
    import scipy.sparse as sp
    mat_csr = mat_csr.tocsr()
    rows, cols, data = [], [], []
    for i in range(mat_csr.shape[0]):
        row_data = mat_csr.data[mat_csr.indptr[i]:mat_csr.indptr[i+1]]
        row_idx = mat_csr.indices[mat_csr.indptr[i]:mat_csr.indptr[i+1]]
        if row_data.size == 0:
            continue
        topk_idx = np.argsort(-row_data)[:k]
        rows.extend([i] * len(topk_idx))
        cols.extend(row_idx[topk_idx])
        data.extend(row_data[topk_idx])
    return sp.csr_matrix((data, (rows, cols)), shape=mat_csr.shape)


def build_ppr_condition_pe(data, num_dim=32, alpha=0.15, topk=32, symmetric=False, normalize=True):
    """
    Build positional embeddings from PPR-based sparse adjacency (S3).

    Args:
        data: PyG Data with edge_index
        num_dim: PE dimension
        alpha: teleport probability
        topk: top-k neighbors per node
        symmetric: symmetrize PPR matrix
        normalize: row-normalize PPR matrix

    Returns:
        torch.Tensor [N, num_dim]
    """
    device = data.x.device
    num_nodes = data.x.size(0)

    # Adjacency with self-loops
    edge_index_with_loops, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
    A = to_scipy_sparse_matrix(edge_index_with_loops, num_nodes=num_nodes).tocsr()

    # Transition matrix T
    deg = np.array(A.sum(axis=1)).flatten()
    inv_deg = 1.0 / np.maximum(deg, 1e-12)
    D_inv = sp.diags(inv_deg)
    T = D_inv @ A  # row-stochastic

    # Power iteration for PPR
    I = sp.eye(num_nodes, format='csr')
    P = I.copy()
    for _ in range(10):
        P = alpha * I + (1 - alpha) * (P @ T)

    # Top-k per row
    if topk > 0:
        P = _row_topk_csr(P, k=topk)

    # Symmetrize if needed
    if symmetric:
        P = 0.5 * (P + P.T)

    # Remove self-loops
    P.setdiag(0)
    P.eliminate_zeros()

    # Row-normalize
    if normalize:
        row_sum = np.array(P.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1.0
        inv_row = 1.0 / row_sum
        D_inv_p = sp.diags(inv_row)
        P = D_inv_p @ P

    # Laplacian PE
    P_dense = torch.from_numpy(P.toarray()).float().to(device)
    L = get_laplacian_matrix(P_dense)
    try:
        eigen_vals, eigen_vecs = torch.linalg.eigh(L)
    except RuntimeError:
        L_np = L.cpu().numpy()
        eigen_vals, eigen_vecs = scipy.linalg.eigh(L_np)
        eigen_vecs = torch.from_numpy(eigen_vecs).to(device)

    if eigen_vecs.shape[1] < num_dim:
        PE = torch.nn.functional.pad(eigen_vecs, (0, num_dim - eigen_vecs.shape[1]))
    else:
        PE = eigen_vecs[:, :num_dim]

    PE = PE.float()
    PE_np = preprocessing.normalize(PE.cpu().numpy(), norm="l2")
    PE = torch.tensor(PE_np, dtype=torch.float32, device=device)
    return PE
