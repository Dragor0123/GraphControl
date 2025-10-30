"""
DisenLink-based Condition Matrix Generator for GraphControl

This module integrates DisenLink's link prediction capabilities to generate
more effective condition matrices for heterophilic graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class Factor(nn.Module):
    """Single-layer factor encoder"""
    def __init__(self, nfeat, nhid):
        super(Factor, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.mlp = nn.Linear(nfeat, nhid)

    def forward(self, x):
        x = self.mlp(x)
        return x


class Factor2(nn.Module):
    """Two-layer factor encoder"""
    def __init__(self, nfeat, nmid, nhid):
        super(Factor2, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nmid = nmid
        self.mlp1 = nn.Linear(nfeat, nmid)
        self.mlp2 = nn.Linear(nmid, nhid)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x


class DisentangleLayer(nn.Module):
    """Factor-aware message passing layer"""
    def __init__(self, nfactor, beta, t=1):
        super(DisentangleLayer, self).__init__()
        self.temperature = t
        self.nfactor = nfactor
        self.beta = beta

    def forward(self, Z, adj):
        """
        Args:
            Z: List of factor embeddings [factor1, factor2, ..., factorK]
            adj: Adjacency matrix

        Returns:
            h_all_factor: List of updated factor embeddings
            alpha0: Factor importance weights
            att: Attention weights for each factor
        """
        # Compute factor importance
        temp = [torch.exp(torch.mm(z, z.t()) / self.temperature) for z in Z]
        alpha0 = torch.stack(temp, dim=0)
        alpha_fea = torch.diagonal(alpha0, 0)
        t = torch.sum(alpha0, dim=0)
        alpha = alpha0 / t

        # Assign edges to factors
        p = torch.argmax(alpha, dim=0) + 1
        p_adj = p * adj

        edge_factor_list = []
        for i in range(self.nfactor):
            mask = (p_adj == i + 1).float()
            edge_factor_list.append(mask)

        # Factor-aware message passing
        h_all_factor = []
        att = []
        for i in range(self.nfactor):
            alpha1 = edge_factor_list[i] * alpha[i]
            sum_alpha = torch.sum(alpha1, dim=1)
            sum_alpha[sum_alpha == 0] = 1
            alpha1 = alpha1 / sum_alpha.unsqueeze(1)
            att.append(alpha1)
            temp = self.beta * Z[i] + (1 - self.beta) * torch.mm(alpha1, Z[i])
            h_all_factor.append(temp)

        return h_all_factor, alpha0, att


class DisenLink(nn.Module):
    """
    Disentangled Link Prediction Module

    Args:
        nfeat: Input feature dimension
        nhid: Hidden dimension for factor encoders
        nembed: Output embedding dimension per factor
        nfactor: Number of disentangled factors
        beta: Self-attention weight (0-1)
        t: Temperature for softmax
    """
    def __init__(self, nfeat, nhid, nembed, nfactor, beta=0.5, t=1):
        super(DisenLink, self).__init__()

        # Create factor encoders
        if nhid == 1:
            self.factors = [Factor(nfeat, nembed) for _ in range(nfactor)]
        else:
            self.factors = [Factor2(nfeat, nhid, nembed) for _ in range(nfactor)]

        for i, factor in enumerate(self.factors):
            self.add_module('factor_{}'.format(i), factor)

        self.disentangle_layer = DisentangleLayer(nfactor, beta, t)
        self.temperature = t
        self.nfactor = nfactor
        self.beta = beta

    def forward(self, x, adj):
        """
        Forward pass for link prediction

        Args:
            x: Node features [num_nodes, nfeat]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            h: Concatenated factor embeddings
            link_pred: Predicted adjacency matrix
            alpha: Factor importance weights
        """
        # Encode features into factors
        Z = [f(x) for f in self.factors]

        # Factor-aware message passing
        h_all_factor, alpha, att = self.disentangle_layer(Z, adj)

        # Link prediction
        link_pred = []
        for i in range(self.nfactor):
            link_pred.append(h_all_factor[i] @ h_all_factor[i].t())
        link_pred = torch.stack(link_pred, dim=0)
        link_pred = torch.sigmoid(torch.sum(link_pred * alpha, dim=0))

        # Concatenate factor embeddings
        h = torch.cat(h_all_factor, dim=1)

        return h, link_pred, alpha


class DisenLinkConditionGenerator:
    """
    Generates condition matrix A' using DisenLink predictions
    instead of or in addition to feature similarity.

    Args:
        num_factors: Number of disentangled factors (K)
        feature_dim: Input feature dimension
        hidden_dim: Hidden dimension per factor
        embed_dim: Output embedding dimension per factor
        use_hybrid: Whether to combine with feature similarity
        hybrid_alpha: Weight for feature similarity (if hybrid)
        beta: Self-attention weight for DisenLink
        device: Device to run on
    """
    def __init__(
        self,
        num_factors: int = 5,
        feature_dim: int = None,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.5,
        beta: float = 0.5,
        device: str = 'cuda',
        train_epochs: int = 100
    ):
        self.num_factors = num_factors
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        self.beta = beta
        self.device = device
        self.train_epochs = train_epochs
        self.disenlink = None

    def _init_model(self, feature_dim: int):
        """Initialize DisenLink model"""
        if self.disenlink is None:
            self.disenlink = DisenLink(
                nfeat=feature_dim,
                nhid=self.hidden_dim,
                nembed=self.embed_dim,
                nfactor=self.num_factors,
                beta=self.beta
            ).to(self.device)

    def _compute_feature_similarity(self, X: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Compute feature similarity matrix (original GraphControl approach)"""
        # Normalize features
        X_norm = F.normalize(X, p=2, dim=1)
        # Cosine similarity
        sim = torch.mm(X_norm, X_norm.t())
        # Discretize by threshold
        sim = torch.where(sim > threshold, 1.0, 0.0)
        return sim

    def train_on_subgraph(
        self,
        subgraph_data,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = False
    ):
        """
        Train DisenLink on a single subgraph

        Args:
            subgraph_data: PyG Data object with x, edge_index
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay
            verbose: Print training progress

        Returns:
            trained_model: Trained DisenLink model
            final_loss: Final training loss
        """
        from torch_geometric.utils import to_dense_adj

        x = subgraph_data.x.to(self.device)
        edge_index = subgraph_data.edge_index.to(self.device)

        # Convert to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]

        # Initialize model if needed
        if self.disenlink is None:
            self._init_model(x.size(1))

        optimizer = torch.optim.Adam(
            self.disenlink.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.disenlink.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            h, link_pred, alpha = self.disenlink(x, adj)

            # Link reconstruction loss (binary cross-entropy)
            loss = F.binary_cross_entropy(link_pred, adj)

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        return self.disenlink, loss.item()

    def predict_links(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict links using trained DisenLink model

        Args:
            x: Node features
            edge_index: Edge index

        Returns:
            link_pred: Predicted adjacency matrix
        """
        from torch_geometric.utils import to_dense_adj

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]

        if self.disenlink is None:
            raise ValueError("Model not initialized. Call train_on_subgraph first.")

        self.disenlink.eval()
        with torch.no_grad():
            _, link_pred, _ = self.disenlink(x, adj)

        return link_pred

    def generate_condition(
        self,
        data,
        subgraphs: Optional[List] = None,
        train_disenlink: bool = True,
        threshold: float = 0.1,
        sparsify_k: int = 16
    ) -> torch.Tensor:
        """
        Generate condition matrix A' for GraphControl

        Args:
            data: Full graph data (PyG Data object)
            subgraphs: List of sampled subgraphs for training
            train_disenlink: Whether to train DisenLink on subgraphs
            threshold: Threshold for feature similarity (if hybrid)
            sparsify_k: Keep top-k neighbors per node

        Returns:
            A_prime: Condition adjacency matrix
        """
        x = data.x
        edge_index = data.edge_index

        # Train on subgraphs if provided
        if train_disenlink and subgraphs is not None:
            print(f"Training DisenLink on {len(subgraphs)} subgraphs (epochs={self.train_epochs})...")
            for i, subgraph in enumerate(subgraphs):
                if i == 0:
                    # First subgraph - initialize and train
                    self.train_on_subgraph(subgraph, epochs=self.train_epochs, verbose=True)
                else:
                    # Subsequent subgraphs - fine-tune
                    fine_tune_epochs = max(self.train_epochs // 2, 50)
                    self.train_on_subgraph(subgraph, epochs=fine_tune_epochs, verbose=False)

        # Predict links on full graph
        A_pred = self.predict_links(x, edge_index)

        # Hybrid mode: combine with feature similarity
        if self.use_hybrid:
            A_feat = self._compute_feature_similarity(x, threshold)
            A_prime = self.hybrid_alpha * A_feat.to(self.device) + \
                      (1 - self.hybrid_alpha) * A_pred
        else:
            A_prime = A_pred

        # Sparsify: keep top-k neighbors per node
        A_prime = self._sparsify(A_prime, k=sparsify_k)

        return A_prime

    def _sparsify(self, adj: torch.Tensor, k: int = 16) -> torch.Tensor:
        """
        Keep only top-k neighbors per node

        Args:
            adj: Dense adjacency matrix
            k: Number of neighbors to keep

        Returns:
            adj_sparse: Sparsified adjacency matrix
        """
        # For each node, keep top-k connections
        values, indices = torch.topk(adj, k=min(k, adj.size(1)), dim=1)

        # Create sparse matrix
        adj_sparse = torch.zeros_like(adj)
        for i in range(adj.size(0)):
            adj_sparse[i, indices[i]] = adj[i, indices[i]]

        # Make symmetric
        adj_sparse = (adj_sparse + adj_sparse.t()) / 2

        # Add self-loops with high weight
        adj_sparse = adj_sparse + 0.9 * torch.eye(adj.size(0)).to(adj.device)

        return adj_sparse


def get_dataset_specific_config(dataset_name: str) -> dict:
    """
    Return optimal configuration based on dataset heterophily

    Args:
        dataset_name: Name of the dataset

    Returns:
        config: Dictionary with optimal hyperparameters
    """
    heterophilic_datasets = ['chameleon', 'squirrel', 'crocodile',
                            'texas', 'wisconsin', 'cornell', 'actor']

    if dataset_name.lower() in heterophilic_datasets:
        return {
            'num_factors': 6,
            'hidden_dim': 64,
            'embed_dim': 16,
            'use_hybrid': True,
            'hybrid_alpha': 0.2,  # Lower weight on feature similarity
            'beta': 0.6,
            'disenlink_epochs': 300
        }
    else:
        return {
            'num_factors': 4,
            'hidden_dim': 32,
            'embed_dim': 16,
            'use_hybrid': True,
            'hybrid_alpha': 0.7,  # Higher weight on feature similarity
            'beta': 0.5,
            'disenlink_epochs': 200
        }
