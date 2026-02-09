"""
HAN (Heterogeneous Attention Network) Model for Fraud Detection
Based on Wang et al. (2019) - "Heterogeneous Graph Attention Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HANConv
from torch_geometric.data import HeteroData


class HAN(nn.Module):
    """
    Heterogeneous Attention Network với hierarchical attention.
    
    HAN có 2 levels của attention:
    1. Node-level attention: Attention aggregation trong mỗi meta-path
    2. Semantic-level attention: Attention aggregation across meta-paths
    
    Architecture:
    - Dùng meta-paths để capture semantic information
    - Node-level: GAT-style attention trong mỗi meta-path
    - Semantic-level: Learn importance weights cho mỗi meta-path
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_heads=8,
                 dropout=0.6,
                 metadata=None):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            num_classes: Number of classes (2 for fraud)
            num_heads: Number of attention heads
            dropout: Dropout rate
            metadata: (node_types, edge_types) for heterogeneous graph
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # HANConv layers
        # HANConv automatically handles meta-paths và hierarchical attention
        self.conv1 = HANConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            metadata=metadata
        )
        
        self.conv2 = HANConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            metadata=metadata
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass.
        
        Args:
            x_dict: Dict of node features {node_type: features}
            edge_index_dict: Dict of edge indices {edge_type: edge_index}
        
        Returns:
            logits: [N_transaction, num_classes]
        """
        # First HANConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                  for key, x in x_dict.items()}
        
        # Second HANConv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                  for key, x in x_dict.items()}
        
        # Classifier (only for transaction nodes)
        logits = self.classifier(x_dict['transaction'])
        
        return logits
    
    def predict(self, x_dict, edge_index_dict):
        """
        Predict with probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


class CustomHAN(nn.Module):
    """
    Custom HAN implementation với explicit meta-path handling.
    
    Meta-paths cho fraud detection:
    1. Transaction-User-Transaction (TUT)
    2. Transaction-Device-Transaction (TDT)
    3. Transaction-User-Device-Transaction (TUDT)
    """
    def __init__(self,
                 in_channels_dict,
                 hidden_channels,
                 num_classes=2,
                 num_heads=8,
                 num_meta_paths=3,
                 dropout=0.6):
        """
        Args:
            in_channels_dict: Dict {node_type: in_channels}
            hidden_channels: Hidden dimension
            num_classes: Number of classes
            num_heads: Number of attention heads
            num_meta_paths: Number of meta-paths
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_meta_paths = num_meta_paths
        self.dropout = dropout
        
        # Node-type specific projections
        self.node_projections = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.node_projections[node_type] = nn.Linear(in_channels, hidden_channels)
        
        # Meta-path specific GAT layers
        self.meta_path_layers = nn.ModuleList([
            nn.ModuleList([
                GATConv(hidden_channels, hidden_channels // num_heads, 
                       heads=num_heads, dropout=dropout, concat=True),
                GATConv(hidden_channels, hidden_channels // num_heads,
                       heads=num_heads, dropout=dropout, concat=True)
            ]) for _ in range(num_meta_paths)
        ])
        
        # Semantic-level attention
        self.semantic_attention = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def node_level_attention(self, x, edge_index, layers):
        """
        Node-level attention aggregation trong 1 meta-path.
        
        Args:
            x: Node features [N, D]
            edge_index: Edge indices [2, E]
            layers: List of GAT layers
        
        Returns:
            h: Aggregated features [N, D]
        """
        h = x
        for layer in layers:
            h = layer(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def semantic_level_attention(self, embeddings_list):
        """
        Semantic-level attention: Aggregate embeddings from different meta-paths.
        
        Args:
            embeddings_list: List of embeddings from each meta-path [P, N, D]
        
        Returns:
            z: Final embeddings [N, D]
            attention_weights: Semantic attention weights [P]
        """
        # Stack embeddings: [N, P, D]
        embeddings = torch.stack(embeddings_list, dim=1)
        
        # Compute attention weights for each meta-path
        # [N, P, D] -> [N, P, 1] -> [N, P]
        attention_scores = self.semantic_attention(embeddings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)  # [N, P]
        
        # Weighted sum: [N, P, D] * [N, P, 1] -> [N, D]
        z = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
        
        # Average attention weights across nodes
        avg_attention_weights = attention_weights.mean(dim=0)
        
        return z, avg_attention_weights
    
    def forward(self, x_dict, edge_index_dict, meta_path_dict):
        """
        Forward pass with meta-paths.
        
        Args:
            x_dict: Dict {node_type: features}
            edge_index_dict: Dict {edge_type: edge_index}
            meta_path_dict: Dict {meta_path_name: (nodes, edges)}
                Example: {
                    'TUT': (transaction_nodes, tut_edge_index),
                    'TDT': (transaction_nodes, tdt_edge_index),
                    'TUDT': (transaction_nodes, tudt_edge_index)
                }
        
        Returns:
            logits: [N_transaction, num_classes]
        """
        # Project node features to hidden space
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.node_projections[node_type](x)
        
        # Process each meta-path with node-level attention
        embeddings_list = []
        
        for i, (meta_path_name, (nodes, edge_index)) in enumerate(meta_path_dict.items()):
            # Get initial embeddings for nodes in this meta-path
            h = h_dict['transaction']  # All meta-paths end at transaction
            
            # Apply meta-path specific GAT layers
            h_meta = self.node_level_attention(h, edge_index, self.meta_path_layers[i])
            
            embeddings_list.append(h_meta)
        
        # Semantic-level attention to aggregate meta-paths
        z, attention_weights = self.semantic_level_attention(embeddings_list)
        
        # Store attention weights for analysis
        self.last_semantic_attention = attention_weights
        
        # Classifier
        logits = self.classifier(z)
        
        return logits
    
    def predict(self, x_dict, edge_index_dict, meta_path_dict):
        """
        Predict with probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict, meta_path_dict)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs
    
    def get_semantic_attention_weights(self):
        """
        Get semantic attention weights từ last forward pass.
        """
        if hasattr(self, 'last_semantic_attention'):
            return self.last_semantic_attention
        return None


class SimpleHAN(nn.Module):
    """
    Simplified HAN cho ease of use.
    Automatically extract meta-paths from heterogeneous graph.
    """
    def __init__(self,
                 num_nodes_dict,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_heads=8,
                 dropout=0.6):
        """
        Args:
            num_nodes_dict: Dict {node_type: num_nodes}
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            num_classes: Number of classes
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_nodes_dict = num_nodes_dict
        
        # Embeddings cho nodes không có features
        self.embeddings = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            if node_type != 'transaction':
                self.embeddings[node_type] = nn.Embedding(num_nodes, in_channels)
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers for each meta-path
        # Meta-path 1: Transaction-User-Transaction
        self.gat_tut = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels // num_heads, 
                   heads=num_heads, dropout=dropout, concat=True),
            GATConv(hidden_channels, hidden_channels // num_heads,
                   heads=num_heads, dropout=dropout, concat=True)
        ])
        
        # Meta-path 2: Transaction-Device-Transaction
        self.gat_tdt = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels // num_heads,
                   heads=num_heads, dropout=dropout, concat=True),
            GATConv(hidden_channels, hidden_channels // num_heads,
                   heads=num_heads, dropout=dropout, concat=True)
        ])
        
        # Semantic-level attention
        self.semantic_attention = nn.Parameter(torch.ones(2) / 2)  # 2 meta-paths
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x_dict, tut_edge_index, tdt_edge_index):
        """
        Forward pass với 2 meta-paths.
        
        Args:
            x_dict: Dict {node_type: features}
            tut_edge_index: Transaction-User-Transaction meta-path edges
            tdt_edge_index: Transaction-Device-Transaction meta-path edges
        
        Returns:
            logits: [N_transaction, num_classes]
        """
        # Get transaction features
        h = self.input_proj(x_dict['transaction'])
        
        # Meta-path 1: TUT
        h_tut = h
        for layer in self.gat_tut:
            h_tut = layer(h_tut, tut_edge_index)
            h_tut = F.elu(h_tut)
            h_tut = F.dropout(h_tut, p=self.dropout, training=self.training)
        
        # Meta-path 2: TDT
        h_tdt = h
        for layer in self.gat_tdt:
            h_tdt = layer(h_tdt, tdt_edge_index)
            h_tdt = F.elu(h_tdt)
            h_tdt = F.dropout(h_tdt, p=self.dropout, training=self.training)
        
        # Semantic-level aggregation
        semantic_weights = F.softmax(self.semantic_attention, dim=0)
        z = semantic_weights[0] * h_tut + semantic_weights[1] * h_tdt
        
        # Classifier
        logits = self.classifier(z)
        
        return logits
    
    def predict(self, x_dict, tut_edge_index, tdt_edge_index):
        """
        Predict with probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, tut_edge_index, tdt_edge_index)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs
    
    def get_semantic_weights(self):
        """
        Get semantic attention weights.
        """
        return F.softmax(self.semantic_attention, dim=0)


def extract_meta_paths(hetero_data):
    """
    Extract meta-path based edges từ heterogeneous graph.
    
    Meta-paths:
    1. Transaction-User-Transaction (TUT): T -> U -> T
    2. Transaction-Device-Transaction (TDT): T -> D -> T
    
    Args:
        hetero_data: PyG HeteroData
    
    Returns:
        tut_edge_index: TUT meta-path edges [2, E_tut]
        tdt_edge_index: TDT meta-path edges [2, E_tdt]
    """
    device = hetero_data['transaction'].x.device
    
    # Extract edges
    t_to_u = hetero_data[('transaction', 'performed_by', 'user')].edge_index
    u_to_t = hetero_data[('user', 'performs', 'transaction')].edge_index
    t_to_d = hetero_data[('transaction', 'uses', 'device')].edge_index
    d_to_t = hetero_data[('device', 'used_in', 'transaction')].edge_index
    
    # Build TUT: Transaction -> User -> Transaction
    # t1 -> u -> t2
    tut_edges = []
    for i in range(t_to_u.shape[1]):
        t1, u = t_to_u[:, i]
        # Find all t2 connected to u
        u_neighbors = u_to_t[1, u_to_t[0] == u]
        for t2 in u_neighbors:
            if t1 != t2:  # Avoid self-loops
                tut_edges.append([t1.item(), t2.item()])
    
    if len(tut_edges) > 0:
        tut_edge_index = torch.tensor(tut_edges, device=device).t()
    else:
        tut_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # Build TDT: Transaction -> Device -> Transaction
    tdt_edges = []
    for i in range(t_to_d.shape[1]):
        t1, d = t_to_d[:, i]
        # Find all t2 connected to d
        d_neighbors = d_to_t[1, d_to_t[0] == d]
        for t2 in d_neighbors:
            if t1 != t2:
                tdt_edges.append([t1.item(), t2.item()])
    
    if len(tdt_edges) > 0:
        tdt_edge_index = torch.tensor(tdt_edges, device=device).t()
    else:
        tdt_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return tut_edge_index, tdt_edge_index


def create_han_model(model_type='simple', **kwargs):
    """
    Factory function để tạo HAN model.
    
    Args:
        model_type: 'han', 'custom', or 'simple'
        **kwargs: Model arguments
    
    Returns:
        model: HAN model
    """
    if model_type == 'han':
        return HAN(**kwargs)
    elif model_type == 'custom':
        return CustomHAN(**kwargs)
    elif model_type == 'simple':
        return SimpleHAN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
