"""
RGCN (Relational Graph Convolutional Network) Model for Fraud Detection
Based on Schlichtkrull et al. (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv


class RGCN(nn.Module):
    """
    Relational GCN cho Heterogeneous Graph Fraud Detection.
    
    RGCN extends GCN để handle multiple edge types:
    h_i^(l+1) = σ(Σ_r∈R Σ_j∈N_i^r (1/c_i,r) W_r^(l) h_j^(l) + W_0^(l) h_i^(l))
    
    Trong đó:
    - R là set of relation types
    - N_i^r là neighbors của node i under relation r
    - W_r^(l) là weight matrix cho relation r at layer l
    - c_i,r là normalization constant
    """
    def __init__(self,
                 num_nodes_dict,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 num_relations=4,
                 num_bases=None,
                 dropout=0.5,
                 use_batch_norm=True):
        """
        Args:
            num_nodes_dict: Dict {node_type: num_nodes}
            in_channels: Số features đầu vào cho transactions
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số RGCN layers
            num_relations: Số relation types
            num_bases: Number of bases cho basis-decomposition (giảm params)
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_nodes_dict = num_nodes_dict
        
        # Embedding layers cho các node types không có features
        self.embeddings = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            if node_type != 'transaction':  # Transaction có features
                self.embeddings[node_type] = nn.Embedding(num_nodes, in_channels)
        
        # RGCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        
        if num_layers > 1:
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x_dict, edge_index, edge_type):
        """
        Forward pass cho heterogeneous graph.
        
        Args:
            x_dict: Dict of node features {node_type: features}
            edge_index: Edge indices [2, E]
            edge_type: Edge types [E]
        
        Returns:
            logits: [N_transaction, num_classes]
        """
        # Prepare node features
        # Combine all node features into single tensor
        x_list = []
        node_offset = 0
        
        for node_type in ['transaction', 'user', 'device']:
            if node_type in x_dict:
                if node_type == 'transaction':
                    x_list.append(x_dict[node_type])
                else:
                    # Use embeddings cho user và device
                    num_nodes = self.num_nodes_dict[node_type]
                    node_ids = torch.arange(num_nodes, device=x_dict['transaction'].device)
                    x_list.append(self.embeddings[node_type](node_ids))
        
        x = torch.cat(x_list, dim=0)
        
        # RGCN Layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Extract transaction node embeddings
        num_transactions = x_dict['transaction'].shape[0]
        x_transaction = x[:num_transactions]
        
        # Classifier
        logits = self.classifier(x_transaction)
        
        return logits
    
    def predict(self, x_dict, edge_index, edge_type):
        """
        Predict với probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index, edge_type)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


class FastRGCN(nn.Module):
    """
    Fast RGCN variant với basis-decomposition và block-diagonal matrices.
    Faster và memory efficient hơn standard RGCN.
    """
    def __init__(self,
                 num_nodes_dict,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 num_relations=4,
                 num_bases=30,
                 dropout=0.5,
                 use_batch_norm=True):
        """
        Args:
            num_bases: Number of bases (nên set < num_relations)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_nodes_dict = num_nodes_dict
        
        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            if node_type != 'transaction':
                self.embeddings[node_type] = nn.Embedding(num_nodes, in_channels)
        
        # Fast RGCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(
            FastRGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                FastRGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        
        if num_layers > 1:
            self.convs.append(
                FastRGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x_dict, edge_index, edge_type):
        """
        Forward pass.
        """
        # Prepare node features
        x_list = []
        
        for node_type in ['transaction', 'user', 'device']:
            if node_type in x_dict:
                if node_type == 'transaction':
                    x_list.append(x_dict[node_type])
                else:
                    num_nodes = self.num_nodes_dict[node_type]
                    node_ids = torch.arange(num_nodes, device=x_dict['transaction'].device)
                    x_list.append(self.embeddings[node_type](node_ids))
        
        x = torch.cat(x_list, dim=0)
        
        # Fast RGCN Layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Extract transaction embeddings
        num_transactions = x_dict['transaction'].shape[0]
        x_transaction = x[:num_transactions]
        
        # Classifier
        logits = self.classifier(x_transaction)
        
        return logits
    
    def predict(self, x_dict, edge_index, edge_type):
        """
        Predict với probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index, edge_type)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


class TypedRGCN(nn.Module):
    """
    RGCN với type-specific transformations.
    Separate processing cho mỗi node type trước khi message passing.
    """
    def __init__(self,
                 num_nodes_dict,
                 in_channels_dict,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 num_relations=4,
                 num_bases=None,
                 dropout=0.5,
                 use_batch_norm=True):
        """
        Args:
            in_channels_dict: Dict {node_type: in_channels}
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_nodes_dict = num_nodes_dict
        self.node_types = list(num_nodes_dict.keys())
        
        # Type-specific input projections
        self.input_projs = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.input_projs[node_type] = nn.Linear(in_channels, hidden_channels)
        
        # RGCN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Type-specific output projections
        self.output_projs = nn.ModuleDict()
        for node_type in num_nodes_dict.keys():
            self.output_projs[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Classifier (chỉ cho transaction)
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x_dict, edge_index, edge_type):
        """
        Forward pass với type-specific transformations.
        """
        # Type-specific input projections
        x_list = []
        node_type_mapping = {}
        current_offset = 0
        
        for node_type in self.node_types:
            if node_type in x_dict:
                x_transformed = self.input_projs[node_type](x_dict[node_type])
                x_list.append(x_transformed)
                
                num_nodes = x_dict[node_type].shape[0]
                node_type_mapping[node_type] = (current_offset, current_offset + num_nodes)
                current_offset += num_nodes
        
        x = torch.cat(x_list, dim=0)
        
        # RGCN Layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Type-specific output projections
        x_dict_out = {}
        for node_type, (start, end) in node_type_mapping.items():
            x_type = x[start:end]
            x_dict_out[node_type] = self.output_projs[node_type](x_type)
        
        # Classifier cho transaction nodes
        logits = self.classifier(x_dict_out['transaction'])
        
        return logits
    
    def predict(self, x_dict, edge_index, edge_type):
        """
        Predict với probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index, edge_type)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


def prepare_rgcn_data(hetero_data):
    """
    Convert HeteroData to format for RGCN.
    
    Args:
        hetero_data: PyG HeteroData object
    
    Returns:
        x_dict: Dict of node features
        edge_index: Combined edge indices
        edge_type: Edge type labels
        num_nodes_dict: Dict of number of nodes per type
    """
    # Extract node features
    x_dict = {
        'transaction': hetero_data['transaction'].x,
    }
    
    # Get number of nodes
    num_nodes_dict = {
        'transaction': hetero_data['transaction'].x.shape[0],
        'user': hetero_data['user'].num_nodes,
        'device': hetero_data['device'].num_nodes,
    }
    
    # Combine edge indices và assign edge types
    edge_index_list = []
    edge_type_list = []
    
    # Map edge types to integers
    edge_type_mapping = {
        ('user', 'performs', 'transaction'): 0,
        ('transaction', 'performed_by', 'user'): 1,
        ('device', 'used_in', 'transaction'): 2,
        ('transaction', 'uses', 'device'): 3,
    }
    
    # Offset cho node indices
    transaction_offset = 0
    user_offset = num_nodes_dict['transaction']
    device_offset = user_offset + num_nodes_dict['user']
    
    for edge_type_name, edge_type_idx in edge_type_mapping.items():
        if edge_type_name in hetero_data.edge_types:
            edge_index = hetero_data[edge_type_name].edge_index.clone()
            
            # Apply offsets
            src_type, _, dst_type = edge_type_name
            if src_type == 'user':
                edge_index[0] += user_offset
            elif src_type == 'device':
                edge_index[0] += device_offset
            
            if dst_type == 'user':
                edge_index[1] += user_offset
            elif dst_type == 'device':
                edge_index[1] += device_offset
            
            edge_index_list.append(edge_index)
            edge_type_list.append(
                torch.full((edge_index.shape[1],), edge_type_idx, dtype=torch.long)
            )
    
    # Combine
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_type = torch.cat(edge_type_list, dim=0)
    
    return x_dict, edge_index, edge_type, num_nodes_dict


def create_rgcn_model(model_type='standard', **kwargs):
    """
    Factory function để tạo RGCN model.
    
    Args:
        model_type: 'standard', 'fast', or 'typed'
        **kwargs: Arguments cho model
    
    Returns:
        model: RGCN model
    """
    if model_type == 'standard':
        return RGCN(**kwargs)
    elif model_type == 'fast':
        return FastRGCN(**kwargs)
    elif model_type == 'typed':
        return TypedRGCN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
