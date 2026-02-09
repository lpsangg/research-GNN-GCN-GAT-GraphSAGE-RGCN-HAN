"""
Basic GNN Model for Fraud Detection
Sử dụng Message Passing cơ bản
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNNLayer(MessagePassing):
    """
    Basic GNN Layer với Message Passing cơ bản.
    Tương tự như GCN nhưng đơn giản hơn.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # Aggregation: mean của messages từ neighbors
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node features
        x = self.lin(x)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j has shape [E, out_channels]
        # E là số edges
        return x_j
    
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


class BasicGNN(nn.Module):
    """
    Basic GNN Model cho Fraud Detection.
    Architecture: Input -> GNN Layers -> MLP -> Output
    """
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GNN layers
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GNN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GNNLayer(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
        
        Returns:
            logits: [N, num_classes]
        """
        # GNN Layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def predict(self, x, edge_index):
        """
        Predict với probabilities.
        
        Returns:
            pred_labels: Predicted labels
            pred_probs: Predicted probabilities cho class 1 (fraud)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]  # Probability của fraud class
        
        return pred_labels, pred_probs
    
    def reset_parameters(self):
        """
        Reset all parameters.
        """
        for conv in self.convs:
            conv.lin.reset_parameters()
        
        if self.use_batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ImprovedGNN(nn.Module):
    """
    Improved GNN với skip connections và attention-like mechanism.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=3,
                 dropout=0.5,
                 use_batch_norm=True,
                 use_residual=True):
        """
        Args:
            in_channels: Input features
            hidden_channels: Hidden dimensions
            num_classes: Number of classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_batch_norm: Use batch normalization
            use_residual: Use residual connections
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Attention weights cho các layers (learnable)
        if use_residual:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass với residual connections.
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Store initial embedding
        x_init = x.clone()
        
        # GNN Layers với residual connections
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual and i > 0:
                x = x + x_init  # Skip connection từ input
            
            layer_outputs.append(x)
        
        # Weighted combination of layer outputs
        if self.use_residual:
            weights = F.softmax(self.layer_weights, dim=0)
            x = sum(w * out for w, out in zip(weights, layer_outputs))
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def predict(self, x, edge_index):
        """
        Predict với probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


def create_gnn_model(model_type='basic', **kwargs):
    """
    Factory function để tạo GNN model.
    
    Args:
        model_type: 'basic' hoặc 'improved'
        **kwargs: Arguments cho model
    
    Returns:
        model: GNN model
    """
    if model_type == 'basic':
        return BasicGNN(**kwargs)
    elif model_type == 'improved':
        return ImprovedGNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
