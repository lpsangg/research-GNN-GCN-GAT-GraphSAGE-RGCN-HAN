"""
GAT (Graph Attention Network) Model for Fraud Detection
Based on Veličković et al. (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(nn.Module):
    """
    Graph Attention Network cho Fraud Detection.
    
    GAT sử dụng attention mechanism để học importance của neighbors:
    α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
    h_i' = σ(Σ_j α_ij W h_j)
    
    Trong đó:
    - α_ij là attention coefficient từ node j đến node i
    - W là learnable weight matrix
    - a là attention mechanism
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 heads=8,
                 concat=True,
                 negative_slope=0.2):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GAT layers
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
            heads: Số attention heads
            concat: Có concatenate multi-head outputs không (True cho hidden layers)
            negative_slope: Negative slope cho LeakyReLU trong attention
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GAT Layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, 
                    heads=heads, concat=concat,
                    dropout=dropout, negative_slope=negative_slope)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATConv(in_dim, hidden_channels,
                        heads=heads, concat=concat,
                        dropout=dropout, negative_slope=negative_slope)
            )
        
        # Last layer (average heads for final layer)
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATConv(in_dim, hidden_channels,
                        heads=heads, concat=False,  # Average heads for last layer
                        dropout=dropout, negative_slope=negative_slope)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers):
                if i < num_layers - 1:
                    bn_dim = hidden_channels * heads if concat else hidden_channels
                else:
                    bn_dim = hidden_channels
                self.bns.append(nn.BatchNorm1d(bn_dim))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
        
        Returns:
            logits: [N, num_classes]
        """
        # GAT Layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:  # No activation after last layer before classifier
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def predict(self, x, edge_index):
        """
        Predict method.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x, edge_index)
        return logits


class GATv2(nn.Module):
    """
    Graph Attention Network v2 for Fraud Detection.
    Uses GATv2Conv for improved attention mechanism.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 heads=8,
                 concat=True,
                 negative_slope=0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, 
                      heads=heads, concat=concat,
                      dropout=dropout, negative_slope=negative_slope)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels,
                          heads=heads, concat=concat,
                          dropout=dropout, negative_slope=negative_slope)
            )
        
        # Last layer (average heads for final layer)
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels,
                          heads=heads, concat=False,  # Average heads for last layer
                          dropout=dropout, negative_slope=negative_slope)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers):
                if i < num_layers - 1:
                    bn_dim = hidden_channels * heads if concat else hidden_channels
                else:
                    bn_dim = hidden_channels
                self.bns.append(nn.BatchNorm1d(bn_dim))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.classifier(x)
        
        return x
    
    def predict(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
        return logits


class MultiHeadGAT(nn.Module):
    """
    Placeholder for MultiHeadGAT implementation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("MultiHeadGAT placeholder initialized.")

    def forward(self, x, edge_index):
        print("MultiHeadGAT forward pass placeholder.")
        return x # Placeholder return


def create_gat_model(model_name, in_channels, hidden_channels, num_classes, num_layers, dropout, use_batch_norm, heads, concat, negative_slope):
    """
    Factory function to create different GAT models.
    """
    if model_name == 'GAT':
        return GAT(in_channels, hidden_channels, num_classes, num_layers, dropout, use_batch_norm, heads, concat, negative_slope)
    elif model_name == 'GATv2':
        return GATv2(in_channels, hidden_channels, num_classes, num_layers, dropout, use_batch_norm, heads, concat, negative_slope)
    # Add other model types as needed
    else:
        raise ValueError(f"Unknown GAT model type: {model_name}")