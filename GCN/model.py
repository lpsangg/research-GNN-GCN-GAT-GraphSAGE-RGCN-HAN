"""
GCN (Graph Convolutional Network) Model for Fraud Detection
Based on Kipf & Welling (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network cho Fraud Detection.
    
    GCN sử dụng normalized adjacency matrix để aggregate information:
    H^(l+1) = σ(D^(-1/2) Ã D^(-1/2) H^(l) W^(l))
    
    Trong đó:
    - Ã = A + I (adjacency matrix + self-loops)
    - D là degree matrix
    - W^(l) là learnable weight matrix
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 improved=False,
                 cached=False):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GCN layers
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
            improved: Có dùng improved GCN variant không
            cached: Cache normalized adjacency matrix (tốt cho full-batch training)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, 
                                  improved=improved, cached=cached))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                      improved=improved, cached=cached))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                  improved=improved, cached=cached))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
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
        # GCN Layers
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
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs
    
    def reset_parameters(self):
        """
        Reset all parameters.
        """
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.use_batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        
        self.classifier.reset_parameters()


class DeepGCN(nn.Module):
    """
    Deep GCN với residual connections để train deeper networks.
    Inspired by ResGCN và GCNII.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=4,
                 dropout=0.5,
                 use_batch_norm=True,
                 alpha=0.1,
                 theta=0.5):
        """
        Args:
            in_channels: Input features
            hidden_channels: Hidden dimensions
            num_classes: Number of classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_batch_norm: Use batch normalization
            alpha: Weight for initial residual connection
            theta: Weight for identity mapping
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.alpha = alpha
        self.theta = theta
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Layer-wise transformation weights
        self.lins = nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass với residual connections.
        
        Implements: H^(l+1) = σ((1-α-θ)·GCN(H^(l)) + α·H^(0) + θ·H^(l))
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x_0 = x.clone()  # Initial embedding
        
        # Deep GCN layers với residual
        for i in range(self.num_layers):
            x_prev = x.clone()
            
            # GCN propagation
            x = self.convs[i](x, edge_index)
            
            # Linear transformation
            x = self.lins[i](x)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            # Residual connections
            # x = (1 - alpha - theta) * x + alpha * x_0 + theta * x_prev
            x = (1 - self.alpha - self.theta) * x + self.alpha * x_0 + self.theta * x_prev
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
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


class JKNetGCN(nn.Module):
    """
    GCN với Jumping Knowledge Network.
    Combine representations từ tất cả các layers.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=3,
                 dropout=0.5,
                 use_batch_norm=True,
                 jk_mode='cat'):
        """
        Args:
            in_channels: Input features
            hidden_channels: Hidden dimensions
            num_classes: Number of classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_batch_norm: Use batch normalization
            jk_mode: 'cat', 'max', or 'lstm'
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.jk_mode = jk_mode
        
        # GCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Jumping Knowledge aggregation
        if jk_mode == 'cat':
            self.jk_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        elif jk_mode == 'lstm':
            self.jk_lstm = nn.LSTM(hidden_channels, hidden_channels, 
                                   batch_first=True, bidirectional=False)
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass với Jumping Knowledge.
        """
        # Collect layer outputs
        layer_outputs = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            layer_outputs.append(x)
        
        # Jumping Knowledge aggregation
        if self.jk_mode == 'cat':
            x = torch.cat(layer_outputs, dim=-1)
            x = self.jk_proj(x)
        elif self.jk_mode == 'max':
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == 'lstm':
            x = torch.stack(layer_outputs, dim=1)  # [N, num_layers, hidden]
            x, _ = self.jk_lstm(x)
            x = x[:, -1, :]  # Take last output
        
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


def create_gcn_model(model_type='standard', **kwargs):
    """
    Factory function để tạo GCN model.
    
    Args:
        model_type: 'standard', 'deep', hoặc 'jknet'
        **kwargs: Arguments cho model
    
    Returns:
        model: GCN model
    """
    if model_type == 'standard':
        return GCN(**kwargs)
    elif model_type == 'deep':
        return DeepGCN(**kwargs)
    elif model_type == 'jknet':
        return JKNetGCN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
