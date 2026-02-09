"""
GAT (Graph Attention Network) Model for Fraud Detection
Based on Veličković et al. (2018) and GATv2 (Brody et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(nn.Module):
    """
    Graph Attention Network cho Fraud Detection.
    
    GAT sử dụng attention mechanism để tự động học importance của neighbors:
    α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    h_i' = σ(Σ_j α_ij W h_j)
    
    Multi-head attention để capture different aspects:
    h_i' = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 heads=8,
                 dropout=0.6,
                 use_batch_norm=True,
                 concat_heads=True,
                 negative_slope=0.2):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units (per head)
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GAT layers
            heads: Số attention heads
            dropout: Dropout rate (GAT thường dùng cao, ~0.6)
            use_batch_norm: Có dùng Batch Normalization không
            concat_heads: Concatenate multi-head outputs (True) hoặc average (False)
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
            GATConv(in_channels, hidden_channels, heads=heads,
                   dropout=dropout, concat=concat_heads,
                   negative_slope=negative_slope)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATConv(in_dim, hidden_channels, heads=heads,
                       dropout=dropout, concat=concat_heads,
                       negative_slope=negative_slope)
            )
        
        # Last layer (average heads for final layer)
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATConv(in_dim, hidden_channels, heads=heads,
                       dropout=dropout, concat=False,  # Average cho output layer
                       negative_slope=negative_slope)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers - 1):
                in_dim = hidden_channels * heads if concat_heads else hidden_channels
                self.bns.append(nn.BatchNorm1d(in_dim))
            # Last layer
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
        # GAT Layers
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:  # No activation after last layer
                x = F.elu(x)
        
        # Classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
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
    
    def get_attention_weights(self, x, edge_index):
        """
        Lấy attention weights để visualize.
        
        Returns:
            attention_weights: List of attention weights cho mỗi layer
        """
        self.eval()
        attention_weights = []
        
        with torch.no_grad():
            for i in range(self.num_layers):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x, (edge_idx, alpha) = self.convs[i](x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, alpha))
                
                if self.use_batch_norm:
                    x = self.bns[i](x)
                
                if i < self.num_layers - 1:
                    x = F.elu(x)
        
        return attention_weights


class GATv2(nn.Module):
    """
    GATv2 - Improved version của GAT.
    
    GATv2 fixes the static attention problem of GAT:
    α_ij = softmax(a^T LeakyReLU(W [h_i || h_j]))
    
    Key difference: LeakyReLU applied AFTER weight matrix multiplication.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 heads=8,
                 dropout=0.6,
                 use_batch_norm=True,
                 concat_heads=True,
                 negative_slope=0.2,
                 share_weights=False):
        """
        Args:
            share_weights: Share weights giữa source và target nodes
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GATv2 Layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads,
                     dropout=dropout, concat=concat_heads,
                     negative_slope=negative_slope,
                     share_weights=share_weights)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels, heads=heads,
                         dropout=dropout, concat=concat_heads,
                         negative_slope=negative_slope,
                         share_weights=share_weights)
            )
        
        # Last layer
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels, heads=heads,
                         dropout=dropout, concat=False,
                         negative_slope=negative_slope,
                         share_weights=share_weights)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers - 1):
                in_dim = hidden_channels * heads if concat_heads else hidden_channels
                self.bns.append(nn.BatchNorm1d(in_dim))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        """
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
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


class MultiHeadGAT(nn.Module):
    """
    GAT với flexible multi-head aggregation strategies.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 heads=8,
                 dropout=0.6,
                 use_batch_norm=True,
                 head_aggregation='concat'):
        """
        Args:
            head_aggregation: 'concat', 'mean', 'max', 'attention'
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.heads = heads
        self.head_aggregation = head_aggregation
        
        # GAT Layers (always concat within layer)
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads,
                   dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                       dropout=dropout, concat=True)
            )
        
        # Last layer
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                       dropout=dropout, concat=True)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Head aggregation layers
        if head_aggregation == 'attention':
            self.head_attention = nn.Linear(hidden_channels, 1)
        
        # Classifier
        if head_aggregation == 'concat':
            self.classifier = nn.Linear(hidden_channels * heads, num_classes)
        else:
            self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def aggregate_heads(self, x):
        """
        Aggregate multi-head outputs.
        
        Args:
            x: [N, hidden_channels * heads]
        
        Returns:
            x: [N, hidden_channels] or [N, hidden_channels * heads]
        """
        if self.head_aggregation == 'concat':
            return x
        
        # Reshape to [N, heads, hidden_channels]
        N = x.size(0)
        x = x.view(N, self.heads, -1)
        
        if self.head_aggregation == 'mean':
            x = x.mean(dim=1)
        elif self.head_aggregation == 'max':
            x = x.max(dim=1)[0]
        elif self.head_aggregation == 'attention':
            # Learned attention over heads
            attn_weights = self.head_attention(x)  # [N, heads, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            x = (x * attn_weights).sum(dim=1)  # [N, hidden_channels]
        
        return x
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        """
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        # Aggregate heads
        x = self.aggregate_heads(x)
        
        # Classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
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


def create_gat_model(model_type='standard', **kwargs):
    """
    Factory function để tạo GAT model.
    
    Args:
        model_type: 'standard', 'v2', hoặc 'multihead'
        **kwargs: Arguments cho model
    
    Returns:
        model: GAT model
    """
    if model_type == 'standard':
        return GAT(**kwargs)
    elif model_type == 'v2':
        return GATv2(**kwargs)
    elif model_type == 'multihead':
        return MultiHeadGAT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
