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


class GATv2(nn.Module):
    """
    GATv2 (Graph Attention Network v2) cho Fraud Detection.
    
    GATv2 cải tiến attention mechanism so với GAT:
    - GAT: α_ij = softmax(a^T LeakyReLU([W h_i || W h_j]))
    - GATv2: α_ij = softmax(a^T LeakyReLU(W [h_i || h_j]))
    
    GATv2 có dynamic attention và expressive hơn GAT.
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
                 negative_slope=0.2,
                 share_weights=False):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GATv2 layers
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
            heads: Số attention heads
            concat: Có concatenate multi-head outputs không
            negative_slope: Negative slope cho LeakyReLU
            share_weights: Share weights between source and target nodes
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GATv2 Layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, 
                      heads=heads, concat=concat,
                      dropout=dropout, negative_slope=negative_slope,
                      share_weights=share_weights)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels,
                          heads=heads, concat=concat,
                          dropout=dropout, negative_slope=negative_slope,
                          share_weights=share_weights)
            )
        
        # Last layer (average heads for final layer)
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat else hidden_channels
            self.convs.append(
                GATv2Conv(in_dim, hidden_channels,
                          heads=heads, concat=False,  # Average heads for last layer
                          dropout=dropout, negative_slope=negative_slope,
                          share_weights=share_weights)
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
        # GATv2 Layers
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


class DeepGAT(nn.Module):
    """
    Deep GAT với residual connections để train deeper networks.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=4,
                 dropout=0.5,
                 use_batch_norm=True,
                 heads=8,
                 alpha=0.1):
        """
        Args:
            in_channels: Input features
            hidden_channels: Hidden dimensions
            num_classes: Number of classes
            num_layers: Number of GAT layers
            dropout: Dropout rate
            use_batch_norm: Use batch normalization
            heads: Number of attention heads
            alpha: Weight for residual connection
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.alpha = alpha
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT Layers (all with concat=False for easier residual connections)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels,
                        heads=heads, concat=False, dropout=dropout)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass với residual connections.
        """
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        x_0 = x.clone()  # Initial embedding
        
        # Deep GAT layers với residual
        for i in range(self.num_layers):
            x_prev = x.clone()
            
            # GAT propagation
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            # Residual connection
            x = (1 - self.alpha) * x + self.alpha * x_prev
            
            x = F.elu(x)
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


class MultiHeadGAT(nn.Module):
    """
    GAT với independent multi-head attention layers.
    Mỗi head có classifier riêng và ensemble predictions.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 num_heads=8):
        """
        Args:
            num_heads: Number of independent attention heads
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_heads = num_heads
        
        # Independent GAT branches for each head
        self.head_branches = nn.ModuleList()
        for _ in range(num_heads):
            branch_convs = nn.ModuleList()
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    branch_convs.append(
                        GATConv(in_channels, hidden_channels, 
                                heads=1, concat=True, dropout=dropout)
                    )
                else:
                    branch_convs.append(
                        GATConv(hidden_channels, hidden_channels,
                                heads=1, concat=True, dropout=dropout)
                    )
            self.head_branches.append(branch_convs)
        
        # Batch Normalization per head
        if use_batch_norm:
            self.head_bns = nn.ModuleList()
            for _ in range(num_heads):
                head_bn = nn.ModuleList()
                for _ in range(num_layers):
                    head_bn.append(nn.BatchNorm1d(hidden_channels))
                self.head_bns.append(head_bn)
        
        # Ensemble classifier
        self.ensemble_classifier = nn.Linear(hidden_channels * num_heads, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass với multiple independent heads.
        """
        head_outputs = []
        
        # Process each head independently
        for head_idx in range(self.num_heads):
            h = x
            for layer_idx in range(self.num_layers):
                h = self.head_branches[head_idx][layer_idx](h, edge_index)
                
                if self.use_batch_norm:
                    h = self.head_bns[head_idx][layer_idx](h)
                
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            head_outputs.append(h)
        
        # Combine all heads
        x = torch.cat(head_outputs, dim=-1)
        
        # Ensemble classifier
        x = self.ensemble_classifier(x)
        
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


class HierarchicalGAT(nn.Module):
    """
    GAT với hierarchical attention: local attention -> global attention.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 local_heads=4,
                 global_heads=2):
        """
        Args:
            local_heads: Number of heads for local attention
            global_heads: Number of heads for global attention
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Local attention layers (more heads, smaller receptive field)
        self.local_convs = nn.ModuleList()
        self.local_convs.append(
            GATConv(in_channels, hidden_channels,
                    heads=local_heads, concat=True, dropout=dropout)
        )
        
        # Global attention layers (fewer heads, larger context)
        self.global_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            in_dim = hidden_channels * local_heads if len(self.global_convs) == 0 else hidden_channels * global_heads
            is_last = len(self.global_convs) == num_layers - 2
            self.global_convs.append(
                GATConv(in_dim, hidden_channels,
                        heads=global_heads, 
                        concat=not is_last,  # Average for last layer
                        dropout=dropout)
            )
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hidden_channels * local_heads))
            for i in range(len(self.global_convs)):
                is_last = i == len(self.global_convs) - 1
                bn_dim = hidden_channels if is_last else hidden_channels * global_heads
                self.bns.append(nn.BatchNorm1d(bn_dim))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass với hierarchical attention.
        """
        # Local attention
        x = self.local_convs[0](x, edge_index)
        if self.use_batch_norm:
            x = self.bns[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global attention
        for i, conv in enumerate(self.global_convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.bns[i + 1](x)
            
            if i < len(self.global_convs) - 1:
                x = F.elu(x)
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


def create_gat_model(model_type='standard', **kwargs):
    """
    Factory function để tạo GAT model.
    
    Args:
        model_type: 'standard', 'gatv2', 'deep', 'multi_head', hoặc 'hierarchical'
        **kwargs: Arguments cho model
    
    Returns:
        model: GAT model
    """
    if model_type == 'standard':
        return GAT(**kwargs)
    elif model_type == 'gatv2':
        return GATv2(**kwargs)
    elif model_type == 'deep':
        return DeepGAT(**kwargs)
    elif model_type == 'multi_head':
        return MultiHeadGAT(**kwargs)
    elif model_type == 'hierarchical':
        return HierarchicalGAT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
