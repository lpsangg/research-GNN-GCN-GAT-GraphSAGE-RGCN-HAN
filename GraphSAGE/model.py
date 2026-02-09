"""
GraphSAGE (SAmple and aggreGatE) Model for Fraud Detection
Based on Hamilton et al. (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Sequential, Linear, ReLU


class GraphSAGE(nn.Module):
    """
    GraphSAGE với Mean aggregator (most common variant).
    
    GraphSAGE algorithm:
    1. Sample neighborhood
    2. Aggregate neighbor features
    3. Concatenate with self features
    4. Apply transformation
    
    h_N(v) = AGGREGATE({h_u : u ∈ N(v)})
    h_v' = σ(W · [h_v || h_N(v)])
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 aggregator='mean'):
        """
        Args:
            in_channels: Số features đầu vào
            hidden_channels: Số hidden units
            num_classes: Số classes (2 cho fraud detection)
            num_layers: Số GraphSAGE layers
            dropout: Dropout rate
            use_batch_norm: Có dùng Batch Normalization không
            aggregator: 'mean', 'max', 'lstm' (mean is default và most stable)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # GraphSAGE Layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        
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
        # GraphSAGE Layers
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
    
    def inference(self, x_all, subgraph_loader, device):
        """
        Mini-batch inference cho large graphs.
        
        Args:
            x_all: All node features
            subgraph_loader: NeighborLoader
            device: Device
        
        Returns:
            logits: Predictions cho tất cả nodes
        """
        self.eval()
        
        # Compute representations layer by layer
        for i in range(self.num_layers):
            xs = []
            
            for batch in subgraph_loader:
                batch = batch.to(device)
                x = batch.x
                
                # Forward through current layer only
                x = self.convs[i](x, batch.edge_index)
                
                if self.use_batch_norm:
                    x = self.bns[i](x)
                
                x = F.relu(x)
                
                # Only keep target nodes
                xs.append(x[:batch.batch_size])
            
            x_all = torch.cat(xs, dim=0)
        
        # Final classifier
        return self.classifier(x_all)


class DeepGraphSAGE(nn.Module):
    """
    Deep GraphSAGE với residual connections.
    Inspired by JK-Net và ResNet.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=3,
                 dropout=0.5,
                 use_batch_norm=True,
                 aggregator='mean',
                 jk_mode='cat'):
        """
        Args:
            jk_mode: Jumping knowledge mode - 'cat', 'max', or None
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.jk_mode = jk_mode
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GraphSAGE Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Jumping Knowledge
        if jk_mode == 'cat':
            classifier_in = hidden_channels * num_layers
        else:
            classifier_in = hidden_channels
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass với Jumping Knowledge.
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x_0 = x.clone()
        
        # Collect layer outputs
        layer_outputs = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_0
            
            layer_outputs.append(x)
        
        # Jumping Knowledge aggregation
        if self.jk_mode == 'cat':
            x = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == 'max':
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        else:
            x = layer_outputs[-1]
        
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


class AttentionGraphSAGE(nn.Module):
    """
    GraphSAGE với attention-based aggregation.
    Combines benefits của GraphSAGE và GAT.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 heads=4):
        """
        Args:
            heads: Number of attention heads
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.heads = heads
        
        # Attention layers
        self.attentions = nn.ModuleList()
        
        # First layer
        for _ in range(heads):
            self.attentions.append(
                nn.Sequential(
                    nn.Linear(in_channels * 2, hidden_channels),
                    nn.Tanh(),
                    nn.Linear(hidden_channels, 1)
                )
            )
        
        # GraphSAGE Layers với custom aggregation
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels * heads, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass với attention aggregation.
        """
        # Multi-head attention aggregation
        head_outputs = []
        
        for head_attn in self.attentions:
            # Aggregate neighbors với attention
            aggregated = self.attention_aggregate(x, edge_index, head_attn)
            head_outputs.append(aggregated)
        
        # Concatenate heads
        x = torch.cat(head_outputs, dim=-1)
        
        # Layers
        for i in range(self.num_layers):
            x = self.convs[i](x)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def attention_aggregate(self, x, edge_index, attention_layer):
        """
        Aggregate neighbors với attention weights.
        """
        row, col = edge_index
        
        # Compute attention scores
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        attention_scores = attention_layer(edge_features).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        # Aggregate
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, row, attention_weights.unsqueeze(-1) * x[col])
        
        return aggregated
    
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


class MultiAggregatorSAGE(nn.Module):
    """
    GraphSAGE với multiple aggregators combined.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 use_batch_norm=True,
                 aggregators=['mean', 'max', 'min']):
        """
        Args:
            aggregators: List of aggregators to use and combine
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.aggregators = aggregators
        
        # Multiple aggregator layers
        self.conv_lists = nn.ModuleList()
        for _ in range(num_layers):
            layer_convs = nn.ModuleList()
            for aggr in aggregators:
                if _ == 0:
                    layer_convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
                else:
                    layer_convs.append(SAGEConv(hidden_channels * len(aggregators), 
                                                hidden_channels, aggr=aggr))
            self.conv_lists.append(layer_convs)
        
        # Batch Normalization
        if use_batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels * len(aggregators)))
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels * len(aggregators), num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass với multiple aggregators.
        """
        for i in range(self.num_layers):
            # Apply all aggregators
            aggr_outputs = []
            for conv in self.conv_lists[i]:
                aggr_outputs.append(conv(x, edge_index))
            
            # Concatenate aggregator outputs
            x = torch.cat(aggr_outputs, dim=-1)
            
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
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            pred_probs = probs[:, 1]
        
        return pred_labels, pred_probs


def create_graphsage_model(model_type='standard', **kwargs):
    """
    Factory function để tạo GraphSAGE model.
    
    Args:
        model_type: 'standard', 'deep', 'attention', or 'multi_aggr'
        **kwargs: Arguments cho model
    
    Returns:
        model: GraphSAGE model
    """
    if model_type == 'standard':
        return GraphSAGE(**kwargs)
    elif model_type == 'deep':
        return DeepGraphSAGE(**kwargs)
    elif model_type == 'attention':
        return AttentionGraphSAGE(**kwargs)
    elif model_type == 'multi_aggr':
        return MultiAggregatorSAGE(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
