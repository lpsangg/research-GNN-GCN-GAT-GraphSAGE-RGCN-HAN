"""
GNN Module for Fraud Detection
"""

try:
    from .model import BasicGNN, ImprovedGNN, GNNLayer, create_gnn_model
except ImportError:
    from GNN.model import BasicGNN, ImprovedGNN, GNNLayer, create_gnn_model

__all__ = ['BasicGNN', 'ImprovedGNN', 'GNNLayer', 'create_gnn_model']
