"""
GCN Module for Fraud Detection
"""

try:
    from .model import GCN, DeepGCN, JKNetGCN, create_gcn_model
except ImportError:
    from GCN.model import GCN, DeepGCN, JKNetGCN, create_gcn_model

__all__ = ['GCN', 'DeepGCN', 'JKNetGCN', 'create_gcn_model']
