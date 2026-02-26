"""
Model Registry and Configuration
Centralized configuration for all 6 GNN models
"""

from GNN.train import train_gnn
from GCN.train import train_gcn
from GAT.train import train_gat
from GraphSAGE.train import train_graphsage
from RGCN.train import train_rgcn
from HAN.train import train_han


MODEL_CONFIGS = {
    'GNN': {
        'train_func': train_gnn,
        'type': 'Homogeneous',
        'description': 'Basic Graph Neural Network',
        'config': {
            'checkpoint_dir': 'checkpoints/GNN',
            'model_type': 'improved',  # 'basic' or 'improved'
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.001,
            'epochs': 200,
            'patience': 20,
        }
    },
    
    'GCN': {
        'train_func': train_gcn,
        'type': 'Homogeneous',
        'description': 'Graph Convolutional Network',
        'config': {
            'checkpoint_dir': 'checkpoints/GCN',
            'model_type': 'jknet',  # 'standard', 'deep', or 'jknet'
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 20,
            'use_cached': True,
        }
    },
    
    'GAT': {
        'train_func': train_gat,
        'type': 'Homogeneous',
        'description': 'Graph Attention Network',
        'config': {
            'checkpoint_dir': 'checkpoints/GAT',
            'model_type': 'gatv2',  # 'standard', 'gatv2', or 'multihead'
            'hidden_channels': 64,
            'num_layers': 2,
            'heads': 8,
            'dropout': 0.6,
            'lr': 0.005,
            'epochs': 200,
            'patience': 30,
        }
    },
    
    'GraphSAGE': {
        'train_func': train_graphsage,
        'type': 'Homogeneous',
        'description': 'Sampling-based Inductive Learning',
        'config': {
            'checkpoint_dir': 'checkpoints/GraphSAGE',
            'model_type': 'attention',  # 'standard', 'deep', 'attention', or 'multi'
            'hidden_channels': 64,
            'num_layers': 2,
            'aggregator': 'mean',  # 'mean', 'max', 'min', 'lstm'
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 20,
        }
    },
    
    'RGCN': {
        'train_func': train_rgcn,
        'type': 'Heterogeneous',
        'description': 'Relational Graph Convolutional Network',
        'config': {
            'checkpoint_dir': 'checkpoints/RGCN',
            'model_type': 'standard',  # 'standard', 'fast', or 'typed'
            'hidden_channels': 64,
            'num_layers': 2,
            'num_bases': 30,
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 30,
        }
    },
    
    'HAN': {
        'train_func': train_han,
        'type': 'Heterogeneous',
        'description': 'Heterogeneous Attention Network',
        'config': {
            'checkpoint_dir': 'checkpoints/HAN',
            'model_type': 'simple',  # 'simple', 'han', or 'custom'
            'hidden_channels': 64,
            'num_heads': 8,
            'dropout': 0.6,
            'lr': 0.005,
            'epochs': 300,
            'patience': 40,
        }
    },
}


def get_model_config(model_name):
    """
    Get configuration cho specific model.
    
    Args:
        model_name: Tên model ('GNN', 'GCN', 'GAT', etc.)
    
    Returns:
        config: Dictionary chứa model configuration
    
    Raises:
        ValueError: Nếu model name không tồn tại
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_name]


def list_available_models():
    """
    List tất cả available models.
    
    Returns:
        list: List of model names
    """
    return list(MODEL_CONFIGS.keys())


def get_homogeneous_models():
    """Get list of homogeneous models."""
    return [name for name, config in MODEL_CONFIGS.items() if config['type'] == 'Homogeneous']


def get_heterogeneous_models():
    """Get list of heterogeneous models."""
    return [name for name, config in MODEL_CONFIGS.items() if config['type'] == 'Heterogeneous']
