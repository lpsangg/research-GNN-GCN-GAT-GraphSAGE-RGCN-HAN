"""
GraphSAGE Module for Fraud Detection
"""

try:
    from .model import (
        GraphSAGE, 
        DeepGraphSAGE, 
        AttentionGraphSAGE, 
        MultiAggregatorSAGE,
        create_graphsage_model
    )
except ImportError:
    from GraphSAGE.model import (
        GraphSAGE, 
        DeepGraphSAGE, 
        AttentionGraphSAGE, 
        MultiAggregatorSAGE,
        create_graphsage_model
    )

__all__ = [
    'GraphSAGE', 
    'DeepGraphSAGE', 
    'AttentionGraphSAGE', 
    'MultiAggregatorSAGE',
    'create_graphsage_model'
]
