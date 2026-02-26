"""
RGCN Module for Fraud Detection
"""

try:
    from .model import (
        RGCN,
        FastRGCN,
        TypedRGCN,
        prepare_rgcn_data,
        create_rgcn_model
    )
except ImportError:
    from RGCN.model import (
        RGCN,
        FastRGCN,
        TypedRGCN,
        prepare_rgcn_data,
        create_rgcn_model
    )

__all__ = [
    'RGCN',
    'FastRGCN',
    'TypedRGCN',
    'prepare_rgcn_data',
    'create_rgcn_model'
]
