\"\"\"
GAT Module for Fraud Detection
\"\"\"

try:
    from .model import GAT, GATv2, MultiHeadGAT, create_gat_model
except ImportError:
    from GAT.model import GAT, GATv2, MultiHeadGAT, create_gat_model

__all__ = ['GAT', 'GATv2', 'MultiHeadGAT', 'create_gat_model']
