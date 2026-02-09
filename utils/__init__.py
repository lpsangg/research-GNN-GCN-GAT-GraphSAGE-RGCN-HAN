"""
Utilities package for GNN Fraud Detection Research
"""

from .graph_construction import (
    create_user_id_refined,
    prepare_homogeneous_data,
    prepare_hetero_data,
    prepare_hetero_data_with_features,
    create_train_val_test_split,
    aggregate_features_for_heterogeneous
)

from .data_loader import (
    load_ieee_fraud_data,
    preprocess_features,
    get_feature_statistics,
    balance_dataset
)

from .metrics import (
    compute_metrics,
    print_metrics,
    get_classification_report,
    compute_roc_curve,
    compute_pr_curve,
    find_best_threshold,
    MetricsTracker
)

from .trainer import (
    EarlyStopping,
    LRSchedulerWrapper,
    ModelCheckpoint,
    GradientClipper,
    ExperimentLogger,
    count_parameters,
    set_seed
)

from .visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison,
    plot_multiple_metrics_comparison,
    plot_learning_rate_schedule
)

__all__ = [
    # Graph Construction
    'create_user_id_refined',
    'prepare_homogeneous_data',
    'prepare_hetero_data',
    'prepare_hetero_data_with_features',
    'create_train_val_test_split',
    'aggregate_features_for_heterogeneous',
    
    # Data Loader
    'load_ieee_fraud_data',
    'preprocess_features',
    'get_feature_statistics',
    'balance_dataset',
    
    # Metrics
    'compute_metrics',
    'print_metrics',
    'get_classification_report',
    'compute_roc_curve',
    'compute_pr_curve',
    'find_best_threshold',
    'MetricsTracker',
    
    # Trainer
    'EarlyStopping',
    'LRSchedulerWrapper',
    'ModelCheckpoint',
    'GradientClipper',
    'ExperimentLogger',
    'count_parameters',
    'set_seed',
    
    # Visualize
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_model_comparison',
    'plot_multiple_metrics_comparison',
    'plot_learning_rate_schedule',
]
