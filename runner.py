"""
Model Training Runner
Handles training of multiple models with unified interface
"""

import time
from models_config import MODEL_CONFIGS


def train_single_model(model_name, base_config):
    """
    Train a single model.
    
    Args:
        model_name: Name of the model to train
        base_config: Base configuration dictionary
    
    Returns:
        result: Dictionary containing training results
    """
    if model_name not in MODEL_CONFIGS:
        return {'error': f'Unknown model: {model_name}'}
    
    model_info = MODEL_CONFIGS[model_name]
    
    # Merge base config with model-specific config
    config = base_config.copy()
    config.update(model_info['config'])
    
    # Train model
    train_func = model_info['train_func']
    
    try:
        start_time = time.time()
        model, test_metrics, tracker = train_func(config)
        training_time = time.time() - start_time
        
        result = {
            'model': model_name,
            'type': model_info['type'],
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ {model_name} completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ {model_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def run_all_models(base_config, model_subset=None):
    """
    Chạy tất cả models hoặc subset of models.
    
    Args:
        base_config: Dictionary chứa common configuration
        model_subset: List of model names to train (None = all models)
    
    Returns:
        results: Dictionary chứa results của tất cả models
    """
    results = {}
    
    # Determine which models to run
    models_to_run = model_subset if model_subset else list(MODEL_CONFIGS.keys())
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE GNN RESEARCH FOR FRAUD DETECTION")
    print("="*70)
    print(f"Models to train: {', '.join(models_to_run)}")
    print(f"Total: {len(models_to_run)} models")
    print("="*70)
    
    # Train each model
    for idx, model_name in enumerate(models_to_run, 1):
        model_info = MODEL_CONFIGS[model_name]
        
        print("\n" + "="*70)
        print(f"MODEL {idx}/{len(models_to_run)}: {model_name} ({model_info['description']})")
        print(f"Type: {model_info['type']}")
        print("="*70)
        
        results[model_name] = train_single_model(model_name, base_config)
    
    return results


def run_homogeneous_models(base_config):
    """Chỉ chạy homogeneous models (GNN, GCN, GAT, GraphSAGE)."""
    homo_models = [name for name, info in MODEL_CONFIGS.items() if info['type'] == 'Homogeneous']
    return run_all_models(base_config, model_subset=homo_models)


def run_heterogeneous_models(base_config):
    """Chỉ chạy heterogeneous models (RGCN, HAN)."""
    hetero_models = [name for name, info in MODEL_CONFIGS.items() if info['type'] == 'Heterogeneous']
    return run_all_models(base_config, model_subset=hetero_models)


def run_selected_models(base_config, model_names):
    """
    Chạy specific models.
    
    Args:
        base_config: Base configuration
        model_names: List of model names to run
    
    Returns:
        results: Dictionary containing results
    """
    # Validate model names
    invalid_models = [m for m in model_names if m not in MODEL_CONFIGS]
    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Available: {list(MODEL_CONFIGS.keys())}")
    
    return run_all_models(base_config, model_subset=model_names)
