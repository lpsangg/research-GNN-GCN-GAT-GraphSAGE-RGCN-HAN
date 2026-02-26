"""
Example: Advanced usage of Model Registry
Demonstrates flexible model training patterns
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models_config import MODEL_CONFIGS, list_available_models, get_homogeneous_models
from runner import run_all_models, run_selected_models, train_single_model
from utils import set_seed


def example_1_train_single_model():
    """Example 1: Train a single specific model"""
    print("="*70)
    print("EXAMPLE 1: Train Single Model")
    print("="*70)
    
    set_seed(42)
    
    base_config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'max_samples': 5000,  # Quick test
        'seed': 42,
    }
    
    # Train just GNN
    result = train_single_model('GNN', base_config)
    
    if 'error' not in result:
        print(f"\n✓ Success!")
        print(f"  AUC-ROC: {result['metrics']['auc_roc']:.4f}")
        print(f"  F1: {result['metrics']['f1']:.4f}")
    else:
        print(f"\n✗ Failed: {result['error']}")


def example_2_compare_attention_models():
    """Example 2: Compare attention-based models (GAT vs HAN)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Compare Attention Models")
    print("="*70)
    
    set_seed(42)
    
    base_config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'max_samples': 5000,
        'seed': 42,
    }
    
    # Train only GAT and HAN
    results = run_selected_models(base_config, ['GAT', 'HAN'])
    
    print("\n" + "="*70)
    print("COMPARISON: GAT vs HAN")
    print("="*70)
    
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Training time: {result['training_time']/60:.2f} min")


def example_3_quick_test_all():
    """Example 3: Quick test all models with small sample"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Quick Test All Models")
    print("="*70)
    
    set_seed(42)
    
    # Quick config for testing
    base_config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'max_samples': 2000,  # Very small sample for speed
        'seed': 42,
        'print_every': 5,
    }
    
    # Override epochs for quick testing
    for model_config in MODEL_CONFIGS.values():
        model_config['config']['epochs'] = 20  # Reduce epochs
        model_config['config']['patience'] = 5
    
    results = run_all_models(base_config)
    
    print("\n" + "="*70)
    print("QUICK TEST RESULTS")
    print("="*70)
    
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name}: AUC-ROC={result['metrics']['auc_roc']:.4f}")
        else:
            print(f"{model_name}: FAILED")


def example_4_custom_pipeline():
    """Example 4: Custom training pipeline with preprocessing"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Pipeline with New Preprocessing")
    print("="*70)
    
    from utils.data_loader import load_ieee_fraud_data, PreprocessingPipeline
    from utils.graph_construction import prepare_homogeneous_data
    
    set_seed(42)
    
    # Load data
    train_trans, train_ident, _, _ = load_ieee_fraud_data()
    
    # Use new preprocessing pipeline
    pipeline = PreprocessingPipeline()
    df, features = pipeline.fit_transform(
        train_trans.head(2000), 
        train_ident[train_ident['TransactionID'].isin(train_trans.head(2000)['TransactionID'])]
    )
    
    print(f"✓ Processed data: {df.shape}")
    print(f"✓ Features: {len(features)}")
    print(f"✓ Feature groups:")
    print(f"  - M features: {len([f for f in features if f.startswith('M')])}")
    print(f"  - V features: {len([f for f in features if f.startswith('V')])}")
    
    # Build graph
    data = prepare_homogeneous_data(df, features)
    print(f"✓ Graph built: {data.num_nodes} nodes, {data.num_edges} edges")


def example_5_programmatic_config():
    """Example 5: Programmatic model configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Programmatic Configuration")
    print("="*70)
    
    # List available models
    print("Available models:")
    for model in list_available_models():
        config = MODEL_CONFIGS[model]
        print(f"  - {model} ({config['type']}): {config['description']}")
    
    # Get homogeneous models
    homo = get_homogeneous_models()
    print(f"\nHomogeneous models: {', '.join(homo)}")
    
    # Inspect specific model config
    gat_config = MODEL_CONFIGS['GAT']
    print(f"\nGAT Configuration:")
    for key, value in gat_config['config'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL REGISTRY USAGE EXAMPLES")
    print("="*70)
    print("\nSelect example to run:")
    print("  [1] Train single model (GNN)")
    print("  [2] Compare attention models (GAT vs HAN)")
    print("  [3] Quick test all models (small sample)")
    print("  [4] Custom pipeline with new preprocessing")
    print("  [5] Programmatic configuration (info only)")
    print("="*70)
    
    choice = input("\nSelect example (1-5): ").strip()
    
    examples = {
        '1': example_1_train_single_model,
        '2': example_2_compare_attention_models,
        '3': example_3_quick_test_all,
        '4': example_4_custom_pipeline,
        '5': example_5_programmatic_config,
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print(f"Invalid choice: {choice}")
        print("Running Example 5 (info only)...")
        example_5_programmatic_config()
