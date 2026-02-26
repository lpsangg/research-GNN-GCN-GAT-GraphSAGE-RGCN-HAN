"""
Optimized Main Script - Chạy full dataset với memory optimization
"""

import os
import sys
import torch
import gc
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from models_config import MODEL_CONFIGS
from runner import train_single_model
from main import create_comparison_table, plot_comprehensive_comparison, save_results, print_final_summary
from utils import set_seed

import time
from datetime import datetime


def main_optimized():
    """
    Optimized main function - train models sequentially để tiết kiệm memory.
    """
    set_seed(42)
    
    # Optimized configuration
    base_config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'plot_dir': 'plots',
        'seed': 42,
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        
        # Training
        'print_every': 10,
        'plot_curves': True,
        
        # FULL DATASET - but with optimizations
        'max_samples': None,  # Use all data
    }
    
    print("\n" + "="*70)
    print("OPTIMIZED GNN TRAINING - FULL DATASET")
    print("="*70)
    print("\n⚡ Optimizations:")
    print("  - Sequential training (one model at a time)")
    print("  - Memory cleanup between models")
    print("  - Reduced V features (top 50 instead of 100)")
    print("  - Cached preprocessed data")
    
    print("\n📊 Configuration:")
    print(f"  - Dataset: Full (590,540 samples)")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Estimated time: 8-12 hours for all 6 models")
    print(f"  - RAM requirement: 16-32 GB")
    
    print("\n🎯 Available models:")
    for idx, (name, info) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"  {idx}. {name} - {info['description']}")
    
    # User options
    print("\n" + "="*70)
    print("OPTIONS:")
    print("  [1] Train all models sequentially (recommended)")
    print("  [2] Train only homogeneous models")
    print("  [3] Train only heterogeneous models")
    print("  [4] Select specific models")
    print("="*70)
    
    choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"
    
    # Determine which models to run
    models_to_run = []
    
    if choice == "1":
        models_to_run = list(MODEL_CONFIGS.keys())
        print(f"\n✓ Training all {len(models_to_run)} models sequentially...")
    elif choice == "2":
        models_to_run = ['GNN', 'GCN', 'GAT', 'GraphSAGE']
        print(f"\n✓ Training {len(models_to_run)} homogeneous models...")
    elif choice == "3":
        models_to_run = ['RGCN', 'HAN']
        print(f"\n✓ Training {len(models_to_run)} heterogeneous models...")
    elif choice == "4":
        print("\nAvailable:", ', '.join(MODEL_CONFIGS.keys()))
        selected = input("Enter model names (comma-separated): ").strip()
        models_to_run = [m.strip() for m in selected.split(',')]
        print(f"\n✓ Training {len(models_to_run)} selected models...")
    
    # Confirm
    print("\n" + "="*70)
    print("⚠️  WARNING: This will take a LONG time (hours)!")
    print("   Make sure:")
    print("   - You have enough RAM (16-32 GB)")
    print("   - Power saving is disabled")
    print("   - You can leave the computer running")
    print("="*70)
    
    response = input("\nProceed with full dataset training? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ Cancelled.")
        print("\n💡 TIP: To test first, set 'max_samples': 10000 in main.py")
        return
    
    # Train models sequentially
    print("\n" + "="*70)
    print("STARTING SEQUENTIAL TRAINING")
    print("="*70)
    
    results = {}
    total_start = time.time()
    
    for idx, model_name in enumerate(models_to_run, 1):
        print(f"\n{'='*70}")
        print(f"MODEL {idx}/{len(models_to_run)}: {model_name}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        try:
            # Train model
            result = train_single_model(model_name, base_config)
            results[model_name] = result
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Progress
            elapsed = (time.time() - total_start) / 60
            remaining = len(models_to_run) - idx
            print(f"\n✅ Progress: {idx}/{len(models_to_run)} models completed")
            print(f"   Elapsed time: {elapsed:.1f} minutes")
            if idx > 0:
                avg_time = elapsed / idx
                est_remaining = avg_time * remaining
                print(f"   Estimated remaining: {est_remaining:.1f} minutes ({est_remaining/60:.1f} hours)")
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
            
            # Ask if continue
            if idx < len(models_to_run):
                cont = input(f"\nContinue with remaining {remaining} models? (y/n): ")
                if cont.lower() != 'y':
                    print("❌ Training stopped by user.")
                    break
    
    total_time = time.time() - total_start
    
    # Results processing
    print("\n" + "="*70)
    print("PROCESSING RESULTS")
    print("="*70)
    
    df = create_comparison_table(results)
    
    # Only generate plots if we have results
    if len(df) > 0:
        print("\n📊 Generating comparison plots...")
        plot_comprehensive_comparison(results)
        
        print("\n💾 Saving results...")
        save_results(results, df)
        
        print_final_summary(results, df)
    else:
        print("\n⚠️  No models completed successfully!")
    
    # Final stats
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = len([r for r in results.values() if 'error' not in r])
    print(f"✅ Successfully trained: {successful}/{len(models_to_run)} models")
    
    if successful > 0:
        print("\n📁 Results saved to:")
        print("   - results/results_*.json")
        print("   - results/comparison_*.csv")
        print("   - plots/*.png")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main_optimized()
