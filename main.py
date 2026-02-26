"""
Main Script để chạy và so sánh tất cả 6 GNN models cho Fraud Detection
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)  # Change to script directory
sys.path.insert(0, str(SCRIPT_DIR))  # Add to Python path

# Import model registry and runner
from models_config import MODEL_CONFIGS, list_available_models
from runner import run_all_models, run_homogeneous_models, run_heterogeneous_models, run_selected_models

# Import utilities
from utils import (
    set_seed,
    plot_model_comparison
)


# Note: run_all_models is now imported from runner.py
# All model training logic has been moved to runner.py and models_config.py


def create_comparison_table(results):
    """
    Tạo comparison table từ results.
    
    Args:
        results: Dictionary chứa results của tất cả models
    
    Returns:
        df: Pandas DataFrame chứa comparison
    """
    data = []
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
        
        metrics = result['metrics']
        
        row = {
            'Model': model_name,
            'Type': result['type'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC-ROC': metrics.get('auc_roc', 0),
            'AUC-PR': metrics.get('auc_pr', 0),
            'Training Time (min)': result['training_time'] / 60,
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by AUC-ROC descending (if dataframe not empty)
    if len(df) > 0 and 'AUC-ROC' in df.columns:
        df = df.sort_values('AUC-ROC', ascending=False)
    
    return df


def plot_comprehensive_comparison(results, save_dir='plots'):
    """
    Tạo comprehensive comparison plots.
    
    Args:
        results: Dictionary chứa results của tất cả models
        save_dir: Directory để save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out failed models
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    # ============================================================
    # 1. Performance Comparison Bar Chart
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(valid_results.keys())
    
    # AUC-ROC
    auc_roc = [valid_results[m]['metrics'].get('auc_roc', 0) for m in models]
    axes[0, 0].bar(models, auc_roc, color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0, 0].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(auc_roc):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1-Score
    f1 = [valid_results[m]['metrics']['f1'] for m in models]
    axes[0, 1].bar(models, f1, color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Precision vs Recall
    precision = [valid_results[m]['metrics']['precision'] for m in models]
    recall = [valid_results[m]['metrics']['recall'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    axes[1, 0].bar(x - width/2, precision, width, label='Precision', color='lightgreen', alpha=0.8)
    axes[1, 0].bar(x + width/2, recall, width, label='Recall', color='lightcoral', alpha=0.8)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Training Time
    training_time = [valid_results[m]['training_time'] / 60 for m in models]
    axes[1, 1].bar(models, training_time, color='mediumpurple', alpha=0.8)
    axes[1, 1].set_ylabel('Time (minutes)', fontsize=12)
    axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(training_time):
        axes[1, 1].text(i, v + max(training_time)*0.02, f'{v:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {save_dir / 'model_comparison.png'}")
    plt.close()
    
    # ============================================================
    # 2. Training Curves Comparison
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot training curves for each model
    for model_name, result in valid_results.items():
        history = result['history']
        
        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label=f'{model_name} (Train)', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], label=f'{model_name} (Val)', alpha=0.7, linestyle='--')
        
        # Accuracy
        if 'train_acc' in history:
            axes[0, 1].plot(history['train_acc'], label=f'{model_name} (Train)', alpha=0.7)
            axes[0, 1].plot(history['val_acc'], label=f'{model_name} (Val)', alpha=0.7, linestyle='--')
        
        # F1
        if 'train_f1' in history:
            axes[1, 0].plot(history['train_f1'], label=f'{model_name} (Train)', alpha=0.7)
            axes[1, 0].plot(history['val_f1'], label=f'{model_name} (Val)', alpha=0.7, linestyle='--')
        
        # AUC
        if 'train_auc' in history:
            axes[1, 1].plot(history['train_auc'], label=f'{model_name} (Train)', alpha=0.7)
            axes[1, 1].plot(history['val_auc'], label=f'{model_name} (Val)', alpha=0.7, linestyle='--')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC-ROC')
    axes[1, 1].set_title('AUC-ROC Comparison', fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_dir / 'training_curves_comparison.png'}")
    plt.close()
    
    # ============================================================
    # 3. Radar Chart
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall', 'Accuracy']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model_name in enumerate(models):
        metrics = valid_results[model_name]['metrics']
        values = [
            metrics.get('auc_roc', 0),
            metrics['f1'],
            metrics['precision'],
            metrics['recall'],
            metrics['accuracy']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved radar chart to {save_dir / 'radar_chart.png'}")
    plt.close()


def save_results(results, df, save_dir='results'):
    """
    Save results to JSON và CSV files.
    
    Args:
        results: Dictionary chứa results
        df: DataFrame chứa comparison table
        save_dir: Directory để save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results to JSON
    results_to_save = {}
    for model_name, result in results.items():
        if 'error' not in result:
            # Remove history (too large) và model object
            results_to_save[model_name] = {
                'model': result['model'],
                'type': result['type'],
                'metrics': result['metrics'],
                'training_time': result['training_time'],
            }
        else:
            results_to_save[model_name] = result
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = save_dir / f'results_{timestamp}.json'
    
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✓ Saved results to {json_path}")
    
    # Save comparison table to CSV
    csv_path = save_dir / f'comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table to {csv_path}")
    
    # Save formatted comparison table to text
    txt_path = save_dir / f'comparison_{timestamp}.txt'
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("GNN MODEL COMPARISON FOR FRAUD DETECTION\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*100 + "\n")
    
    print(f"✓ Saved formatted table to {txt_path}")


def print_final_summary(results, df):
    """
    Print final summary của experiment.
    
    Args:
        results: Dictionary chứa results
        df: DataFrame chứa comparison table
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Overall statistics
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    print(f"\nModels trained successfully: {len(valid_results)}/6")
    
    if len(valid_results) == 0:
        print("\n⚠️  No models completed successfully!")
        print("\nPossible reasons:")
        print("  1. Dataset not found - Did you download the IEEE-CIS dataset?")
        print("  2. Missing dependencies - Check requirements.txt")
        print("  3. CUDA/GPU issues - Try running on CPU")
        print("\nTo download dataset:")
        print("  kaggle competitions download -c ieee-fraud-detection")
        print("  unzip ieee-fraud-detection.zip -d dataset/ieee-fraud-detection/")
        return
    
    # Failed models
    failed = [k for k, v in results.items() if 'error' in v]
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        print("\nErrors:")
        for model_name in failed:
            print(f"  - {model_name}: {results[model_name]['error']}")
    
    print("\n" + "-"*70)
    print("COMPARISON TABLE")
    print("-"*70)
    print(df.to_string(index=False))
    
    # Best model (only if df not empty)
    if len(df) > 0:
        print("\n" + "-"*70)
        print("BEST MODEL")
        print("-"*70)
        
        best_model = df.iloc[0]
        print(f"Model: {best_model['Model']}")
        print(f"Type: {best_model['Type']}")
        print(f"AUC-ROC: {best_model['AUC-ROC']:.4f}")
        print(f"F1-Score: {best_model['F1-Score']:.4f}")
        print(f"Precision: {best_model['Precision']:.4f}")
        print(f"Recall: {best_model['Recall']:.4f}")
        print(f"Training Time: {best_model['Training Time (min)']:.2f} minutes")
    
    # Comparison insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)
    
    # Homogeneous vs Heterogeneous
    homo_models = df[df['Type'] == 'Homogeneous']
    hetero_models = df[df['Type'] == 'Heterogeneous']
    
    if len(homo_models) > 0 and len(hetero_models) > 0:
        avg_homo_auc = homo_models['AUC-ROC'].mean()
        avg_hetero_auc = hetero_models['AUC-ROC'].mean()
        
        print(f"\n1. Graph Type Performance:")
        print(f"   - Homogeneous models avg AUC-ROC: {avg_homo_auc:.4f}")
        print(f"   - Heterogeneous models avg AUC-ROC: {avg_hetero_auc:.4f}")
        
        if avg_hetero_auc > avg_homo_auc:
            diff = ((avg_hetero_auc - avg_homo_auc) / avg_homo_auc) * 100
            print(f"   → Heterogeneous models perform {diff:.1f}% better!")
        else:
            diff = ((avg_homo_auc - avg_hetero_auc) / avg_hetero_auc) * 100
            print(f"   → Homogeneous models perform {diff:.1f}% better!")
    
    # Training time (only if df not empty)
    if len(df) > 0:
        fastest = df.iloc[-1]
        slowest = df.iloc[0]
        print(f"\n2. Training Efficiency:")
        print(f"   - Fastest: {fastest['Model']} ({fastest['Training Time (min)']:.2f} min)")
        print(f"   - Slowest: {slowest['Model']} ({slowest['Training Time (min)']:.2f} min)")
    
    # Performance range (only if df not empty)
    if len(df) > 0:
        print(f"\n3. Performance Range:")
        print(f"   - AUC-ROC: {df['AUC-ROC'].min():.4f} - {df['AUC-ROC'].max():.4f}")
        print(f"   - F1-Score: {df['F1-Score'].min():.4f} - {df['F1-Score'].max():.4f}")
        print(f"   - Spread: {(df['AUC-ROC'].max() - df['AUC-ROC'].min()):.4f}")
    
    # Recommendation (only if df not empty)
    if len(df) > 0:
        print("\n" + "-"*70)
        print("RECOMMENDATION")
        print("-"*70)
        
        best_auc = df.iloc[0]['AUC-ROC']
        best_model_name = df.iloc[0]['Model']
        
        if best_auc >= 0.90:
            print(f"✓ Excellent performance! {best_model_name} achieves AUC-ROC >= 0.90")
        elif best_auc >= 0.85:
            print(f"✓ Good performance! {best_model_name} achieves AUC-ROC >= 0.85")
        else:
            print(f"⚠ Moderate performance. Consider hyperparameter tuning or more data.")
        
        print(f"\nFor production fraud detection, we recommend:")
        print(f"  → {best_model_name} (Best AUC-ROC: {best_auc:.4f})")
        
        # Second best
        if len(df) > 1:
            second_best = df.iloc[1]
            print(f"  → Alternative: {second_best['Model']} (AUC-ROC: {second_best['AUC-ROC']:.4f})")
    
    print("\n" + "="*70)


def main():
    """
    Main function để chạy toàn bộ experiment.
    """
    # Set random seed
    set_seed(42)
    
    # Base configuration
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
        
        # For quick testing, set max_samples to small number (e.g., 5000)
        # For full training, set to None
        'max_samples': 10000,  # Change to 5000 for quick testing, None for full training
    }
    
    print("\n" + "="*70)
    print("COMPREHENSIVE GNN RESEARCH FOR FRAUD DETECTION")
    print("="*70)
    print("\nAvailable models:")
    for idx, (name, info) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"  {idx}. {name} - {info['description']} ({info['type']})")
    
    print("\nConfiguration:")
    print(f"  - Max samples: {base_config['max_samples'] or 'All'}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Seed: {base_config['seed']}")
    
    # User options
    print("\n" + "="*70)
    print("OPTIONS:")
    print("  [1] Train all models (default)")
    print("  [2] Train only homogeneous models (GNN, GCN, GAT, GraphSAGE)")
    print("  [3] Train only heterogeneous models (RGCN, HAN)")
    print("  [4] Select specific models")
    print("="*70)
    
    choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"
    
    # Determine which models to run
    if choice == "1":
        print("\n✓ Training all models...")
        run_func = run_all_models
    elif choice == "2":
        print("\n✓ Training homogeneous models only...")
        run_func = run_homogeneous_models
    elif choice == "3":
        print("\n✓ Training heterogeneous models only...")
        run_func = run_heterogeneous_models
    elif choice == "4":
        print("\nAvailable models:", ', '.join(MODEL_CONFIGS.keys()))
        selected = input("Enter model names (comma-separated): ").strip()
        selected_models = [m.strip() for m in selected.split(',')]
        print(f"\n✓ Training selected models: {', '.join(selected_models)}")
        
        def run_func(config):
            return run_selected_models(config, selected_models)
    else:
        print(f"Invalid choice: {choice}. Using default (all models).")
        run_func = run_all_models
    
    # Confirm
    print("\n" + "="*70)
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run models
    start_time = time.time()
    results = run_func(base_config)
    total_time = time.time() - start_time
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Plot comparisons
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    plot_comprehensive_comparison(results)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    save_results(results, df)
    
    # Print final summary
    print_final_summary(results, df)
    
    print(f"\n✓ Total experiment time: {total_time/60:.2f} minutes")
    print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/results_*.json")
    print("  - results/comparison_*.csv")
    print("  - results/comparison_*.txt")
    print("\nPlots saved to:")
    print("  - plots/model_comparison.png")
    print("  - plots/training_curves_comparison.png")
    print("  - plots/radar_chart.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
