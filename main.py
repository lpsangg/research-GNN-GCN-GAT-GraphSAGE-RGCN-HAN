"""
Main Script để chạy và so sánh tất cả 6 GNN models cho Fraud Detection
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import training functions
from GNN.train import train_gnn
from GCN.train import train_gcn
from GAT.train import train_gat
from GraphSAGE.train import train_graphsage
from RGCN.train import train_rgcn
from HAN.train import train_han

# Import utilities
from utils import (
    set_seed,
    plot_model_comparison
)


def create_experiment_config(base_config, model_specific=None):
    """
    Tạo config cho mỗi model với base config và model-specific settings.
    
    Args:
        base_config: Dict chứa common settings
        model_specific: Dict chứa model-specific settings
    
    Returns:
        config: Complete config dictionary
    """
    config = base_config.copy()
    if model_specific:
        config.update(model_specific)
    return config


def run_all_models(base_config):
    """
    Chạy tất cả 6 models và thu thập results.
    
    Args:
        base_config: Dictionary chứa common configuration
    
    Returns:
        results: Dictionary chứa results của tất cả models
    """
    results = {}
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE GNN RESEARCH FOR FRAUD DETECTION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Max samples: {base_config.get('max_samples', 'All')}")
    print("="*70)
    
    # ============================================================
    # 1. GNN (Basic Graph Neural Network)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 1/6: GNN (Basic Graph Neural Network)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        gnn_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/GNN',
            'model_type': 'improved',  # 'basic' or 'improved'
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.001,
            'epochs': 200,
            'patience': 20,
        })
        
        model, test_metrics, tracker = train_gnn(gnn_config)
        
        training_time = time.time() - start_time
        
        results['GNN'] = {
            'model': 'GNN',
            'type': 'Homogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ GNN completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ GNN failed: {str(e)}")
        results['GNN'] = {'error': str(e)}
    
    # ============================================================
    # 2. GCN (Graph Convolutional Network)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 2/6: GCN (Graph Convolutional Network)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        gcn_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/GCN',
            'model_type': 'jknet',  # 'standard', 'deep', or 'jknet'
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 20,
            'use_cached': True,
        })
        
        model, test_metrics, tracker = train_gcn(gcn_config)
        
        training_time = time.time() - start_time
        
        results['GCN'] = {
            'model': 'GCN',
            'type': 'Homogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ GCN completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ GCN failed: {str(e)}")
        results['GCN'] = {'error': str(e)}
    
    # ============================================================
    # 3. GAT (Graph Attention Network)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 3/6: GAT (Graph Attention Network)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        gat_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/GAT',
            'model_type': 'gatv2',  # 'standard', 'gatv2', or 'multihead'
            'hidden_channels': 64,
            'num_layers': 2,
            'heads': 8,
            'dropout': 0.6,
            'lr': 0.005,
            'epochs': 200,
            'patience': 30,
        })
        
        model, test_metrics, tracker = train_gat(gat_config)
        
        training_time = time.time() - start_time
        
        results['GAT'] = {
            'model': 'GAT',
            'type': 'Homogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ GAT completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ GAT failed: {str(e)}")
        results['GAT'] = {'error': str(e)}
    
    # ============================================================
    # 4. GraphSAGE (Sampling-based)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 4/6: GraphSAGE (Sampling-based Inductive Learning)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        sage_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/GraphSAGE',
            'model_type': 'attention',  # 'standard', 'deep', 'attention', or 'multi'
            'hidden_channels': 64,
            'num_layers': 2,
            'aggregator': 'mean',  # 'mean', 'max', 'min', 'lstm'
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 20,
        })
        
        model, test_metrics, tracker = train_graphsage(sage_config)
        
        training_time = time.time() - start_time
        
        results['GraphSAGE'] = {
            'model': 'GraphSAGE',
            'type': 'Homogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ GraphSAGE completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ GraphSAGE failed: {str(e)}")
        results['GraphSAGE'] = {'error': str(e)}
    
    # ============================================================
    # 5. RGCN (Relational GCN)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 5/6: RGCN (Relational Graph Convolutional Network)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        rgcn_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/RGCN',
            'model_type': 'standard',  # 'standard', 'fast', or 'typed'
            'hidden_channels': 64,
            'num_layers': 2,
            'num_bases': 30,
            'dropout': 0.5,
            'lr': 0.01,
            'epochs': 200,
            'patience': 30,
        })
        
        model, test_metrics, tracker = train_rgcn(rgcn_config)
        
        training_time = time.time() - start_time
        
        results['RGCN'] = {
            'model': 'RGCN',
            'type': 'Heterogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ RGCN completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ RGCN failed: {str(e)}")
        results['RGCN'] = {'error': str(e)}
    
    # ============================================================
    # 6. HAN (Heterogeneous Attention Network)
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 6/6: HAN (Heterogeneous Attention Network)")
    print("="*70)
    
    try:
        start_time = time.time()
        
        han_config = create_experiment_config(base_config, {
            'checkpoint_dir': 'checkpoints/HAN',
            'model_type': 'simple',  # 'simple', 'han', or 'custom'
            'hidden_channels': 64,
            'num_heads': 8,
            'dropout': 0.6,
            'lr': 0.005,
            'epochs': 300,
            'patience': 40,
        })
        
        model, test_metrics, tracker = train_han(han_config)
        
        training_time = time.time() - start_time
        
        results['HAN'] = {
            'model': 'HAN',
            'type': 'Heterogeneous',
            'metrics': test_metrics,
            'training_time': training_time,
            'history': tracker.get_history(),
        }
        
        print(f"\n✓ HAN completed in {training_time/60:.2f} minutes")
        print(f"  Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"\n✗ HAN failed: {str(e)}")
        results['HAN'] = {'error': str(e)}
    
    return results


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
    
    # Sort by AUC-ROC descending
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
        print("No models completed successfully")
        return
    
    # Failed models
    failed = [k for k, v in results.items() if 'error' in v]
    if failed:
        print(f"Failed models: {', '.join(failed)}")
    
    print("\n" + "-"*70)
    print("COMPARISON TABLE")
    print("-"*70)
    print(df.to_string(index=False))
    
    # Best model
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
    
    # Training time
    fastest = df.iloc[-1]
    slowest = df.iloc[0]
    print(f"\n2. Training Efficiency:")
    print(f"   - Fastest: {fastest['Model']} ({fastest['Training Time (min)']:.2f} min)")
    print(f"   - Slowest: {slowest['Model']} ({slowest['Training Time (min)']:.2f} min)")
    
    # Performance range
    print(f"\n3. Performance Range:")
    print(f"   - AUC-ROC: {df['AUC-ROC'].min():.4f} - {df['AUC-ROC'].max():.4f}")
    print(f"   - F1-Score: {df['F1-Score'].min():.4f} - {df['F1-Score'].max():.4f}")
    print(f"   - Spread: {(df['AUC-ROC'].max() - df['AUC-ROC'].min()):.4f}")
    
    # Recommendation
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
        'max_samples': None,  # Change to 5000 for quick testing
    }
    
    print("\n" + "="*70)
    print("COMPREHENSIVE GNN RESEARCH FOR FRAUD DETECTION")
    print("="*70)
    print("\nThis script will train and compare 6 different GNN models:")
    print("  1. GNN - Basic Graph Neural Network")
    print("  2. GCN - Graph Convolutional Network")
    print("  3. GAT - Graph Attention Network")
    print("  4. GraphSAGE - Sampling-based Inductive Learning")
    print("  5. RGCN - Relational Graph Convolutional Network")
    print("  6. HAN - Heterogeneous Attention Network")
    print("\nConfiguration:")
    print(f"  - Max samples: {base_config['max_samples'] or 'All'}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Seed: {base_config['seed']}")
    
    # Confirm
    print("\n" + "="*70)
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run all models
    start_time = time.time()
    results = run_all_models(base_config)
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
