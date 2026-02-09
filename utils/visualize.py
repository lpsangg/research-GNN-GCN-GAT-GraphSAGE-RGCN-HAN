"""
Visualization Utilities
Chung cho tất cả 6 models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(history, save_path=None):
    """
    Plot training và validation curves (loss, accuracy, etc.).
    
    Args:
        history: Dictionary chứa training history
        save_path: Path để save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy curves
    if 'train_acc' in history and len(history['train_acc']) > 0:
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        if 'val_acc' in history and len(history['val_acc']) > 0:
            axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # F1 curves
    if 'train_f1' in history and len(history['train_f1']) > 0:
        axes[1, 0].plot(history['train_f1'], label='Train F1', marker='o')
        if 'val_f1' in history and len(history['val_f1']) > 0:
            axes[1, 0].plot(history['val_f1'], label='Val F1', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # AUC curves
    if 'train_auc' in history and len(history['train_auc']) > 0:
        axes[1, 1].plot(history['train_auc'], label='Train AUC', marker='o')
        if 'val_auc' in history and len(history['val_auc']) > 0:
            axes[1, 1].plot(history['val_auc'], label='Val AUC', marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].set_title('AUC-ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        save_path: Path để save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                cbar=True)
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm_normalized[i,j]*100:.1f}%)',
                    ha='center', va='center', color='gray', fontsize=10)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC-ROC score
        save_path: Path để save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(precision, recall, auc_pr, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        auc_pr: AUC-PR score
        save_path: Path để save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {auc_pr:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, metric='auc_roc', save_path=None):
    """
    So sánh performance của các models.
    
    Args:
        results_dict: Dictionary {model_name: metrics_dict}
        metric: Metric để so sánh
        save_path: Path để save figure (optional)
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison saved to {save_path}")
    
    plt.show()


def plot_multiple_metrics_comparison(results_dict, metrics=['accuracy', 'f1', 'auc_roc'], save_path=None):
    """
    So sánh nhiều metrics của các models.
    
    Args:
        results_dict: Dictionary {model_name: metrics_dict}
        metrics: List các metrics để so sánh
        save_path: Path để save figure (optional)
    """
    models = list(results_dict.keys())
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, metric in enumerate(metrics):
        scores = [results_dict[model].get(metric, 0) for model in models]
        offset = width * (i - len(metrics)/2 + 0.5)
        bars = ax.bar(x + offset, scores, width, label=metric.upper(), 
                     color=colors[i % len(colors)], alpha=0.7)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Metric Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Multi-metric comparison saved to {save_path}")
    
    plt.show()


def plot_learning_rate_schedule(lr_history, save_path=None):
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List learning rates qua các epochs
        save_path: Path để save figure (optional)
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(lr_history, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ LR schedule saved to {save_path}")
    
    plt.show()
