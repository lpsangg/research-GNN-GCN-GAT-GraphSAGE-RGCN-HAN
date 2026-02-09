"""
Evaluation Metrics for Fraud Detection
Chung cho tất cả 6 models
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Tính toán tất cả metrics quan trọng cho fraud detection.
    
    Args:
        y_true: Ground truth labels (numpy array hoặc torch tensor)
        y_pred: Predicted labels (numpy array hoặc torch tensor)
        y_prob: Predicted probabilities (numpy array hoặc torch tensor), optional
    
    Returns:
        metrics_dict: Dictionary chứa tất cả metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC metrics (requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Khi chỉ có 1 class trong y_true
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return metrics


def print_metrics(metrics, prefix=''):
    """
    In ra metrics một cách đẹp mắt.
    
    Args:
        metrics: Dictionary metrics
        prefix: Prefix cho output (vd: 'Train', 'Val', 'Test')
    """
    print(f"\n{'='*50}")
    print(f"{prefix} Metrics:")
    print(f"{'='*50}")
    
    # Main metrics
    if 'accuracy' in metrics:
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
    if 'precision' in metrics:
        print(f"Precision:  {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"Recall:     {metrics['recall']:.4f}")
    if 'f1' in metrics:
        print(f"F1-Score:   {metrics['f1']:.4f}")
    
    # AUC metrics
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    if 'auc_pr' in metrics:
        print(f"AUC-PR:     {metrics['auc_pr']:.4f}")
    
    # Confusion Matrix
    if 'true_positive' in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negative']:<6} FP: {metrics['false_positive']:<6}")
        print(f"  FN: {metrics['false_negative']:<6} TP: {metrics['true_positive']:<6}")
    
    # Additional metrics
    if 'specificity' in metrics:
        print(f"\nSpecificity: {metrics['specificity']:.4f}")
    if 'fpr' in metrics:
        print(f"FPR:         {metrics['fpr']:.4f}")
    if 'fnr' in metrics:
        print(f"FNR:         {metrics['fnr']:.4f}")
    
    print(f"{'='*50}\n")


def get_classification_report(y_true, y_pred):
    """
    Lấy classification report chi tiết.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
    
    Returns:
        report: String classification report
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    report = classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Fraud'],
                                   digits=4)
    return report


def compute_roc_curve(y_true, y_prob):
    """
    Tính ROC curve.
    
    Args:
        y_true: Ground truth
        y_prob: Predicted probabilities
    
    Returns:
        fpr, tpr, thresholds
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def compute_pr_curve(y_true, y_prob):
    """
    Tính Precision-Recall curve.
    
    Args:
        y_true: Ground truth
        y_prob: Predicted probabilities
    
    Returns:
        precision, recall, thresholds
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def find_best_threshold(y_true, y_prob, metric='f1'):
    """
    Tìm threshold tốt nhất để maximize metric.
    
    Args:
        y_true: Ground truth
        y_prob: Predicted probabilities
        metric: 'f1', 'precision', 'recall', hoặc 'balanced'
    
    Returns:
        best_threshold: Threshold tối ưu
        best_score: Score tương ứng
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            # Balance giữa precision và recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec + 1e-10)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


class MetricsTracker:
    """
    Class để track metrics qua các epochs.
    """
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
        }
    
    def update(self, epoch_metrics):
        """
        Update metrics cho 1 epoch.
        
        Args:
            epoch_metrics: Dict chứa metrics của epoch
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric='val_auc'):
        """
        Lấy epoch tốt nhất theo metric.
        
        Args:
            metric: Metric để đánh giá
        
        Returns:
            best_epoch: Epoch tốt nhất (0-indexed)
            best_value: Giá trị tốt nhất
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0, 0.0
        
        values = self.history[metric]
        
        # Với loss thì càng thấp càng tốt
        if 'loss' in metric:
            best_epoch = np.argmin(values)
        else:
            best_epoch = np.argmax(values)
        
        best_value = values[best_epoch]
        
        return best_epoch, best_value
    
    def get_history(self):
        """
        Lấy toàn bộ history.
        """
        return self.history
