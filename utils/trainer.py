"""
Training Utilities
Chung cho tất cả 6 models
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class EarlyStopping:
    """
    Early Stopping để tránh overfitting.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Số epochs chờ trước khi stop
            min_delta: Minimum change để coi là improvement
            mode: 'max' hoặc 'min' (maximize hoặc minimize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        """
        Check xem có nên stop không.
        
        Args:
            score: Current metric score
            epoch: Current epoch
        
        Returns:
            improved: True nếu có improvement
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        # Check improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def should_stop(self):
        """
        Check xem có nên stop training không.
        """
        return self.early_stop


class LRSchedulerWrapper:
    """
    Wrapper cho Learning Rate Scheduler.
    """
    def __init__(self, optimizer, scheduler_type='step', **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: 'step', 'cosine', 'plateau', 'exponential'
            **kwargs: Arguments cho scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        
        elif scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 100)
            eta_min = kwargs.get('eta_min', 0.0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        
        elif scheduler_type == 'plateau':
            mode = kwargs.get('mode', 'max')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        
        elif scheduler_type == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric=None):
        """
        Step scheduler.
        
        Args:
            metric: Metric value (required for ReduceLROnPlateau)
        """
        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("Metric required for ReduceLROnPlateau")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """
        Lấy learning rate hiện tại.
        """
        return self.scheduler.get_last_lr()[0]


class ModelCheckpoint:
    """
    Save model checkpoints.
    """
    def __init__(self, save_dir, model_name, mode='max'):
        """
        Args:
            save_dir: Directory để save checkpoints
            model_name: Tên model
            mode: 'max' hoặc 'min'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.best_score = None
    
    def save(self, model, optimizer, epoch, score, metrics=None):
        """
        Save checkpoint nếu score tốt hơn.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            score: Current score
            metrics: Dictionary metrics (optional)
        
        Returns:
            saved: True nếu đã save
        """
        # Check if this is best score
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == 'max' and score > self.best_score:
            is_best = True
        elif self.mode == 'min' and score < self.best_score:
            is_best = True
        
        if is_best:
            self.best_score = score
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'metrics': metrics
            }
            
            checkpoint_path = self.save_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"✓ Model saved at epoch {epoch} with score {score:.4f}")
            return True
        
        return False
    
    def load(self, model, optimizer=None):
        """
        Load checkpoint tốt nhất.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
        
        Returns:
            checkpoint: Dictionary chứa checkpoint info
        """
        checkpoint_path = self.save_dir / f'{self.model_name}_best.pth'
        
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Model loaded from epoch {checkpoint['epoch']} "
              f"with score {checkpoint['score']:.4f}")
        
        return checkpoint


class GradientClipper:
    """
    Gradient Clipping để tránh exploding gradients.
    """
    def __init__(self, max_norm=1.0, norm_type=2.0):
        """
        Args:
            max_norm: Maximum norm
            norm_type: Type of norm (2.0 for L2)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, model):
        """
        Clip gradients của model.
        
        Args:
            model: PyTorch model
        
        Returns:
            total_norm: Total norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            self.norm_type
        )


class ExperimentLogger:
    """
    Log experiments và hyperparameters.
    """
    def __init__(self, log_dir, experiment_name):
        """
        Args:
            log_dir: Directory để save logs
            experiment_name: Tên experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f'{experiment_name}_log.json'
        
        # Initialize log
        self.log_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': {},
            'epochs': []
        }
    
    def log_hyperparameters(self, hyperparams):
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary hyperparameters
        """
        self.log_data['hyperparameters'] = hyperparams
        self._save()
    
    def log_epoch(self, epoch, metrics):
        """
        Log metrics của 1 epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary metrics
        """
        epoch_data = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.log_data['epochs'].append(epoch_data)
        self._save()
    
    def log_final_results(self, results):
        """
        Log kết quả cuối cùng.
        
        Args:
            results: Dictionary final results
        """
        self.log_data['final_results'] = results
        self.log_data['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save()
    
    def _save(self):
        """
        Save log to file.
        """
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
    
    def load(self):
        """
        Load log từ file.
        """
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.log_data = json.load(f)
        return self.log_data


def count_parameters(model):
    """
    Đếm số parameters của model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Tổng số parameters
        trainable_params: Số parameters có thể train
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def set_seed(seed=42):
    """
    Set random seed cho reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
