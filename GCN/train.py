"""
Training Script cho GCN Model
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from GCN.model import create_gcn_model
from utils import (
    load_ieee_fraud_data,
    preprocess_features,
    prepare_homogeneous_data,
    create_train_val_test_split,
    compute_metrics,
    print_metrics,
    MetricsTracker,
    EarlyStopping,
    ModelCheckpoint,
    set_seed,
    plot_training_curves,
    count_parameters
)


def train_epoch(model, data, optimizer, device):
    """
    Train 1 epoch.
    
    Args:
        model: GCN model
        data: PyG Data object
        optimizer: Optimizer
        device: Device
    
    Returns:
        loss: Training loss
        metrics: Training metrics
    """
    model.train()
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    
    # Compute loss only on training nodes
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics on training set
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred_labels = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        y_true = data.y[data.train_mask].cpu()
        y_pred = pred_labels[data.train_mask].cpu()
        y_prob = probs[data.train_mask].cpu()
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate(model, data, mask, device):
    """
    Evaluate model.
    
    Args:
        model: GCN model
        data: PyG Data object
        mask: Boolean mask (val_mask or test_mask)
        device: Device
    
    Returns:
        loss: Evaluation loss
        metrics: Evaluation metrics
    """
    model.eval()
    data = data.to(device)
    
    # Forward pass
    logits = model(data.x, data.edge_index)
    
    # Compute loss
    loss = F.cross_entropy(logits[mask], data.y[mask])
    
    # Compute metrics
    pred_labels = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]
    
    y_true = data.y[mask].cpu()
    y_pred = pred_labels[mask].cpu()
    y_prob = probs[mask].cpu()
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return loss.item(), metrics


def train_gcn(config):
    """
    Main training function.
    
    Args:
        config: Dictionary chứa configuration
    """
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    train_trans, train_ident, _, _ = load_ieee_fraud_data(config['data_dir'])
    df, feature_cols = preprocess_features(train_trans, train_ident)
    
    # Giới hạn số samples nếu cần (cho testing nhanh)
    if config.get('max_samples'):
        df = df.sample(n=min(config['max_samples'], len(df)), random_state=42)
        print(f"Using {len(df)} samples for quick testing")
    
    # Create graph
    print("\nCreating homogeneous graph...")
    data = prepare_homogeneous_data(
        df, 
        feature_cols, 
        edge_strategy=config.get('edge_strategy', 'user_device')
    )
    data = create_train_val_test_split(
        data,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15)
    )
    
    print(f"Graph created: {data}")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    # Create model
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    model = create_gcn_model(
        model_type=config.get('model_type', 'standard'),
        in_channels=data.x.shape[1],
        hidden_channels=config.get('hidden_channels', 128),
        num_classes=2,
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5),
        use_batch_norm=config.get('use_batch_norm', True),
        improved=config.get('improved', False),
        cached=config.get('cached', True)
    )
    
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model type: {config.get('model_type', 'standard').upper()}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    # Training utilities
    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(
        patience=config.get('patience', 20),
        mode='max'
    )
    checkpoint = ModelCheckpoint(
        save_dir=config.get('checkpoint_dir', 'checkpoints'),
        model_name='GCN',
        mode='max'
    )
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    best_val_auc = 0
    
    for epoch in range(1, config.get('epochs', 200) + 1):
        # Train
        train_loss, train_metrics = train_epoch(model, data, optimizer, device)
        
        # Validate
        val_loss, val_metrics = evaluate(model, data, data.val_mask, device)
        
        # Track metrics
        metrics_tracker.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_auc': train_metrics.get('auc_roc', 0),
            'val_auc': val_metrics.get('auc_roc', 0),
        })
        
        # Print progress
        if epoch % config.get('print_every', 10) == 0:
            print(f"\nEpoch {epoch}/{config.get('epochs', 200)}")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics.get('auc_roc', 0):.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_metrics.get('auc_roc', 0):.4f}")
        
        # Save best model
        if val_metrics.get('auc_roc', 0) > best_val_auc:
            best_val_auc = val_metrics.get('auc_roc', 0)
            checkpoint.save(model, optimizer, epoch, best_val_auc, val_metrics)
        
        # Early stopping
        early_stopping(val_metrics.get('auc_roc', 0), epoch)
        if early_stopping.should_stop():
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    print("\n" + "="*50)
    print("LOADING BEST MODEL")
    print("="*50)
    
    checkpoint.load(model, optimizer)
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Train set
    _, train_metrics = evaluate(model, data, data.train_mask, device)
    print_metrics(train_metrics, prefix="Train")
    
    # Validation set
    _, val_metrics = evaluate(model, data, data.val_mask, device)
    print_metrics(val_metrics, prefix="Validation")
    
    # Test set
    _, test_metrics = evaluate(model, data, data.test_mask, device)
    print_metrics(test_metrics, prefix="Test")
    
    # Plot training curves
    if config.get('plot_curves', True):
        print("\nPlotting training curves...")
        plot_dir = Path(config.get('plot_dir', 'plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(
            metrics_tracker.get_history(),
            save_path=plot_dir / 'gcn_training_curves.png'
        )
    
    return model, test_metrics, metrics_tracker


if __name__ == '__main__':
    # Configuration
    config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'checkpoint_dir': 'checkpoints/GCN',
        'plot_dir': 'plots',
        
        # Data
        'edge_strategy': 'user_device',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'max_samples': None,  # Set to số nhỏ (vd: 10000) để test nhanh
        
        # Model
        'model_type': 'standard',  # 'standard', 'deep', or 'jknet'
        'hidden_channels': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'use_batch_norm': True,
        'improved': False,  # Improved GCN variant (A^2 instead of A)
        'cached': True,     # Cache normalized adjacency (good for full-batch)
        
        # Training
        'epochs': 200,
        'lr': 0.01,         # GCN thường dùng lr cao hơn GNN
        'weight_decay': 5e-4,
        'patience': 20,
        'print_every': 10,
        'plot_curves': True,
        
        # Other
        'seed': 42,
    }
    
    # Train
    model, test_metrics, tracker = train_gcn(config)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
    print(f"Best Test F1: {test_metrics['f1']:.4f}")
