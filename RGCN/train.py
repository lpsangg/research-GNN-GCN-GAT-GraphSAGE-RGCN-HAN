"""
Training Script cho RGCN Model
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from RGCN.model import create_rgcn_model, prepare_rgcn_data
from utils import (
    load_ieee_fraud_data,
    preprocess_features,
    prepare_hetero_data_with_features,
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


def train_epoch(model, data_dict, edge_index, edge_type, y, train_mask, optimizer, device):
    """
    Train 1 epoch.
    
    Args:
        model: RGCN model
        data_dict: Dict of node features
        edge_index: Edge indices
        edge_type: Edge types
        y: Labels
        train_mask: Training mask
        optimizer: Optimizer
        device: Device
    
    Returns:
        loss: Training loss
        metrics: Training metrics
    """
    model.train()
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(data_dict, edge_index, edge_type)
    
    # Compute loss only on training nodes
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics on training set
    model.eval()
    with torch.no_grad():
        logits = model(data_dict, edge_index, edge_type)
        pred_labels = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        y_true = y[train_mask].cpu()
        y_pred = pred_labels[train_mask].cpu()
        y_prob = probs[train_mask].cpu()
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate(model, data_dict, edge_index, edge_type, y, mask, device):
    """
    Evaluate model.
    
    Args:
        model: RGCN model
        data_dict: Dict of node features
        edge_index: Edge indices
        edge_type: Edge types
        y: Labels
        mask: Boolean mask (val_mask or test_mask)
        device: Device
    
    Returns:
        loss: Evaluation loss
        metrics: Evaluation metrics
    """
    model.eval()
    
    # Forward pass
    logits = model(data_dict, edge_index, edge_type)
    
    # Compute loss
    loss = F.cross_entropy(logits[mask], y[mask])
    
    # Compute metrics
    pred_labels = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]
    
    y_true = y[mask].cpu()
    y_pred = pred_labels[mask].cpu()
    y_prob = probs[mask].cpu()
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return loss.item(), metrics


def train_rgcn(config):
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
    
    # Create heterogeneous graph
    print("\nCreating heterogeneous graph...")
    hetero_data = prepare_hetero_data_with_features(df, feature_cols)
    hetero_data = create_train_val_test_split(hetero_data)
    
    print(f"Heterogeneous graph created:")
    print(f"  - Transaction nodes: {hetero_data['transaction'].x.shape[0]}")
    print(f"  - User nodes: {hetero_data['user'].num_nodes}")
    print(f"  - Device nodes: {hetero_data['device'].num_nodes}")
    print(f"  - Number of edge types: {len(hetero_data.edge_types)}")
    
    # Convert to RGCN format
    print("\nConverting to RGCN format...")
    x_dict, edge_index, edge_type, num_nodes_dict = prepare_rgcn_data(hetero_data)
    
    # Move to device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    y = hetero_data['transaction'].y.to(device)
    train_mask = hetero_data['transaction'].train_mask.to(device)
    val_mask = hetero_data['transaction'].val_mask.to(device)
    test_mask = hetero_data['transaction'].test_mask.to(device)
    
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Number of relations: {edge_type.max().item() + 1}")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # Create model
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    num_relations = edge_type.max().item() + 1
    
    model_kwargs = {
        'num_nodes_dict': num_nodes_dict,
        'in_channels': x_dict['transaction'].shape[1],
        'hidden_channels': config.get('hidden_channels', 64),
        'num_classes': 2,
        'num_layers': config.get('num_layers', 2),
        'num_relations': num_relations,
        'dropout': config.get('dropout', 0.5),
        'use_batch_norm': config.get('use_batch_norm', True),
    }
    
    # Add model-specific kwargs
    if config.get('model_type') in ['standard', 'fast']:
        model_kwargs['num_bases'] = config.get('num_bases', min(30, num_relations))
    elif config.get('model_type') == 'typed':
        # Create in_channels_dict for typed model
        in_channels_dict = {
            'transaction': x_dict['transaction'].shape[1],
            'user': x_dict['transaction'].shape[1],  # Will be projected
            'device': x_dict['transaction'].shape[1],  # Will be projected
        }
        model_kwargs['in_channels_dict'] = in_channels_dict
        model_kwargs.pop('in_channels')
        if 'num_bases' in config:
            model_kwargs['num_bases'] = config['num_bases']
    
    model = create_rgcn_model(
        model_type=config.get('model_type', 'standard'),
        **model_kwargs
    )
    
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model type: {config.get('model_type', 'standard').upper()}")
    print(f"Number of relations: {num_relations}")
    print(f"Number of bases: {config.get('num_bases', 'None')}")
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
        patience=config.get('patience', 30),
        mode='max'
    )
    checkpoint = ModelCheckpoint(
        save_dir=config.get('checkpoint_dir', 'checkpoints'),
        model_name='RGCN',
        mode='max'
    )
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    best_val_auc = 0
    
    for epoch in range(1, config.get('epochs', 200) + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, x_dict, edge_index, edge_type, y, train_mask, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = evaluate(
            model, x_dict, edge_index, edge_type, y, val_mask, device
        )
        
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
    _, train_metrics = evaluate(model, x_dict, edge_index, edge_type, y, train_mask, device)
    print_metrics(train_metrics, prefix="Train")
    
    # Validation set
    _, val_metrics = evaluate(model, x_dict, edge_index, edge_type, y, val_mask, device)
    print_metrics(val_metrics, prefix="Validation")
    
    # Test set
    _, test_metrics = evaluate(model, x_dict, edge_index, edge_type, y, test_mask, device)
    print_metrics(test_metrics, prefix="Test")
    
    # Plot training curves
    if config.get('plot_curves', True):
        print("\nPlotting training curves...")
        plot_dir = Path(config.get('plot_dir', 'plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(
            metrics_tracker.get_history(),
            save_path=plot_dir / 'rgcn_training_curves.png'
        )
    
    return model, test_metrics, metrics_tracker


if __name__ == '__main__':
    # Configuration
    config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'checkpoint_dir': 'checkpoints/RGCN',
        'plot_dir': 'plots',
        
        # Data
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'max_samples': None,  # Set to số nhỏ (vd: 10000) để test nhanh
        
        # Model
        'model_type': 'standard',  # 'standard', 'fast', or 'typed'
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5,
        'use_batch_norm': True,
        'num_bases': 30,       # Basis-decomposition (giảm params, tốc độ training)
        
        # Training
        'epochs': 200,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 30,
        'print_every': 10,
        'plot_curves': True,
        
        # Other
        'seed': 42,
    }
    
    # Train
    model, test_metrics, tracker = train_rgcn(config)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
    print(f"Best Test F1: {test_metrics['f1']:.4f}")
