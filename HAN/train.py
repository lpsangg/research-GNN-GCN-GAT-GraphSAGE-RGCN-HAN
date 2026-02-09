"""
Training Script cho HAN (Heterogeneous Attention Network)
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from HAN.model import create_han_model, extract_meta_paths
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


def train_epoch(model, model_type, data, y, train_mask, optimizer, device):
    """
    Train 1 epoch.
    
    Args:
        model: HAN model
        model_type: 'han', 'custom', or 'simple'
        data: Graph data (hetero_data or processed data)
        y: Labels
        train_mask: Training mask
        optimizer: Optimizer
        device: Device
    
    Returns:
        loss: Training loss
        metrics: Training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass based on model type
    if model_type == 'simple':
        x_dict, tut_edge_index, tdt_edge_index = data
        logits = model(x_dict, tut_edge_index, tdt_edge_index)
    elif model_type == 'han':
        x_dict, edge_index_dict = data
        logits = model(x_dict, edge_index_dict)
    elif model_type == 'custom':
        x_dict, edge_index_dict, meta_path_dict = data
        logits = model(x_dict, edge_index_dict, meta_path_dict)
    
    # Compute loss
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    model.eval()
    with torch.no_grad():
        if model_type == 'simple':
            logits = model(x_dict, tut_edge_index, tdt_edge_index)
        elif model_type == 'han':
            logits = model(x_dict, edge_index_dict)
        elif model_type == 'custom':
            logits = model(x_dict, edge_index_dict, meta_path_dict)
        
        pred_labels = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        y_true = y[train_mask].cpu()
        y_pred = pred_labels[train_mask].cpu()
        y_prob = probs[train_mask].cpu()
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return loss.item(), metrics


@torch.no_grad()
def evaluate(model, model_type, data, y, mask, device):
    """
    Evaluate model.
    
    Args:
        model: HAN model
        model_type: 'han', 'custom', or 'simple'
        data: Graph data
        y: Labels
        mask: Boolean mask
        device: Device
    
    Returns:
        loss: Evaluation loss
        metrics: Evaluation metrics
    """
    model.eval()
    
    # Forward pass
    if model_type == 'simple':
        x_dict, tut_edge_index, tdt_edge_index = data
        logits = model(x_dict, tut_edge_index, tdt_edge_index)
    elif model_type == 'han':
        x_dict, edge_index_dict = data
        logits = model(x_dict, edge_index_dict)
    elif model_type == 'custom':
        x_dict, edge_index_dict, meta_path_dict = data
        logits = model(x_dict, edge_index_dict, meta_path_dict)
    
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


def train_han(config):
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
    
    # Giới hạn samples nếu cần
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
    print(f"  - Edge types: {len(hetero_data.edge_types)}")
    
    # Prepare data based on model type
    model_type = config.get('model_type', 'simple')
    
    if model_type == 'simple':
        # Extract meta-paths
        print("\nExtracting meta-paths...")
        tut_edge_index, tdt_edge_index = extract_meta_paths(hetero_data)
        print(f"  - TUT edges: {tut_edge_index.shape[1]}")
        print(f"  - TDT edges: {tdt_edge_index.shape[1]}")
        
        # Move to device
        x_dict = {'transaction': hetero_data['transaction'].x.to(device)}
        tut_edge_index = tut_edge_index.to(device)
        tdt_edge_index = tdt_edge_index.to(device)
        data = (x_dict, tut_edge_index, tdt_edge_index)
        
    elif model_type == 'han':
        # Use HANConv (requires metadata)
        metadata = (hetero_data.node_types, hetero_data.edge_types)
        
        # Move to device
        hetero_data = hetero_data.to(device)
        x_dict = {node_type: hetero_data[node_type].x 
                  for node_type in hetero_data.node_types 
                  if hasattr(hetero_data[node_type], 'x')}
        edge_index_dict = {edge_type: hetero_data[edge_type].edge_index 
                          for edge_type in hetero_data.edge_types}
        data = (x_dict, edge_index_dict)
        
    elif model_type == 'custom':
        # Extract meta-paths for custom model
        print("\nExtracting meta-paths for custom model...")
        tut_edge_index, tdt_edge_index = extract_meta_paths(hetero_data)
        
        meta_path_dict = {
            'TUT': (hetero_data['transaction'].x, tut_edge_index),
            'TDT': (hetero_data['transaction'].x, tdt_edge_index),
        }
        
        # Move to device
        x_dict = {'transaction': hetero_data['transaction'].x.to(device)}
        edge_index_dict = {
            edge_type: hetero_data[edge_type].edge_index.to(device)
            for edge_type in hetero_data.edge_types
        }
        meta_path_dict = {
            name: (nodes.to(device), edges.to(device))
            for name, (nodes, edges) in meta_path_dict.items()
        }
        data = (x_dict, edge_index_dict, meta_path_dict)
    
    # Labels and masks
    y = hetero_data['transaction'].y.to(device)
    train_mask = hetero_data['transaction'].train_mask.to(device)
    val_mask = hetero_data['transaction'].val_mask.to(device)
    test_mask = hetero_data['transaction'].test_mask.to(device)
    
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # Create model
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    in_channels = hetero_data['transaction'].x.shape[1]
    
    if model_type == 'simple':
        model = create_han_model(
            model_type='simple',
            num_nodes_dict={
                'transaction': hetero_data['transaction'].x.shape[0],
                'user': hetero_data['user'].num_nodes,
                'device': hetero_data['device'].num_nodes,
            },
            in_channels=in_channels,
            hidden_channels=config.get('hidden_channels', 64),
            num_classes=2,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.6)
        )
    elif model_type == 'han':
        model = create_han_model(
            model_type='han',
            in_channels=in_channels,
            hidden_channels=config.get('hidden_channels', 64),
            num_classes=2,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.6),
            metadata=metadata
        )
    elif model_type == 'custom':
        in_channels_dict = {
            'transaction': in_channels,
            'user': in_channels,
            'device': in_channels,
        }
        model = create_han_model(
            model_type='custom',
            in_channels_dict=in_channels_dict,
            hidden_channels=config.get('hidden_channels', 64),
            num_classes=2,
            num_heads=config.get('num_heads', 8),
            num_meta_paths=2,
            dropout=config.get('dropout', 0.6)
        )
    
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model type: {model_type.upper()}")
    print(f"Number of attention heads: {config.get('num_heads', 8)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.get('lr', 0.005),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    # Training utilities
    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(
        patience=config.get('patience', 40),
        mode='max'
    )
    checkpoint = ModelCheckpoint(
        save_dir=config.get('checkpoint_dir', 'checkpoints'),
        model_name='HAN',
        mode='max'
    )
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    best_val_auc = 0
    
    for epoch in range(1, config.get('epochs', 300) + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, model_type, data, y, train_mask, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = evaluate(
            model, model_type, data, y, val_mask, device
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
            print(f"\nEpoch {epoch}/{config.get('epochs', 300)}")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics.get('auc_roc', 0):.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_metrics.get('auc_roc', 0):.4f}")
            
            # Print semantic attention weights if available
            if model_type == 'simple' and hasattr(model, 'get_semantic_weights'):
                weights = model.get_semantic_weights()
                print(f"Semantic Attention: TUT={weights[0]:.3f}, TDT={weights[1]:.3f}")
            elif model_type == 'custom' and hasattr(model, 'get_semantic_attention_weights'):
                weights = model.get_semantic_attention_weights()
                if weights is not None:
                    print(f"Semantic Attention: {weights}")
        
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
    _, train_metrics = evaluate(model, model_type, data, y, train_mask, device)
    print_metrics(train_metrics, prefix="Train")
    
    # Validation set
    _, val_metrics = evaluate(model, model_type, data, y, val_mask, device)
    print_metrics(val_metrics, prefix="Validation")
    
    # Test set
    _, test_metrics = evaluate(model, model_type, data, y, test_mask, device)
    print_metrics(test_metrics, prefix="Test")
    
    # Print final semantic attention weights
    print("\n" + "="*50)
    print("SEMANTIC ATTENTION ANALYSIS")
    print("="*50)
    
    if model_type == 'simple':
        weights = model.get_semantic_weights()
        print(f"Meta-path importance:")
        print(f"  - Transaction-User-Transaction (TUT): {weights[0]:.4f}")
        print(f"  - Transaction-Device-Transaction (TDT): {weights[1]:.4f}")
        
        if weights[0] > weights[1]:
            print("\n→ User relationships are more important for fraud detection")
        else:
            print("\n→ Device relationships are more important for fraud detection")
    
    # Plot training curves
    if config.get('plot_curves', True):
        print("\nPlotting training curves...")
        plot_dir = Path(config.get('plot_dir', 'plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(
            metrics_tracker.get_history(),
            save_path=plot_dir / 'han_training_curves.png'
        )
    
    return model, test_metrics, metrics_tracker


if __name__ == '__main__':
    # Configuration
    config = {
        'data_dir': 'dataset/ieee-fraud-detection',
        'checkpoint_dir': 'checkpoints/HAN',
        'plot_dir': 'plots',
        
        # Data
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'max_samples': None,  # Set to small number for quick testing
        
        # Model
        'model_type': 'simple',  # 'simple', 'han', or 'custom'
        'hidden_channels': 64,
        'num_heads': 8,
        'dropout': 0.6,        # High dropout cho attention
        
        # Training
        'epochs': 300,
        'lr': 0.005,
        'weight_decay': 5e-4,
        'patience': 40,        # HAN needs more patience
        'print_every': 10,
        'plot_curves': True,
        
        # Other
        'seed': 42,
    }
    
    # Train
    model, test_metrics, tracker = train_han(config)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Test AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
    print(f"Best Test F1: {test_metrics['f1']:.4f}")
