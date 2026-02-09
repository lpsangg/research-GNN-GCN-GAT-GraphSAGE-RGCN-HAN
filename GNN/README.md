# GNN (Graph Neural Network) for Fraud Detection

## Mô tả

Basic GNN implementation sử dụng Message Passing cơ bản để phát hiện gian lận trong giao dịch.

## Architecture

### BasicGNN
- **Input Layer**: Linear projection
- **GNN Layers**: Message Passing với mean aggregation
- **Batch Normalization**: Chuẩn hóa sau mỗi layer
- **MLP Classifier**: 2-layer MLP với ReLU và Dropout

### ImprovedGNN
- Tất cả features của BasicGNN plus:
- **Residual Connections**: Skip connections từ input
- **Layer Weighting**: Learnable weights cho từng layer
- **Deeper MLP**: 3-layer classifier

## Message Passing

GNN Layer thực hiện:
1. **Transform**: Linear transformation của node features
2. **Aggregate**: Mean aggregation từ neighbors
3. **Update**: Update node representations

```
h_i^(l+1) = σ(W^(l) · MEAN({h_j^(l) : j ∈ N(i) ∪ {i}}))
```

## Sử dụng

### Training

```python
# Basic usage
python GNN/train.py

# Custom config
from GNN.train import train_gnn

config = {
    'model_type': 'improved',
    'hidden_channels': 256,
    'num_layers': 3,
    'epochs': 200,
    'lr': 0.001,
}

model, test_metrics, tracker = train_gnn(config)
```

### Inference

```python
from GNN.model import create_gnn_model

# Load model
model = create_gnn_model(
    model_type='basic',
    in_channels=feature_dim,
    hidden_channels=128,
    num_classes=2
)

# Load checkpoint
checkpoint = torch.load('checkpoints/GNN/GNN_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
pred_labels, pred_probs = model.predict(data.x, data.edge_index)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_channels | 128 | Hidden dimension |
| num_layers | 2 | Number of GNN layers |
| dropout | 0.5 | Dropout rate |
| lr | 0.001 | Learning rate |
| weight_decay | 5e-4 | L2 regularization |
| patience | 20 | Early stopping patience |

## So sánh với các models khác

| Model | Complexity | Expressiveness | Training Speed |
|-------|-----------|----------------|----------------|
| **GNN** | Low | Basic | Fast |
| GCN | Low | Medium | Fast |
| GAT | Medium | High | Medium |
| GraphSAGE | Medium | Medium | Fast |
| RGCN/HAN | High | Very High | Slow |

## Tips

1. **Số layers**: 2-3 layers thường đủ cho fraud detection
2. **Hidden dim**: 128-256 cho datasets vừa
3. **Dropout**: 0.5 giúp tránh overfitting
4. **Edge strategy**: `user_device_time` cho kết quả tốt nhất
5. **Batch norm**: Luôn bật cho training ổn định

## Kết quả mong đợi

- **AUC-ROC**: 0.75 - 0.85
- **F1-Score**: 0.50 - 0.65
- **Training time**: ~5-10 phút (10K samples, CPU)
