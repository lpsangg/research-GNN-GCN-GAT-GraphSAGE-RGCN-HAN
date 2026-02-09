# GCN (Graph Convolutional Network) for Fraud Detection

## Mô tả

GCN implementation dựa trên paper "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, ICLR 2017).

## Architecture

### Standard GCN
GCN sử dụng **spectral-based approach** với normalized adjacency matrix:

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

Trong đó:
- `Ã = A + I` (adjacency matrix với self-loops)
- `D̃` là degree matrix của Ã
- `W^(l)` là learnable weight matrix
- `σ` là activation function (ReLU)

### DeepGCN
Variant cho deeper networks với:
- **Residual connections**: Skip connections từ input và previous layer
- **Layer-wise transformation**: Separate linear layers
- Formula: `H^(l+1) = (1-α-θ)·GCN(H^(l)) + α·H^(0) + θ·H^(l)`

### JKNetGCN
GCN với **Jumping Knowledge Networks**:
- Aggregate representations từ tất cả layers
- 3 modes: `cat` (concatenation), `max` (max pooling), `lstm` (LSTM aggregation)
- Better capture multi-scale graph structure

## Ưu điểm của GCN

1. **Spectral foundation**: Dựa trên solid spectral graph theory
2. **Efficient**: Fast convolution với sparse matrices
3. **Localized**: Local neighborhoods (k-hop)
4. **Scalable**: Linear complexity O(|E|)

## So sánh với GNN cơ bản

| Feature | Basic GNN | GCN |
|---------|-----------|-----|
| Aggregation | Mean | Normalized (by degree) |
| Theoretical basis | Empirical | Spectral graph theory |
| Normalization | Manual | Built-in (D^(-1/2) A D^(-1/2)) |
| Performance | Good | Better |
| Stability | Lower | Higher |

## Sử dụng

### Training

```python
# Standard GCN
python GCN/train.py

# Deep GCN
from GCN.train import train_gcn

config = {
    'model_type': 'deep',
    'hidden_channels': 256,
    'num_layers': 4,
    'alpha': 0.1,      # Initial residual weight
    'theta': 0.5,      # Identity weight
    'epochs': 200,
}

model, test_metrics, tracker = train_gcn(config)

# JKNet GCN
config = {
    'model_type': 'jknet',
    'jk_mode': 'cat',  # 'cat', 'max', or 'lstm'
    'num_layers': 3,
}

model, test_metrics, tracker = train_gcn(config)
```

### Inference

```python
from GCN.model import create_gcn_model

# Create model
model = create_gcn_model(
    model_type='standard',
    in_channels=feature_dim,
    hidden_channels=128,
    num_classes=2,
    improved=False,    # Use improved GCN variant
    cached=True        # Cache normalized adjacency
)

# Load checkpoint
checkpoint = torch.load('checkpoints/GCN/GCN_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
pred_labels, pred_probs = model.predict(data.x, data.edge_index)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_channels | 128 | Hidden dimension |
| num_layers | 2 | Number of GCN layers |
| dropout | 0.5 | Dropout rate |
| lr | 0.01 | Learning rate (cao hơn GNN) |
| weight_decay | 5e-4 | L2 regularization |
| improved | False | Use improved GCN (A^2) |
| cached | True | Cache normalized adj matrix |

### DeepGCN specific
| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | 0.1 | Weight cho initial residual |
| theta | 0.5 | Weight cho identity mapping |

### JKNetGCN specific
| Parameter | Default | Description |
|-----------|---------|-------------|
| jk_mode | 'cat' | Aggregation mode |

## Tips & Best Practices

1. **Learning rate**: GCN thường dùng lr cao hơn (0.01) so với GNN (0.001)
2. **Layers**: 2-3 layers cho most datasets, DeepGCN cho deeper (4-8 layers)
3. **Cached**: Set `cached=True` khi dùng full-batch training
4. **Improved variant**: `improved=True` có thể tốt hơn trên sparse graphs
5. **JKNet mode**: 
   - `cat`: Best for small layers (<= 3)
   - `max`: Memory efficient
   - `lstm`: Best expressiveness nhưng slow hơn

## Kết quả mong đợi

| Model | AUC-ROC | F1-Score | Training Time (10K samples) |
|-------|---------|----------|---------------------------|
| Standard GCN | 0.80-0.88 | 0.55-0.70 | ~5 phút |
| Deep GCN | 0.82-0.90 | 0.58-0.72 | ~8 phút |
| JKNet GCN | 0.81-0.89 | 0.56-0.71 | ~10 phút |

## References

1. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
2. Li et al. (2019). "DeepGCNs: Can GCNs Go as Deep as CNNs?", ICCV 2019
3. Xu et al. (2018). "Representation Learning on Graphs with Jumping Knowledge Networks", ICML 2018

## Model Comparison

```
GNN (Basic) → GCN (Normalized) → DeepGCN (Residual) → JKNetGCN (Multi-scale)
    ↓              ↓                    ↓                      ↓
  Fast         Stable            Deeper networks      Better feature learning
```
