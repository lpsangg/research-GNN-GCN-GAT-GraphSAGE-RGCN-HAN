# GraphSAGE (SAmple and aggreGatE) for Fraud Detection

## Mô tả

GraphSAGE implementation dựa trên paper "Inductive Representation Learning on Large Graphs" (Hamilton et al., NeurIPS 2017).

## Architecture

### GraphSAGE Algorithm

GraphSAGE là **inductive learning** framework với 3 bước chính:

**Step 1 - Sample neighborhood:**
```
Sample K neighbors cho mỗi node (thay vì dùng tất cả)
```

**Step 2 - Aggregate neighbor features:**
```
h_N(v)^k = AGGREGATE_k({h_u^(k-1) : u ∈ N(v)})
```

**Step 3 - Update với self features:**
```
h_v^k = σ(W^k · [h_v^(k-1) || h_N(v)^k])
```

### Aggregator Functions

GraphSAGE support nhiều aggregators:

#### 1. **Mean Aggregator** (Default, most stable)
```
AGGREGATE = mean({h_u : u ∈ N(v)})
```
- Simple và effective
- Symmetric (permutation invariant)

#### 2. **Max Aggregator**
```
AGGREGATE = max({h_u : u ∈ N(v)})
```
- Capture prominent features
- Better cho skewed distributions

#### 3. **LSTM Aggregator**
```
AGGREGATE = LSTM({h_u : u ∈ N(v)})
```
- More expressive
- Need random permutation (LSTM not permutation invariant)

#### 4. **Min Aggregator**
```
AGGREGATE = min({h_u : u ∈ N(v)})
```
- Rare features focus

## 4 Variants đã implement

### 1. **Standard GraphSAGE**
- Single aggregator (mean/max/min/lstm)
- Most common và stable
- Good baseline

### 2. **DeepGraphSAGE**
- Deeper networks (3+ layers)
- Residual connections
- Jumping Knowledge (JK-Net)
- Better feature learning

### 3. **AttentionGraphSAGE**
- Attention-based aggregation
- Multi-head attention
- Combine GraphSAGE + GAT benefits
- Adaptive neighbor weighting

### 4. **MultiAggregatorSAGE**
- Multiple aggregators combined
- Concatenate mean + max + min
- Capture different aspects
- More expressive

## Ưu điểm của GraphSAGE

1. **Inductive**: Generalize to unseen nodes/graphs
2. **Scalable**: Sampling-based (không cần full graph)
3. **Flexible**: Multiple aggregators
4. **Fast**: Linear complexity với sampling
5. **Memory efficient**: Mini-batch training

## So sánh với các models khác

| Feature | GCN | GAT | GraphSAGE |
|---------|-----|-----|-----------|
| **Learning** | Transductive | Transductive/Inductive | **Inductive** |
| **Scalability** | Limited | Limited | **High** (sampling) |
| **Aggregation** | Normalized | Attention | Flexible (mean/max/lstm) |
| **Memory** | Need full graph | Need full graph | **Mini-batch** |
| **Speed** | Fast | Medium | Fast-Medium |
| **Unseen nodes** | ❌ | Limited | ✅ |
| **Expected AUC** | 0.80-0.88 | 0.82-0.90 | 0.81-0.89 |

## Sử dụng

### Training

```python
# Standard GraphSAGE với mean aggregator
python GraphSAGE/train.py

# Deep GraphSAGE với JK-Net
from GraphSAGE.train import train_graphsage

config = {
    'model_type': 'deep',
    'aggregator': 'mean',
    'jk_mode': 'cat',
    'num_layers': 3,
    'hidden_channels': 128,
}

model, test_metrics, tracker = train_graphsage(config)

# Attention GraphSAGE
config = {
    'model_type': 'attention',
    'heads': 4,
    'hidden_channels': 128,
}

# Multi-Aggregator GraphSAGE
config = {
    'model_type': 'multi_aggr',
    'aggregators': ['mean', 'max', 'min'],
}
```

### Inference

```python
from GraphSAGE.model import create_graphsage_model

# Create model
model = create_graphsage_model(
    model_type='standard',
    in_channels=feature_dim,
    hidden_channels=128,
    num_classes=2,
    aggregator='mean'
)

# Load checkpoint
checkpoint = torch.load('checkpoints/GraphSAGE/GraphSAGE_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
pred_labels, pred_probs = model.predict(data.x, data.edge_index)

# Mini-batch inference cho large graphs
from torch_geometric.loader import NeighborLoader

subgraph_loader = NeighborLoader(
    data, 
    num_neighbors=[10, 5],
    batch_size=1024,
    input_nodes=None
)

logits = model.inference(data.x, subgraph_loader, device)
```

### Mini-batch Training với NeighborLoader

```python
from torch_geometric.loader import NeighborLoader

# Create train loader
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 2-hop: 25 neighbors layer 1, 10 layer 2
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True
)

# Training loop
for batch in train_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_channels | 128 | Hidden dimension |
| num_layers | 2 | Number of layers |
| dropout | 0.5 | Dropout rate |
| lr | 0.01 | Learning rate |
| weight_decay | 5e-4 | L2 regularization |
| aggregator | 'mean' | Aggregation function |

### Sampling parameters (for NeighborLoader)
| Parameter | Default | Description |
|-----------|---------|-------------|
| num_neighbors | [25, 10] | Neighbors per layer |
| batch_size | 1024 | Batch size |

### Model-specific

**DeepGraphSAGE:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| jk_mode | 'cat' | JK aggregation: 'cat', 'max', None |

**AttentionGraphSAGE:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| heads | 4 | Number of attention heads |

**MultiAggregatorSAGE:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| aggregators | ['mean', 'max', 'min'] | List of aggregators |

## Aggregator Comparison

| Aggregator | Speed | Expressiveness | Stability | Best for |
|------------|-------|----------------|-----------|----------|
| **Mean** | Fast | Medium | High | General purpose |
| **Max** | Fast | Medium | High | Prominent features |
| **Min** | Fast | Low | Medium | Rare features |
| **LSTM** | Slow | High | Low | Complex patterns |
| **Multi** | Medium | High | High | Best performance |

## Tips & Best Practices

1. **Aggregator selection**:
   - Start với `mean` (most stable)
   - Try `max` nếu cần highlight prominent features
   - Use `multi_aggr` cho best performance
   - Avoid `lstm` unless có specific need (slower, less stable)

2. **Sampling**:
   - num_neighbors=[25, 10] cho 2 layers
   - Higher sampling → better quality, slower
   - Lower sampling → faster, might lose info

3. **Layers**:
   - 2 layers sufficient cho most graphs
   - Use DeepGraphSAGE cho 3+ layers
   - More layers → larger receptive field

4. **Batch size**:
   - 1024-2048 cho medium graphs
   - Larger batch → more stable gradient
   - Limited by GPU memory

5. **Learning rate**:
   - GraphSAGE stable với lr=0.01 (similar to GCN)
   - Can use higher lr than GAT (0.01 vs 0.005)

6. **Inductive learning**:
   - GraphSAGE best cho unseen nodes
   - Train on subset, test on all
   - No retraining needed for new nodes

## Kết quả mong đợi

| Model | AUC-ROC | F1-Score | Training Time (10K samples) | Scalability |
|-------|---------|----------|---------------------------|-------------|
| Standard (mean) | 0.81-0.89 | 0.57-0.72 | ~7 phút | High |
| Deep + JK | 0.82-0.90 | 0.59-0.74 | ~10 phút | High |
| Attention | 0.83-0.90 | 0.60-0.74 | ~12 phút | Medium |
| Multi-Aggr | 0.84-0.91 | 0.61-0.75 | ~15 phút | Medium |

## Khi nào dùng GraphSAGE?

✅ **Use GraphSAGE when:**
- Large graphs (millions of nodes)
- Need inductive learning (unseen nodes)
- Limited memory
- Need fast inference
- Require scalability

❌ **Avoid GraphSAGE when:**
- Small graphs (< 10K nodes) → GCN/GAT might be better
- Need interpretability → GAT better
- Need best possible accuracy → Try GAT or ensemble

## References

1. Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs", NeurIPS 2017
2. Xu et al. (2018). "Representation Learning on Graphs with Jumping Knowledge Networks", ICML 2018
3. Ying et al. (2018). "Graph Convolutional Neural Networks for Web-Scale Recommender Systems", KDD 2018

## Model Evolution

```
GCN (Spectral, fixed)
    ↓
GAT (Attention, adaptive)
    ↓
GraphSAGE (Sampling, inductive)
    ↓
PinSAGE (Industrial scale, Pinterest)
```

**Trade-off:** Expressiveness ↔ Scalability
