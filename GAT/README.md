# GAT (Graph Attention Network) for Fraud Detection

## Mô tả

GAT implementation dựa trên paper "Graph Attention Networks" (Veličković et al., ICLR 2018) và improved version GATv2 (Brody et al., ICLR 2022).

## Architecture

### Standard GAT
GAT sử dụng **attention mechanism** để tự động học importance của neighbors:

**Step 1 - Compute attention coefficients:**
```
e_ij = LeakyReLU(a^T [W·h_i || W·h_j])
```

**Step 2 - Normalize với softmax:**
```
α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
```

**Step 3 - Aggregate với attention weights:**
```
h_i' = σ(Σ_{j∈N(i)} α_ij · W·h_j)
```

**Multi-head attention:**
```
h_i' = ||_{k=1}^K σ(Σ_{j∈N(i)} α_ij^k · W^k·h_j)
```

### GATv2 (Improved)
GATv2 fixes the **static attention problem** của GAT:

**Key difference:**
- **GAT**: `e_ij = a^T·LeakyReLU(W[h_i||h_j])`  ← LeakyReLU trước
- **GATv2**: `e_ij = a^T·LeakyReLU(W·h_i + W·h_j)`  ← LeakyReLU sau

Cho phép **dynamic attention** thay vì chỉ phụ thuộc vào node pairs.

### MultiHeadGAT
Flexible head aggregation strategies:
- **concat**: Concatenate tất cả heads `[h_1 || h_2 || ... || h_K]`
- **mean**: Average heads `mean(h_1, h_2, ..., h_K)`
- **max**: Max pooling `max(h_1, h_2, ..., h_K)`
- **attention**: Learned attention over heads

## Ưu điểm của GAT

1. **Adaptive**: Tự động học importance của neighbors (không cần degree normalization)
2. **Inductive**: Generalize sang unseen nodes/graphs
3. **Interpretable**: Attention weights có thể visualize
4. **Expressive**: Multi-head cho different aspects
5. **No preprocessing**: Không cần tính normalized adjacency

## So sánh với GCN

| Feature | GCN | GAT |
|---------|-----|-----|
| **Aggregation** | Fixed (degree-based) | Learned (attention) |
| **Weights** | Same cho tất cả neighbors | Different cho mỗi neighbor |
| **Preprocessing** | Cần normalize adjacency | Không cần |
| **Inductive** | Transductive | Inductive |
| **Interpretability** | Low | High (attention viz) |
| **Computational cost** | O(\|E\|) | O(\|E\|·K) (K heads) |
| **Dropout** | ~0.5 | ~0.6 (cao hơn) |
| **Learning rate** | 0.01 | 0.005 |

## Sử dụng

### Training

```python
# Standard GAT
python GAT/train.py

# GATv2
from GAT.train import train_gat

config = {
    'model_type': 'v2',
    'hidden_channels': 64,
    'heads': 8,
    'num_layers': 2,
    'share_weights': False,
    'epochs': 300,
}

model, test_metrics, tracker = train_gat(config)

# MultiHead with attention aggregation
config = {
    'model_type': 'multihead',
    'head_aggregation': 'attention',  # or 'mean', 'max'
    'heads': 8,
}

model, test_metrics, tracker = train_gat(config)
```

### Inference

```python
from GAT.model import create_gat_model

# Create model
model = create_gat_model(
    model_type='standard',
    in_channels=feature_dim,
    hidden_channels=64,
    heads=8,
    num_classes=2
)

# Load checkpoint
checkpoint = torch.load('checkpoints/GAT/GAT_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
pred_labels, pred_probs = model.predict(data.x, data.edge_index)

# Get attention weights (for visualization)
if hasattr(model, 'get_attention_weights'):
    attention_weights = model.get_attention_weights(data.x, data.edge_index)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_channels | 64 | Hidden dimension **per head** |
| num_layers | 2 | Number of GAT layers |
| heads | 8 | Number of attention heads |
| dropout | 0.6 | Dropout rate (cao hơn GCN) |
| lr | 0.005 | Learning rate |
| weight_decay | 5e-4 | L2 regularization |
| patience | 30 | Early stopping patience (cao hơn) |
| negative_slope | 0.2 | LeakyReLU slope |

### Model-specific

**Standard GAT:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| concat_heads | True | Concat (True) hoặc average (False) |

**GATv2:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| share_weights | False | Share weights src/target |

**MultiHeadGAT:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| head_aggregation | 'concat' | 'concat', 'mean', 'max', 'attention' |

## Tips & Best Practices

1. **Heads**: 
   - 8 heads cho most tasks
   - Nhiều heads → better expressiveness nhưng slower
   - Last layer nên average heads (concat=False)

2. **Hidden channels**:
   - Set per-head dimension (total = hidden × heads)
   - hidden=64, heads=8 → total=512 dimension

3. **Dropout**:
   - GAT cần dropout cao hơn (~0.6) vì expressive hơn
   - Dropout applied ở input và attention

4. **Learning rate**:
   - GAT stable với lr thấp hơn GCN (0.005 vs 0.01)
   - Adam optimizer recommended

5. **Early stopping**:
   - Patience cao hơn (30+) vì training chậm hơn
   - Monitor validation AUC

6. **Model selection**:
   - **Standard GAT**: Most datasets
   - **GATv2**: Complex graphs, better expressiveness
   - **MultiHead-attention**: Best performance nhưng slow nhất

## Kết quả mong đợi

| Model | AUC-ROC | F1-Score | Training Time (10K samples) | Parameters |
|-------|---------|----------|---------------------------|------------|
| Standard GAT | 0.82-0.90 | 0.58-0.74 | ~10 phút | ~500K |
| GATv2 | 0.83-0.91 | 0.60-0.75 | ~12 phút | ~500K |
| MultiHead-Attn | 0.84-0.92 | 0.61-0.76 | ~15 phút | ~600K |

## Visualization

GAT cho phép visualize attention weights:

```python
# Get attention weights
model.eval()
attention_weights = model.get_attention_weights(data.x, data.edge_index)

# attention_weights[layer] = (edge_index, alpha)
# alpha shape: [E, heads] - attention weight cho mỗi edge, mỗi head

# Visualize cho layer 0, head 0
edge_index, alpha = attention_weights[0]
edge_attention = alpha[:, 0]  # First head

# Plot graph với edge thickness = attention weight
import networkx as nx
G = nx.Graph()
for i, (src, dst) in enumerate(edge_index.T.numpy()):
    weight = edge_attention[i].item()
    G.add_edge(src, dst, weight=weight)
```

## References

1. Veličković et al. (2018). "Graph Attention Networks", ICLR 2018
2. Brody et al. (2022). "How Attentive are Graph Attention Networks?", ICLR 2022
3. Lee et al. (2019). "Attention Models in Graphs: A Survey", ACM TKDD

## Model Evolution

```
GCN (Fixed weights) 
    ↓
GAT (Attention) 
    ↓
GATv2 (Dynamic attention) 
    ↓
MultiHeadGAT (Flexible aggregation)
```

Performance: **GCN < GAT < GATv2 < MultiHeadGAT**  
Speed: **GCN > GAT > GATv2 > MultiHeadGAT**
