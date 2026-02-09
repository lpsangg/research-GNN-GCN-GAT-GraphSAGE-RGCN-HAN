# RGCN (Relational Graph Convolutional Network)

## Mô tả

RGCN implementation dựa trên paper "Modeling Relational Data with Graph Convolutional Networks" (Schlichtkrull et al., ESWC 2018).

RGCN extends GCN để handle **heterogeneous graphs** với multiple node types và edge types (relations).

## Architecture

### RGCN Algorithm

RGCN generalizes GCN cho relational/heterogeneous data:

```
h_i^(l+1) = σ(Σ_r∈R Σ_j∈N_i^r (1/c_i,r) W_r^(l) h_j^(l) + W_0^(l) h_i^(l))
```

Trong đó:
- **R**: Set of relation types (edge types)
- **N_i^r**: Neighbors của node i under relation r
- **W_r^(l)**: Weight matrix cho relation r at layer l
- **c_i,r**: Normalization constant = |N_i^r|
- **W_0^(l)**: Self-loop weight matrix

### Key Concepts

#### 1. **Multiple Relations**
```
Transaction ←(performed_by)→ User
Transaction ←(uses)→ Device
```
Mỗi relation có riêng weight matrix W_r

#### 2. **Basis Decomposition**
Để giảm số parameters khi có nhiều relations:
```
W_r^(l) = Σ_b a_rb^(l) V_b^(l)
```
- V_b: Basis matrices (shared)
- a_rb: Relation-specific coefficients

#### 3. **Block-Diagonal Decomposition**
FastRGCN sử dụng block-diagonal matrices:
```
W_r = diag(B_r1, B_r2, ..., B_rB)
```
Faster và memory efficient

## 3 Variants đã implement

### 1. **Standard RGCN**
- Full RGCN với basis-decomposition
- Best accuracy
- Moderate speed

### 2. **FastRGCN**
- Block-diagonal matrices
- Faster training
- Lower memory usage
- Slightly lower accuracy

### 3. **TypedRGCN**
- Type-specific input/output projections
- Separate processing cho mỗi node type
- Most flexible
- Best cho heterogeneous data

## Fraud Detection Graph Structure

```
[Transaction] --performed_by--> [User]
      |
      |--uses--> [Device]
      
Edges:
1. user → transaction (performs)
2. transaction → user (performed_by)
3. device → transaction (used_in)
4. transaction → device (uses)
```

## Ưu điểm của RGCN

1. **Heterogeneous graphs**: Handle multiple node/edge types
2. **Relational**: Different weights cho different relations
3. **Expressive**: Capture complex relationships
4. **Flexible**: Basis/block decomposition cho efficiency
5. **Inductive**: Generalize to new relations

## So sánh với Homogeneous Models

| Feature | GCN/GAT/GraphSAGE | RGCN |
|---------|-------------------|------|
| **Graph type** | Homogeneous | Heterogeneous |
| **Edge types** | Single | Multiple |
| **Node types** | Single | Multiple |
| **Parameters** | O(d²) per layer | O(R·d²) hoặc O(B·d²) with bases |
| **Expressiveness** | Medium | High |
| **Use case** | Simple graphs | Complex relational data |
| **For Fraud** | Good | **Better** (captures user-device-transaction) |

## Sử dụng

### Training

```python
# Standard RGCN
python RGCN-HAN/train_rgcn.py

# Fast RGCN
from RGCN_HAN.train_rgcn import train_rgcn

config = {
    'model_type': 'fast',
    'hidden_channels': 64,
    'num_layers': 2,
    'num_bases': 30,      # Basis decomposition
    'epochs': 200,
}

model, test_metrics, tracker = train_rgcn(config)

# Typed RGCN
config = {
    'model_type': 'typed',
    'hidden_channels': 64,
    'num_bases': None,    # No basis decomposition
}
```

### Inference

```python
from RGCN_HAN.rgcn_model import create_rgcn_model, prepare_rgcn_data

# Prepare heterogeneous data
x_dict, edge_index, edge_type, num_nodes_dict = prepare_rgcn_data(hetero_data)

# Create model
model = create_rgcn_model(
    model_type='standard',
    num_nodes_dict=num_nodes_dict,
    in_channels=feature_dim,
    hidden_channels=64,
    num_relations=4,
    num_bases=30
)

# Load checkpoint
checkpoint = torch.load('checkpoints/RGCN/RGCN_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
pred_labels, pred_probs = model.predict(x_dict, edge_index, edge_type)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_channels | 64 | Hidden dimension |
| num_layers | 2 | Number of RGCN layers |
| num_relations | 4 | Number of edge types |
| num_bases | 30 | Number of bases (None = no decomposition) |
| dropout | 0.5 | Dropout rate |
| lr | 0.01 | Learning rate |
| weight_decay | 5e-4 | L2 regularization |
| patience | 30 | Early stopping patience |

### Basis Decomposition

**Trade-off:**
- **No bases** (num_bases=None): 
  - Parameters: O(R·d²)
  - Best accuracy
  - Slower training
  
- **Few bases** (num_bases=30):
  - Parameters: O(B·d²) where B << R
  - Faster training
  - Slight accuracy drop
  - Recommended: B = min(30, num_relations)

## Tips & Best Practices

1. **Basis decomposition**:
   - Always use khi num_relations > 10
   - Set num_bases = min(30, num_relations)
   - Trade-off: speed vs accuracy

2. **Model selection**:
   - **Standard RGCN**: Default choice, best accuracy
   - **FastRGCN**: Large graphs, need speed
   - **TypedRGCN**: Complex heterogeneous data

3. **Number of layers**:
   - 2 layers sufficient cho most graphs
   - 3 layers cho complex relations
   - Avoid too deep (overfitting)

4. **Hidden channels**:
   - 64-128 cho fraud detection
   - Lower than homogeneous models (RGCN more expressive)

5. **Node embeddings**:
   - User/Device nodes không có features → use embeddings
   - Embeddings automatically learned during training

6. **Graph construction**:
   - Use `prepare_hetero_data_with_features()` from utils
   - Aggregate features cho user/device nodes

## Kết quả mong đợi

| Model | AUC-ROC | F1-Score | Training Time (10K samples) | Parameters |
|-------|---------|----------|---------------------------|------------|
| Standard RGCN | 0.84-0.92 | 0.62-0.77 | ~12 phút | ~400K (với bases) |
| FastRGCN | 0.83-0.91 | 0.61-0.76 | ~8 phút | ~300K |
| TypedRGCN | 0.85-0.93 | 0.63-0.78 | ~15 phút | ~500K |

## Performance vs Homogeneous Models

| Model Type | AUC-ROC | Captures Relations | Scalability |
|------------|---------|-------------------|-------------|
| GCN | 0.80-0.88 | ❌ Limited | High |
| GAT | 0.82-0.90 | ❌ Limited | Medium |
| GraphSAGE | 0.81-0.89 | ❌ Limited | High |
| **RGCN** | **0.84-0.92** | ✅ **Full** | Medium |

RGCN thường **outperform homogeneous models** trên fraud detection vì:
- Capture user-transaction-device relationships
- Different weights cho different relation types
- Better modeling của heterogeneous data

## When to use RGCN?

✅ **Use RGCN when:**
- Multiple node types (user, device, transaction)
- Multiple edge types (performs, uses, etc.)
- Need to model different types of relationships
- Fraud detection with heterogeneous data

❌ **Consider alternatives when:**
- Single node type → Use GCN/GAT/GraphSAGE
- Need maximum speed → Use GraphSAGE
- Small graph (<5K nodes) → GCN/GAT sufficient

## Comparison với HAN

| Feature | RGCN | HAN |
|---------|------|-----|
| **Attention** | ❌ No | ✅ Hierarchical |
| **Relations** | Fixed weights | Learned attention |
| **Complexity** | Medium | High |
| **Speed** | Faster | Slower |
| **Accuracy** | Good | Better |

**Recommendation**: 
- RGCN: Baseline cho heterogeneous graphs
- HAN: Best performance, need hierarchical attention

## References

1. Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018
2. Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs", NeurIPS 2017
3. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

## Model Evolution

```
GCN (Homogeneous)
    ↓
RGCN (Heterogeneous + Relations)
    ↓
HAN (Heterogeneous + Hierarchical Attention)
```
