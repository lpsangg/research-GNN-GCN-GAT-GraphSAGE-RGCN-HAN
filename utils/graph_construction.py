import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


def create_user_id_refined(df_trans, df_ident):
    # (Code của bạn giữ nguyên phần này)
    df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device')

    # Giả sử đã chạy xong logic tạo User_ID
    if 'User_ID' not in df.columns:
        df['User_ID'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

    return df


def prepare_hetero_data(df, feature_cols):
    """
    Xây dựng đồ thị Dị thể (Heterogeneous Graph) chuẩn cho PyG.
    Có 3 loại node: Transaction, User, Device
    """
    data = HeteroData()

    # 1. Tạo mapping ID cho từng loại node để tránh xung đột chỉ số
    # Unique IDs
    unique_user_id = df['User_ID'].unique()
    unique_device_id = df['DeviceInfo'].unique()
    unique_trans_id = df['TransactionID'].unique()

    # Mapping từ String sang Index (0, 1, 2...)
    user_map = {id: i for i, id in enumerate(unique_user_id)}
    device_map = {id: i for i, id in enumerate(unique_device_id)}
    # Transaction thì index chính là thứ tự dòng trong df (nếu df đã sort)

    # 2. Xây dựng Features cho Node Transaction (Chính)
    # Lưu ý: Chỉ Transaction mới có features đầy đủ từ bảng
    x_trans = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)
    y = torch.tensor(df['isFraud'].values, dtype=torch.long)

    # Gán vào data
    data['transaction'].x = x_trans
    data['transaction'].y = y

    # 3. Xây dựng Features cho Node User và Device
    # Vì User/Device không có feature riêng trong bảng gốc, ta có 2 cách:
    # Cách 1: Dùng One-hot encoding (Tốn RAM)
    # Cách 2: Dùng Embedding (Tốt nhất cho RGCN) - Ở đây ta khởi tạo ngẫu nhiên hoặc hằng số
    # Để đơn giản, ta gán feature rỗng hoặc vector 1 để model tự học embedding sau này
    data['user'].num_nodes = len(unique_user_id)
    data['device'].num_nodes = len(unique_device_id)

    # (Optional) Gán feature giả lập nếu model yêu cầu x phải có
    # data['user'].x = torch.eye(len(unique_user_id)) # Cẩn thận OOM nếu user quá nhiều
    # Cách tối ưu thường dùng: dùng torch.nn.Embedding trong model, không truyền x vào đây.

    # 4. Xây dựng Edge Index (Quan trọng)
    # Lấy index tương ứng từ map
    src_user = [user_map[u] for u in df['User_ID']]
    dst_trans = list(range(len(df)))  # Transaction index

    src_device = [device_map[d] for d in df['DeviceInfo']]

    # Định nghĩa các loại cạnh (Edge Types)
    # Quan hệ: User thực hiện Transaction
    data['user', 'performs', 'transaction'].edge_index = torch.tensor(
        [src_user, dst_trans], dtype=torch.long
    )

    # Quan hệ: Device được dùng trong Transaction
    data['device', 'used_in', 'transaction'].edge_index = torch.tensor(
        [src_device, dst_trans], dtype=torch.long
    )

    # Quan hệ ngược (Bắt buộc cho GNN để tin truyền 2 chiều)
    data['transaction', 'performed_by', 'user'].edge_index = torch.tensor(
        [dst_trans, src_user], dtype=torch.long
    )
    data['transaction', 'uses', 'device'].edge_index = torch.tensor(
        [dst_trans, src_device], dtype=torch.long
    )

    return data


def prepare_homogeneous_data(df, feature_cols, edge_strategy='user_device_time'):
    """
    Xây dựng đồ thị Đồng nhất (Homogeneous Graph) cho GNN, GCN, GAT, GraphSAGE.
    Chỉ có 1 loại node: Transaction
    Edges được tạo dựa trên các quan hệ giữa transactions.
    
    Args:
        df: DataFrame với transaction data
        feature_cols: List các cột features
        edge_strategy: Chiến lược tạo edges
            - 'user_device_time': Kết nối transactions cùng user, device, hoặc gần thời gian
            - 'user_device': Chỉ kết nối cùng user hoặc device
            - 'knn': K-Nearest Neighbors dựa trên features
    
    Returns:
        data: PyG Data object
    """
    # 1. Node Features
    x = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x.numpy()), dtype=torch.float)
    
    # Labels
    y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    
    # 2. Tạo Edges giữa các Transactions
    edge_list = []
    
    if edge_strategy == 'user_device_time':
        # Tạo edges cho transactions có cùng User
        user_groups = df.groupby('User_ID').groups
        for user_id, indices in user_groups.items():
            indices = list(indices)
            if len(indices) > 1:
                # Kết nối tất cả các transactions của cùng 1 user
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        edge_list.append([indices[i], indices[j]])
                        edge_list.append([indices[j], indices[i]])  # Bidirectional
        
        # Tạo edges cho transactions có cùng Device
        device_groups = df.groupby('DeviceInfo').groups
        for device_id, indices in device_groups.items():
            if device_id == 'unknown_device':
                continue
            indices = list(indices)
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        edge_list.append([indices[i], indices[j]])
                        edge_list.append([indices[j], indices[i]])
        
        # Tạo edges cho transactions gần nhau về thời gian (nếu có time feature)
        if 'TransactionDT' in df.columns:
            # Sort by time và kết nối k transactions liền kề
            df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
            k_neighbors = 5
            for i in range(len(df_sorted) - 1):
                for j in range(i+1, min(i+k_neighbors, len(df_sorted))):
                    edge_list.append([i, j])
                    edge_list.append([j, i])
    
    elif edge_strategy == 'user_device':
        # Chỉ kết nối cùng user hoặc device
        user_groups = df.groupby('User_ID').groups
        for user_id, indices in user_groups.items():
            indices = list(indices)
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        edge_list.append([indices[i], indices[j]])
                        edge_list.append([indices[j], indices[i]])
        
        device_groups = df.groupby('DeviceInfo').groups
        for device_id, indices in device_groups.items():
            if device_id == 'unknown_device':
                continue
            indices = list(indices)
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        edge_list.append([indices[i], indices[j]])
                        edge_list.append([indices[j], indices[i]])
    
    elif edge_strategy == 'knn':
        # K-Nearest Neighbors dựa trên feature similarity
        from sklearn.neighbors import NearestNeighbors
        k = 10
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(x.numpy())
        distances, indices = nbrs.kneighbors(x.numpy())
        
        for i in range(len(indices)):
            for j in indices[i][1:]:  # Skip first (self)
                edge_list.append([i, j])
    
    # Remove duplicates
    edge_list = list(set(map(tuple, edge_list)))
    
    if len(edge_list) == 0:
        # Fallback: create some random edges
        print("Warning: No edges created. Creating random edges...")
        num_nodes = len(df)
        for i in range(num_nodes):
            for _ in range(min(5, num_nodes-1)):
                j = np.random.randint(0, num_nodes)
                if i != j:
                    edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 3. Tạo Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def create_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Tạo train/val/test masks cho Data object.
    
    Args:
        data: PyG Data hoặc HeteroData object
        train_ratio, val_ratio, test_ratio: Tỷ lệ split
    
    Returns:
        data: Data object với train_mask, val_mask, test_mask
    """
    num_nodes = data.y.size(0) if hasattr(data, 'y') else data['transaction'].y.size(0)
    
    # Random permutation
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # Assign masks
    if isinstance(data, HeteroData):
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        data['transaction'].test_mask = test_mask
    else:
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    
    return data


def aggregate_features_for_heterogeneous(df, feature_cols):
    """
    Aggregate features từ transactions lên User và Device nodes.
    Tạo features cho user/device bằng mean, max, min của transactions.
    
    Args:
        df: DataFrame
        feature_cols: List features
    
    Returns:
        user_features: Dict {user_id: feature_vector}
        device_features: Dict {device_id: feature_vector}
    """
    # Aggregate for Users
    user_features = {}
    for user_id in df['User_ID'].unique():
        user_trans = df[df['User_ID'] == user_id][feature_cols]
        # Mean, Max, Min features
        mean_feat = user_trans.mean().values
        max_feat = user_trans.max().values
        min_feat = user_trans.min().values
        # Concatenate
        user_features[user_id] = np.concatenate([mean_feat, max_feat, min_feat])
    
    # Aggregate for Devices
    device_features = {}
    for device_id in df['DeviceInfo'].unique():
        device_trans = df[df['DeviceInfo'] == device_id][feature_cols]
        mean_feat = device_trans.mean().values
        max_feat = device_trans.max().values
        min_feat = device_trans.min().values
        device_features[device_id] = np.concatenate([mean_feat, max_feat, min_feat])
    
    return user_features, device_features


def prepare_hetero_data_with_features(df, feature_cols):
    """
    Xây dựng đồ thị Heterogeneous với features đầy đủ cho tất cả loại nodes.
    Phiên bản cải tiến của prepare_hetero_data().
    
    Args:
        df: DataFrame
        feature_cols: List features
    
    Returns:
        data: HeteroData object với features cho tất cả node types
    """
    data = HeteroData()
    
    # 1. Mapping IDs
    unique_user_id = df['User_ID'].unique()
    unique_device_id = df['DeviceInfo'].unique()
    
    user_map = {id: i for i, id in enumerate(unique_user_id)}
    device_map = {id: i for i, id in enumerate(unique_device_id)}
    
    # 2. Transaction Features
    x_trans = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)
    y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    
    data['transaction'].x = x_trans
    data['transaction'].y = y
    
    # 3. User và Device Features (Aggregated)
    user_feat_dict, device_feat_dict = aggregate_features_for_heterogeneous(df, feature_cols)
    
    # Convert to tensors
    user_features_list = [user_feat_dict[uid] for uid in unique_user_id]
    device_features_list = [device_feat_dict[did] for did in unique_device_id]
    
    data['user'].x = torch.tensor(np.array(user_features_list), dtype=torch.float)
    data['device'].x = torch.tensor(np.array(device_features_list), dtype=torch.float)
    
    # 4. Edge Index
    src_user = [user_map[u] for u in df['User_ID']]
    dst_trans = list(range(len(df)))
    src_device = [device_map[d] for d in df['DeviceInfo']]
    
    data['user', 'performs', 'transaction'].edge_index = torch.tensor(
        [src_user, dst_trans], dtype=torch.long
    )
    data['device', 'used_in', 'transaction'].edge_index = torch.tensor(
        [src_device, dst_trans], dtype=torch.long
    )
    data['transaction', 'performed_by', 'user'].edge_index = torch.tensor(
        [dst_trans, src_user], dtype=torch.long
    )
    data['transaction', 'uses', 'device'].edge_index = torch.tensor(
        [dst_trans, src_device], dtype=torch.long
    )
    
    return data


# --- Cách sử dụng ---
# # Cho Homogeneous models (GNN, GCN, GAT, GraphSAGE):
# df_full = create_user_id_refined(df_trans, df_ident)
# homo_data = prepare_homogeneous_data(df_full, feature_cols, edge_strategy='user_device_time')
# homo_data = create_train_val_test_split(homo_data)
# 
# # Cho Heterogeneous models (RGCN, HAN):
# hetero_data = prepare_hetero_data_with_features(df_full, feature_cols)
# hetero_data = create_train_val_test_split(hetero_data)
