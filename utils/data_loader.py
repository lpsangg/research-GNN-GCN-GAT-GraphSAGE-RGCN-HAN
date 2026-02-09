"""
Data Loading and Preprocessing Utilities
Chung cho tất cả 6 models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_ieee_fraud_data(data_dir='dataset/ieee-fraud-detection'):
    """
    Load IEEE Fraud Detection dataset từ thư mục.
    
    Args:
        data_dir: Đường dẫn đến thư mục chứa data
    
    Returns:
        train_transaction, train_identity, test_transaction, test_identity
    """
    data_path = Path(data_dir)
    
    print("Loading IEEE Fraud Detection dataset...")
    
    train_transaction = pd.read_csv(data_path / 'train_transaction.csv')
    train_identity = pd.read_csv(data_path / 'train_identity.csv')
    test_transaction = pd.read_csv(data_path / 'test_transaction.csv')
    test_identity = pd.read_csv(data_path / 'test_identity.csv')
    
    print(f"Train transactions: {len(train_transaction)}")
    print(f"Train identities: {len(train_identity)}")
    print(f"Test transactions: {len(test_transaction)}")
    print(f"Test identities: {len(test_identity)}")
    
    return train_transaction, train_identity, test_transaction, test_identity


def preprocess_features(df_trans, df_ident, is_train=True):
    """
    Preprocess và feature engineering cho fraud detection.
    
    Args:
        df_trans: Transaction dataframe
        df_ident: Identity dataframe
        is_train: True nếu là training data
    
    Returns:
        df: Merged và processed dataframe
        feature_cols: List các cột features đã xử lý
    """
    # Merge transaction với identity
    df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
    
    print(f"Merged data shape: {df.shape}")
    
    # --- 1. Handle Missing Values ---
    
    # Fill DeviceInfo
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device')
    
    # Fill ProductCD (categorical)
    df['ProductCD'] = df['ProductCD'].fillna('Unknown')
    
    # Fill card features
    card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    for col in card_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(-1)
    
    # Fill addr columns
    df['addr1'] = df['addr1'].fillna(-1)
    df['addr2'] = df['addr2'].fillna(-1)
    
    # Fill email domain
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('unknown')
    
    # --- 2. Feature Engineering ---
    
    # Create User_ID (combination of card + address)
    df['User_ID'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    
    # Transaction amount features
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        df['TransactionAmt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
    
    # Time features (nếu có TransactionDT)
    if 'TransactionDT' in df.columns:
        df['Transaction_hour'] = (df['TransactionDT'] / 3600) % 24
        df['Transaction_day'] = (df['TransactionDT'] / 86400) % 7
    
    # --- 3. Encode Categorical Features ---
    
    categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    
    # --- 4. Select Feature Columns ---
    
    # Numeric features
    numeric_features = ['TransactionAmt', 'TransactionAmt_log', 'TransactionAmt_decimal']
    
    # Card features
    card_features = ['card1', 'card2', 'card3', 'card5']
    
    # Address features
    addr_features = ['addr1', 'addr2']
    
    # Time features
    time_features = []
    if 'Transaction_hour' in df.columns:
        time_features = ['Transaction_hour', 'Transaction_day']
    
    # C features (counts)
    c_features = [f'C{i}' for i in range(1, 15) if f'C{i}' in df.columns]
    
    # D features (time deltas)
    d_features = [f'D{i}' for i in range(1, 16) if f'D{i}' in df.columns]
    
    # V features (Vesta engineered features) - chọn một số quan trọng
    v_features = [f'V{i}' for i in range(1, 340) if f'V{i}' in df.columns]
    # Giới hạn số lượng V features (quá nhiều)
    v_features = v_features[:50]  # Chỉ lấy 50 đầu tiên
    
    # Encoded categorical features
    encoded_features = [col + '_encoded' for col in categorical_cols if col in df.columns]
    
    # Combine all features
    feature_cols = (numeric_features + card_features + addr_features + 
                   time_features + c_features + d_features + 
                   v_features + encoded_features)
    
    # Chỉ giữ các cột tồn tại trong df
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"Total features selected: {len(feature_cols)}")
    print(f"Feature categories: Numeric={len(numeric_features)}, Card={len(card_features)}, "
          f"C={len(c_features)}, D={len(d_features)}, V={len(v_features)}")
    
    # Fill NaN trong feature columns
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols


def get_feature_statistics(df, feature_cols):
    """
    Lấy thống kê về features.
    
    Args:
        df: DataFrame
        feature_cols: List features
    
    Returns:
        stats_dict: Dictionary chứa statistics
    """
    stats = {
        'num_features': len(feature_cols),
        'num_samples': len(df),
        'fraud_rate': df['isFraud'].mean() if 'isFraud' in df.columns else None,
        'missing_rate': df[feature_cols].isnull().mean().mean(),
        'feature_ranges': {}
    }
    
    # Range cho mỗi feature
    for col in feature_cols:
        stats['feature_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    return stats


def balance_dataset(df, target_col='isFraud', strategy='undersample', ratio=1.0):
    """
    Balance dataset cho fraud detection (imbalanced data).
    
    Args:
        df: DataFrame
        target_col: Tên cột target
        strategy: 'undersample', 'oversample', hoặc 'smote'
        ratio: Tỷ lệ minority/majority sau khi balance
    
    Returns:
        df_balanced: Balanced dataframe
    """
    from sklearn.utils import resample
    
    # Separate majority and minority classes
    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]
    
    print(f"Original - Majority: {len(df_majority)}, Minority: {len(df_minority)}")
    
    if strategy == 'undersample':
        # Undersample majority class
        n_samples = int(len(df_minority) / ratio)
        df_majority_downsampled = resample(df_majority, 
                                          replace=False,
                                          n_samples=n_samples,
                                          random_state=42)
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
    elif strategy == 'oversample':
        # Oversample minority class
        n_samples = int(len(df_majority) * ratio)
        df_minority_upsampled = resample(df_minority,
                                        replace=True,
                                        n_samples=n_samples,
                                        random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    else:
        raise ValueError("Strategy must be 'undersample' or 'oversample'")
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced - Total: {len(df_balanced)}, Fraud rate: {df_balanced[target_col].mean():.4f}")
    
    return df_balanced
