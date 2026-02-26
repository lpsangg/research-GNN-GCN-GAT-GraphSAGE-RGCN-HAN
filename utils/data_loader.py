"""
Data Loading and Preprocessing Utilities
Chung cho tất cả 6 models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


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


class PreprocessingPipeline:
    """
    Pipeline xử lý dữ liệu an toàn, tránh data leakage.
    Fit trên train set, transform trên test set với cùng encoders/scalers.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.v_features_selected = None
        
    def _create_base_features(self, df):
        """Tạo các features cơ bản"""
        df = df.copy()
        
        # Create User_ID
        df['User_ID'] = df['card1'].astype(str) + '_' + df['addr1'].fillna(-1).astype(str)
        
        # Transaction amount features
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
        
        # Time features
        if 'TransactionDT' in df.columns:
            df['Transaction_hour'] = (df['TransactionDT'] / 3600) % 24
            df['Transaction_day'] = (df['TransactionDT'] / 86400) % 7
        
        return df
    
    def _handle_missing_values(self, df, is_train=True):
        """Xử lý missing values thông minh với missing indicators"""
        df = df.copy()
        
        # Categorical - fill với explicit missing value
        categorical_fill = {
            'DeviceInfo': 'unknown_device',
            'ProductCD': 'Unknown',
            'card4': 'missing',
            'card6': 'missing',
            'P_emaildomain': 'unknown',
            'R_emaildomain': 'unknown'
        }
        
        for col, fill_value in categorical_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        
        # Numeric card/addr - fill với -1 (explicit outlier)
        numeric_fill = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
        for col in numeric_fill:
            if col in df.columns:
                df[col] = df[col].fillna(-1)
        
        # Create missing indicators cho features quan trọng (missing có ý nghĩa)
        important_cols = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo']
        for col in important_cols:
            if col in df.columns:
                # Check if originally missing (before fillna)
                # Note: Đã fill rồi nên dùng cách khác
                pass
        
        return df
    
    def _encode_categorical(self, df, is_train=True):
        """Encode categorical features an toàn"""
        df = df.copy()
        
        categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            col_encoded = col + '_encoded'
            
            if is_train:
                # Fit encoder trên train
                self.encoders[col] = LabelEncoder()
                df[col_encoded] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform test - handle unseen categories
                encoder = self.encoders.get(col)
                if encoder is None:
                    raise ValueError(f"Encoder for {col} not found. Must fit on train first.")
                
                # Handle unseen categories bằng cách map về 'unknown'
                def safe_transform(val):
                    try:
                        return encoder.transform([str(val)])[0]
                    except ValueError:
                        # Unseen category - map to 'unknown' or first class
                        return encoder.transform(['unknown'])[0] if 'unknown' in encoder.classes_ else 0
                
                df[col_encoded] = df[col].astype(str).apply(safe_transform)
        
        return df
    
    def _select_features(self, df, is_train=True):
        """Chọn features thông minh"""
        
        if is_train:
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
            
            # M features (match features) - CRITICAL cho fraud detection
            m_features = [f'M{i}' for i in range(1, 10) if f'M{i}' in df.columns]
            
            # V features - lọc theo missing rate
            v_features = [f'V{i}' for i in range(1, 340) if f'V{i}' in df.columns]
            v_features_filtered = [v for v in v_features if df[v].isna().sum() < len(df) * 0.7]
            v_features = v_features_filtered[:100]
            self.v_features_selected = v_features  # Lưu lại để dùng cho test
            
            print(f"V features: {len(v_features_filtered)} available after filtering, using {len(v_features)}")
            
            # Encoded categorical features
            categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
            encoded_features = [col + '_encoded' for col in categorical_cols if col in df.columns]
            
            # Combine all features
            feature_cols = (numeric_features + card_features + addr_features + 
                           time_features + c_features + d_features + m_features +
                           v_features + encoded_features)
            
            # Lưu lại
            self.feature_cols = [col for col in feature_cols if col in df.columns]
            self.numeric_cols = numeric_features + card_features + addr_features + time_features + \
                               c_features + d_features + m_features + v_features
            self.categorical_cols = encoded_features
            
            print(f"Total features: {len(self.feature_cols)} (Numeric={len(self.numeric_cols)}, "
                  f"C={len(c_features)}, D={len(d_features)}, M={len(m_features)}, V={len(v_features)})")
        else:
            # Test set - dùng same features từ train
            if self.feature_cols is None:
                raise ValueError("Must fit on train data first")
        
        return self.feature_cols
    
    def _scale_features(self, df, is_train=True):
        """Scale numeric features"""
        df = df.copy()
        
        # Chỉ scale numeric columns (không scale encoded categorical)
        numeric_cols_to_scale = [col for col in self.numeric_cols if col in df.columns]
        
        if is_train:
            # Fit scaler trên train
            df[numeric_cols_to_scale] = self.scaler.fit_transform(
                df[numeric_cols_to_scale].fillna(0)
            )
        else:
            # Transform test
            df[numeric_cols_to_scale] = self.scaler.transform(
                df[numeric_cols_to_scale].fillna(0)
            )
        
        return df
    
    def fit_transform(self, df_trans, df_ident):
        """
        Fit pipeline trên training data và transform.
        
        Args:
            df_trans: Training transaction dataframe
            df_ident: Training identity dataframe
        
        Returns:
            df: Processed dataframe
            feature_cols: List of feature columns
        """
        # Merge
        df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
        print(f"Train - Merged shape: {df.shape}")
        
        # Pipeline steps
        df = self._create_base_features(df)
        df = self._handle_missing_values(df, is_train=True)
        df = self._encode_categorical(df, is_train=True)
        
        # Select features (phải làm trước khi scale để biết numeric cols)
        feature_cols = self._select_features(df, is_train=True)
        
        # Fill remaining NaN in features
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Scale features
        df = self._scale_features(df, is_train=True)
        
        return df, feature_cols
    
    def transform(self, df_trans, df_ident):
        """
        Transform test data với fitted pipeline.
        
        Args:
            df_trans: Test transaction dataframe
            df_ident: Test identity dataframe
        
        Returns:
            df: Processed dataframe
            feature_cols: List of feature columns
        """
        if self.feature_cols is None:
            raise ValueError("Pipeline not fitted. Call fit_transform on train data first.")
        
        # Merge
        df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
        print(f"Test - Merged shape: {df.shape}")
        
        # Pipeline steps
        df = self._create_base_features(df)
        df = self._handle_missing_values(df, is_train=False)
        df = self._encode_categorical(df, is_train=False)
        
        # Fill remaining NaN in features
        df[self.feature_cols] = df[self.feature_cols].fillna(0)
        
        # Scale features
        df = self._scale_features(df, is_train=False)
        
        return df, self.feature_cols
    
    def save(self, filepath):
        """Lưu pipeline"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'encoders': self.encoders,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'numeric_cols': self.numeric_cols,
                'categorical_cols': self.categorical_cols,
                'v_features_selected': self.v_features_selected
            }, f)
        print(f"Pipeline saved to {filepath}")
    
    def load(self, filepath):
        """Load pipeline"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.encoders = data['encoders']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
            self.numeric_cols = data['numeric_cols']
            self.categorical_cols = data['categorical_cols']
            self.v_features_selected = data['v_features_selected']
        print(f"Pipeline loaded from {filepath}")


def preprocess_features(df_trans, df_ident, is_train=True, pipeline=None):
    """
    LEGACY FUNCTION - Backward compatibility.
    Khuyến nghị dùng PreprocessingPipeline class cho production.
    
    Preprocess và feature engineering cho fraud detection.
    
    Args:
        df_trans: Transaction dataframe
        df_ident: Identity dataframe
        is_train: True nếu là training data
        pipeline: PreprocessingPipeline object (optional)
    
    Returns:
        df: Merged và processed dataframe
        feature_cols: List các cột features đã xử lý
    """
    if pipeline is None:
        # Fallback to old logic (NOT RECOMMENDED)
        print("WARNING: Using legacy preprocessing. Recommend using PreprocessingPipeline class.")
        pipeline = PreprocessingPipeline()
        if is_train:
            return pipeline.fit_transform(df_trans, df_ident)
        else:
            raise ValueError("For test data, must provide fitted pipeline object")
    else:
        if is_train:
            return pipeline.fit_transform(df_trans, df_ident)
        else:
            return pipeline.transform(df_trans, df_ident)


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
