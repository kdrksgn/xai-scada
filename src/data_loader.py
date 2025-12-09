
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import MinMaxScaler

def load_batadal():
    print("Loading BATADAL...")
    df_train = pd.read_csv(config.BATADAL_TRAIN)
    df_test = pd.read_csv(config.BATADAL_TEST)
    
    # BATADAL Date format: dd/mm/yyyy hh:mm or similar. 
    # Columns: DATETIME, L_T1, L_T2... ATT_FLAG
    # We drop DATETIME and use ATT_FLAG as label
    
    # Clean up column names
    df_train.columns = [c.strip() for c in df_train.columns]
    df_test.columns = [c.strip() for c in df_test.columns]
    
    # Features (all except datetime and label)
    # BATADAL 'ATT_FLAG' is usually the label (-999 or 1 for attack, etc. need to check)
    # Usually: 1 = Normal, -1 = Attack? Or 0/1. 
    # In dataset03 (Train), usually no attacks.
    # In dataset04 (Test), attacks exist.
    
    # Assuming 'ATT_FLAG' exists.
    label_col = 'ATT_FLAG'
    if label_col not in df_train.columns:
        # Check columns
        pass
        
    X_train = df_train.drop(columns=['DATETIME', label_col], errors='ignore').values
    X_test = df_test.drop(columns=['DATETIME', label_col], errors='ignore').values
    # Map Labels: {0: 0, 1: 0, -999: 1}
    # Dataset03 has 0 (Normal). Dataset04 has 1 (Normal?) and -999 (Attack).
    # We verify this by counts, but usually -999 is the outlier.
    
    def map_label(val):
        if val == -999: return 1
        return 0
        
    y_train = np.array([map_label(x) for x in df_train[label_col].values]) if label_col in df_train.columns else np.zeros(len(df_train))
    y_test = np.array([map_label(x) for x in df_test[label_col].values]) if label_col in df_test.columns else np.zeros(len(df_test))

    print(f"BATADAL Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"BATADAL Test labels: {np.unique(y_test, return_counts=True)}")

    # Normalize
    # Handle NaNs and Infs
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # print(f"BATADAL loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, list(df_train.columns)

def load_swat():
    print("Loading SWaT...")
    files = sorted(config.SWAT_FILES)
    # print(f"Found {len(files)} SWaT files.")
    
    # Load all and concat (might be heavy, let's limit for reproduction if huge)
    # The files are daily logs.
    dfs = []
    
    # Use first 5 files for Train, next 5 for Test for demonstration/speed if too many
    # Paper uses 2015 dataset which is continuous. 2021 might be huge.
    # Let's read first 10 files.
    
    for f in files[:10]: 
        try:
            df = pd.read_csv(f)
            # Standardize columns
            df.columns = [c.strip() for c in df.columns]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)

    # Clean NaNs
    full_df = full_df.fillna(method='ffill').fillna(0)
    
    # Sort by Timestamp if possible, else assume file order is chronological
    if 'Timestamp' in full_df.columns:
        full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'], errors='coerce')
        full_df = full_df.sort_values('Timestamp')
    
    # Identify Features
    # Filter columns that are sensors/actuators
    feature_cols = [c for c in full_df.columns if any(x in c for x in ['FIT', 'LIT', 'MV', 'P', 'UV', 'AIT', 'DPIT'])]
    
    # Label
    # 'Annotation' or 'Other Anomalies'
    label_col = 'Annotation' # 'Attack' or 'Normal' usually
    
    # Map labels
    if label_col in full_df.columns:
        # If text, map to 0/1. usually 'Normal' -> 0, 'Attack' -> 1
        full_df['label'] = full_df[label_col].apply(lambda x: 1 if str(x).lower() == 'attack' else 0)
    else:
        full_df['label'] = 0
        
    print(f"SWaT Label Distribution: {full_df['label'].value_counts().to_dict()}")

    data = full_df[feature_cols].values
    labels = full_df['label'].values
    
    # Split Train/Test
    split_idx = int(len(data) * 0.8)
    
    X_train = data[:split_idx]
    y_train = labels[:split_idx]
    X_test = data[split_idx:]
    y_test = labels[split_idx:]
    
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, feature_cols

def create_windows(X, y, window_size=60, step=10):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size, step):
        Xs.append(X[i:i+window_size])
        # Label is 1 if any anomaly in window, or last step. 
        # Paper usually prediction determines anomaly at t, using t-window.
        # So label is y[i+window_size-1] or max(y[i:i+window_size])
        ys.append(max(y[i:i+window_size])) 
    return np.array(Xs), np.array(ys)
