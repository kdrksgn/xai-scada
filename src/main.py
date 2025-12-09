
import numpy as np
import data_loader
import model as model_lib
import explain
import ontology
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_performance(y_true, y_pred):
    # Thresholding
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_bin)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return f1

def main():
    # 1. Load Data
    print(">>> 1. Loading Data...")
    # Using BATADAL for verification as SWaT A8 labeling is uncertain (all normal?)
    X_train, y_train, X_test, y_test, features = data_loader.load_batadal()
    # X_train, y_train, X_test, y_test, features = data_loader.load_swat()
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"Features: {len(features)} detected.")
    
    # Windowing
    print(">>> Creating Windows...")
    X_train_w, y_train_w = data_loader.create_windows(X_train, y_train, config.WINDOW_SIZE)
    X_test_w, y_test_w = data_loader.create_windows(X_test, y_test, config.WINDOW_SIZE)
    print(f"Windowed Train: {X_train_w.shape}, Label: {y_train_w.shape}")
    print(f"Windowed Test: {X_test_w.shape}, Label: {y_test_w.shape}")
    
    # 2. Model
    print(">>> 2. Creating & Training Model...")
    input_shape = (X_train_w.shape[1], X_train_w.shape[2])
    lstm_model = model_lib.create_model(input_shape)
    
    # Train
    model_lib.train_model(lstm_model, X_train_w, y_train_w, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    
    # 3. Evaluation
    print(">>> 3. Evaluation...")
    y_pred = lstm_model.predict(X_test_w)
    evaluate_performance(y_test_w, y_pred)
    
    # 4. Explainability
    print(">>> 4. Generating Explanations (OntoXAI)...")
    
    # Select an anomaly index
    anom_indices = np.where(y_test_w == 1)[0]
    if len(anom_indices) > 0:
        idx = anom_indices[0] # Take first anomaly
        print(f"Explaining Anomaly at Index {idx}...")
        
        sample = X_test_w[idx:idx+1]
        background = X_train_w[:100] # Use subset of train as background
        
        # Compute SHAP
        shap_vals = explain.compute_shap(lstm_model, background, sample, n_runs=3) # Reduced runs for speed
        
        # Get Top Features
        top_feats = explain.get_top_features(shap_vals[0], features)
        print("Top Feature Attributions:")
        for f, s in top_feats:
            print(f"  {f}: {s:.4f}")
            
        # Ontology Mapping
        ontology.generate_alert(top_feats)
        
        # Calculate Variance (Mock/Simulated check)
        # We can run explain multiple times on same sample to check stability if needed.
    else:
        print("No anomalies found in test set to explain.")

if __name__ == "__main__":
    main()
