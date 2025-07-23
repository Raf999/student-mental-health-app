import joblib

model_path = "c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/stacked_model.pkl"
scaler_path = "c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/minmax_scaler.pkl"

try:
    model = joblib.load(model_path)
    print("\n✅ Model loaded successfully.")

    if hasattr(model, 'feature_names_in_'):
        print("Model expects features in this exact order:")
        print(model.feature_names_in_)
    else:
        print("⚠️ model.feature_names_in_ not found.")
        print("Try checking your training script to confirm column order.")
except Exception as e:
    print(f"❌ Could not load model: {e}")

try:
    scaler = joblib.load(scaler_path)
    print("\n✅ Scaler loaded successfully.")

    if hasattr(scaler, 'feature_names_in_'):
        print("Scaler expects these columns:")
        print(scaler.feature_names_in_)
    else:
        print("⚠️ scaler.feature_names_in_ not found.")
except Exception as e:
    print(f"❌ Could not load scaler: {e}")
    
import os
print("Current working directory:", os.getcwd())