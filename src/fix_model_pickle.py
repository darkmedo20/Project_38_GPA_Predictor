# fix_model_pickle.py
import pickle
import joblib
import os

# Define the MissingValueHandler class to allow unpickling
class MissingValueHandler:
    def __init__(self, strategy='contextual'):
        self.strategy = strategy
        self.imputation_values = {}
    
    def analyze_missing_patterns(self, df):
        return {'missing_cells': df.isnull().sum().sum() if df is not None else 0}
    
    def handle_missing_values(self, df):
        return df.fillna(0) if df is not None else df

# Paths
model_path = "models/best_gpa_model.pkl"
fixed_path = "models/best_gpa_model_fixed.joblib"

print(f"Loading pickle model from: {model_path}")

try:
    # Load the pickle file with our class defined
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("✅ Successfully unpickled model data")
    print(f"Keys in model data: {list(model_data.keys())}")
    
    # Extract the actual sklearn model
    model = model_data.get('model')
    if model is None:
        print("❌ No model found in saved data")
        exit(1)
    
    print(f"Model type: {type(model)}")
    
    # Get feature names
    feature_names = model_data.get('feature_names', [])
    print(f"Feature names: {feature_names}")
    
    # Create new model data without problematic classes
    new_model_data = {
        'model': model,
        'feature_names': feature_names,
        'model_type': model_data.get('model_type', 'random_forest'),
        'scaler': model_data.get('scaler', None),
        'min_gpa': model_data.get('min_gpa', 0.0),
        'max_gpa': model_data.get('max_gpa', 4.0),
        'timestamp': model_data.get('timestamp', '2024-01-01')
    }
    
    # Save with joblib
    joblib.dump(new_model_data, fixed_path)
    print(f"\n✅ Saved fixed model to: {fixed_path}")
    
    # Test loading
    test_data = joblib.load(fixed_path)
    print("✅ Successfully loaded fixed model")
    print(f"Test keys: {list(test_data.keys())}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()