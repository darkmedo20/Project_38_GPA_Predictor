# debug_missing_cells.py
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipeline import GPAModelPipeline

# Load your CSV data
csv_path = "data/new_students_for_prediction.csv"
df = pd.read_csv(csv_path)

print("📊 CSV Data Analysis:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst row:\n{df.iloc[0].to_dict()}")
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, "models", "best_gpa_model.pkl")

print(f"\n🔍 Loading model from: {model_path}")

try:
    pipeline = GPAModelPipeline()
    pipeline.load_model(model_path)
    
    print("✅ Model loaded")
    print(f"Feature names: {pipeline.feature_names}")
    print(f"Missing strategy: {pipeline.missing_strategy}")
    
    # Check which features are in the CSV
    available_features = [f for f in pipeline.feature_names if f in df.columns]
    missing_features = [f for f in pipeline.feature_names if f not in df.columns]
    
    print(f"\n📋 Feature Analysis:")
    print(f"Available in CSV: {available_features}")
    print(f"Missing from CSV: {missing_features}")
    
    # Prepare data for prediction
    X = df[available_features].copy()
    print(f"\n📐 Data for prediction shape: {X.shape}")
    print(f"Data:\n{X}")
    
    # Test the missing handler directly
    print(f"\n🔍 Testing MissingValueHandler...")
    if hasattr(pipeline, 'missing_handler'):
        handler = pipeline.missing_handler
        print(f"Handler type: {type(handler)}")
        print(f"Handler strategy: {handler.strategy}")
        
        # Try analyze_missing_patterns
        try:
            missing_report = handler.analyze_missing_patterns(X)
            print(f"Missing report type: {type(missing_report)}")
            print(f"Missing report keys: {list(missing_report.keys()) if isinstance(missing_report, dict) else 'Not a dict'}")
            print(f"Missing cells: {missing_report.get('missing_cells', 'Key not found')}")
        except Exception as e:
            print(f"❌ Error in analyze_missing_patterns: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Try prediction
    print(f"\n🎯 Attempting prediction...")
    try:
        predictions = pipeline.predict(X)
        print(f"✅ Predictions successful!")
        for i, pred in enumerate(predictions):
            print(f"Student {i+1} ({df.iloc[i]['StudentID']}): GPA = {pred:.2f}")
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Model loading error: {str(e)}")
    import traceback
    traceback.print_exc()