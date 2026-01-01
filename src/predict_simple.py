# predict_simple.py
import pandas as pd
import numpy as np
import joblib
import os

def load_model_simple(model_path):
    """Load model without using GPAModelPipeline."""
    try:
        model_data = joblib.load(model_path)
        return model_data
    except:
        # Try pickle with simple handler
        import pickle
        
        class SimpleMissingHandler:
            def __init__(self):
                pass
        
        # Register the class
        import __main__
        __main__.MissingValueHandler = SimpleMissingHandler
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def predict_students(csv_path, model_path):
    """Make predictions for new students."""
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} students from {csv_path}")
    
    # Load model
    model_data = load_model_simple(model_path)
    model = model_data['model']
    feature_names = model_data.get('feature_names', [])
    
    print(f"Model type: {type(model).__name__}")
    print(f"Expected features: {feature_names}")
    
    # Prepare features
    # Remove StudentID
    if 'StudentID' in df.columns:
        student_ids = df['StudentID']
        df_features = df.drop('StudentID', axis=1)
    else:
        student_ids = [f"Student_{i}" for i in range(len(df))]
        df_features = df.copy()
    
    # Make sure we have all required features
    missing_features = [f for f in feature_names if f not in df_features.columns]
    extra_features = [f for f in df_features.columns if f not in feature_names]
    
    if missing_features:
        print(f"Warning: Adding missing features: {missing_features}")
        for feature in missing_features:
            df_features[feature] = 0
    
    if extra_features:
        print(f"Warning: Removing extra features: {extra_features}")
        df_features = df_features.drop(columns=extra_features)
    
    # Reorder columns to match training
    df_features = df_features[feature_names]
    
    print(f"\nFinal data shape: {df_features.shape}")
    
    # Make predictions
    predictions = model.predict(df_features.values)
    
    # Cap predictions to 0-4 GPA range
    predictions = np.clip(predictions, 0.0, 4.0)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, (student_id, pred) in enumerate(zip(student_ids, predictions)):
        print(f"\nStudent #{i+1}: {student_id}")
        print(f"  Predicted Final GPA: {pred:.2f}")
        
        # Add insights
        if pred >= 3.5:
            print(f"  ✅ Excellent performance expected")
        elif pred >= 3.0:
            print(f"  👍 Good performance expected")
        elif pred >= 2.0:
            print(f"  ⚠️  Average performance - may need support")
        else:
            print(f"  ❌ Low performance - intervention recommended")
    
    print("\n" + "="*60)
    
    return predictions

if __name__ == "__main__":
    # Paths
    csv_path = "data/new_students_for_prediction.csv"
    model_path = "models/best_gpa_model.pkl"
    
    # Convert model if needed
    if model_path.endswith('.pkl'):
        fixed_path = model_path.replace('.pkl', '_fixed.joblib')
        if not os.path.exists(fixed_path):
            print("Converting pickle model to joblib...")
            # Run conversion
            exec(open("src/fix_model_pickle.py").read())
            model_path = fixed_path
    
    predict_students(csv_path, model_path)