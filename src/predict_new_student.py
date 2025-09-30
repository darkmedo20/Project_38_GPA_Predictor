# File: src/predict_new_student.py
# Purpose: Demonstrates Task I6 (Model Serialization) by loading the trained model
# and using it to predict the Final Year GPA for a batch of new students (supporting X.XX format).

import joblib
import pandas as pd
import os
import numpy as np

# --- Configuration ---
MODEL_FILE = '../models/final_gpa_predictor_rf.pkl'
# This is the CSV file containing the data for the new students you want to predict
NEW_STUDENT_DATA_FILE = '../data/new_students_for_prediction.csv'

def load_model(file_path):
    """Loads the trained model from the specified path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)
    try:
        model = joblib.load(full_path)
        print(f"[SUCCESS] Model loaded from: {full_path}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at: {full_path}")
        print("Please run src/model_pipeline.py first to create the model file.")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load model. Error: {e}")
        return None

def predict_new_student_gpa(model, new_student_data):
    """
    Uses the loaded model to predict GPA for one or multiple new students (batch prediction).
    
    Args:
        model: The loaded Scikit-learn model object.
        new_student_data (pd.DataFrame): DataFrame containing new student features (one or more rows).

    Returns:
        list: A list of predicted GPAs formatted as strings (X.XX).
    """
    if model is None:
        return ["Error: Model not loaded"]

    # 1. Prediction: The model predicts the continuous GPA score for all rows
    predictions = model.predict(new_student_data)

    # 2. Formatting: Convert numpy array to list and format each element to X.XX
    predictions_list = predictions.tolist()
    
    # Ensure prediction is non-negative and format to two decimal places
    # Apply max(0, gpa) for safety, then format to X.XX string.
    formatted_predictions = [f"{max(0, gpa):.2f}" for gpa in predictions_list]

    return formatted_predictions

if __name__ == "__main__":
    
    # --- Step 1: Load Input Data from CSV File ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, NEW_STUDENT_DATA_FILE)

    try:
        # Load the new student data from the CSV file
        # This data file must contain the StudentID and the features needed for prediction.
        df_all_students = pd.read_csv(data_path)
        print(f"[SUCCESS] Loaded new student data from: {NEW_STUDENT_DATA_FILE}")

    except FileNotFoundError:
        print(f"[FATAL ERROR] Input CSV file not found at: {data_path}")
        print("Please ensure '../data/new_students_for_prediction.csv' exists.")
        exit()

    # Separate StudentID for output (not needed for prediction)
    student_ids = df_all_students['StudentID']
    
    # Select only the features required by the model, dropping StudentID
    # NOTE: The feature columns MUST match the columns used during model training in src/model_pipeline.py
    feature_columns = ['Gender_F', 'Year1_GPA', 'Year2_GPA', 'Year3_GPA', 'Credit_Hours_Avg', 'Attendance_Rate']
    df_new_students_features = df_all_students[feature_columns]

    # --- Step 2: Load Model and Predict ---
    
    # Load the saved model (Task I6)
    model = load_model(MODEL_FILE)

    if model:
        # Predict the GPAs (Task I6)
        predicted_gpas = predict_new_student_gpa(model, df_new_students_features)

        # --- Step 3: Display Results ---
        print("\n--- Predicted Final Year GPAs (X.XX Format) ---")
        
        # Combine student IDs and predictions for clear output
        results = pd.DataFrame({
            'StudentID': student_ids,
            'Predicted_GPA': predicted_gpas
        })
        
        print(results)
        print("\nPrediction Script Complete.")
