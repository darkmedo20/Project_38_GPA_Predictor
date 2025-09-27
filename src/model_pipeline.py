# File: src/model_pipeline.py
# Purpose: Executes Exploratory Data Analysis (Task I3) and initiates Model Training (Task I4),
# then saves the best-performing model (Task I6).

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib # New import for saving and loading models

# --- Configuration ---
FILE_NAME = 'cleaned_student_performance.csv'
TARGET_COLUMN = 'Final_Year_GPA'
MODEL_FILE = '../models/final_gpa_predictor_rf.pkl' # Define file path to save the model
TEST_SIZE = 0.2  # 20% of data for testing
RANDOM_SEED = 42 # For reproducibility

def load_data(file_name):
    """Loads data, similar to data_loader.py, but returns the DataFrame."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'data', file_name)
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"[LOAD ERROR] Could not load data from {file_path}. Error: {e}")
        return None

def perform_eda(df):
    """
    Task I3: Performs Exploratory Data Analysis, focusing on correlations.
    """
    print("\n--- Task I3: Exploratory Data Analysis (EDA) ---")
    
    # 1. Visualize Distribution of the Target Variable (Final GPA)
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=5, color='darkblue')
    plt.title(f'Distribution of {TARGET_COLUMN}')
    plt.xlabel('Final Year GPA')
    plt.ylabel('Number of Students')
    plt.show()

    # 2. Correlation Heatmap
    # Check correlations between all numerical features
    numerical_df = df.select_dtypes(include=np.number)
    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Academic Features')
    plt.show()

    # Report highest correlation with target
    target_corr = correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    print("\n[EDA Finding] Correlation with Final Year GPA:")
    print(target_corr)


def save_trained_model(model, file_path):
    """
    Task I6: Serializes and saves the trained model to a file using joblib.
    """
    # Ensure the directory exists before saving
    model_dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"[INFO] Created directory: {model_dir}")
        
    try:
        joblib.dump(model, file_path)
        print(f"\n[Task I6 Success] Trained model saved successfully to: {file_path}")
    except Exception as e:
        print(f"[Task I6 ERROR] Could not save model. Error: {e}")


def train_and_evaluate_model(df):
    """
    Task I4: Splits data, trains a Random Forest Regressor, and evaluates it (Task I5 start).
    """
    print("\n--- Task I4: Model Building and Training (Regression) ---")

    # 1. Prepare Features (X) and Target (y)
    # Exclude StudentID and Enrollment_Status since they are identifiers/filters
    X = df.drop(['StudentID', 'Enrollment_Status', TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    
    print(f"Features used for training: {list(X.columns)}")

    # 2. Split the Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Data Split: Training samples = {X_train.shape[0]}, Testing samples = {X_test.shape[0]}")

    # 3. Initialize and Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    print("\nTraining Random Forest Regressor (100 trees)...")
    model.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    # 4. Evaluate the Model (Task I5 - initial assessment)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Initial Model Performance (on Test Set - Task I5) ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} (This is the average error in GPA points)")
    print(f"R-squared (R2 Score): {r2:.4f} (Closer to 1.0 is better)")
    
    # 5. Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n[Finding] Top Feature Importances:")
    print(feature_importances.sort_values(ascending=False).head())
    
    return model, MODEL_FILE

# --- Main Execution ---
if __name__ == "__main__":
    
    # Load the cleaned data
    df_data = load_data(FILE_NAME)
    
    if df_data is not None and not df_data.empty:
        
        # Step 1: Perform EDA (Task I3)
        perform_eda(df_data)
        
        # Step 2: Train Model (Task I4) & Initial Evaluation (Task I5)
        trained_model, model_filepath = train_and_evaluate_model(df_data)
        
        # Step 3: Save the Model (Task I6)
        # We need to train the model on ALL available data (X and y combined) for final saving
        X_final = df_data.drop(['StudentID', 'Enrollment_Status', TARGET_COLUMN], axis=1)
        y_final = df_data[TARGET_COLUMN]
        
        final_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        final_model.fit(X_final, y_final)
        
        save_trained_model(final_model, model_filepath)
        
        print("\n--- Implementation Phase Tasks Complete (I3, I4, I6) ---")
        print("Next up is Task I5: Comprehensive Results and Evaluation (Chapter 5 Documentation).")
    else:
        print("\nModeling pipeline terminated due to data loading error.")
