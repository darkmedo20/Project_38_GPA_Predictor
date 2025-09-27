# File: data_loader.py
# Purpose: Executes the 'Load and Explore the Data Set' (Task I1) of the Implementation Phase.
# This script loads the cleaned academic data, checks its structure, and performs
# basic statistical summaries before model training begins.

import pandas as pd
import os

def load_and_explore_data(file_name='../data/cleaned_student_performance.csv'):
    """
    Loads the student performance data from a CSV file and performs initial exploration.
    
    Args:
        file_name (str): The name of the CSV file containing the cleaned data.
    """
    
    # 1. Define the File Path
    # We assume the CSV file is located in a 'data' folder relative to this script.
    # Adjust the path if your file structure is different.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'data', file_name)
    
    print(f"--- Attempting to load data from: {file_path} ---")

    try:
        # 2. Load the Data into a DataFrame
        df = pd.read_csv(file_path)
        print("\n[SUCCESS] Data loaded successfully into a Pandas DataFrame.")

        # 3. Display Basic Info and Statistics

        # Display the first few rows (head)
        print("\n--- 3.1 Data Head (First 5 Rows) ---")
        print(df.head())

        # Display structure, data types, and non-null counts (info)
        print("\n--- 3.2 DataFrame Information (Data Types and Missing Values) ---")
        df.info()

        # Display descriptive statistics for numerical columns (describe)
        print("\n--- 3.3 Descriptive Statistics (Mean, Std, Min, Max, Quartiles) ---")
        print(df.describe().T) # .T transposes the output for better readability

        # Display the data shape (rows and columns)
        print(f"\n--- 3.4 Data Shape ---")
        print(f"The dataset contains {df.shape[0]} students (rows) and {df.shape[1]} features (columns).")
        
        # 4. Choose the Target Variable (Required for the next step: Splitting)
        target_column = 'Final_Year_GPA'
        
        if target_column in df.columns:
            print(f"\n[TARGET] Identified the target variable: '{target_column}'")
            # Show distribution of the target variable
            print(f"Target variable mean: {df[target_column].mean():.2f}")
            print(f"Target variable standard deviation: {df[target_column].std():.2f}")
        else:
            print(f"\n[ERROR] Target column '{target_column}' not found. Check column names.")

        print("\n--- Data Loading and Initial Exploration Complete ---")
        
    except FileNotFoundError:
        print(f"\n[ERROR] File not found. Please ensure '{file_name}' is saved in the correct 'data/' folder.")
        print("Expected path: ", file_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred during data loading: {e}")

# Execute the function
if __name__ == "__main__":
    load_and_explore_data()
