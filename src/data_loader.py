"""
Data Loader Module for GPA Predictor Project
Updated for the actual student performance dataset structure.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPADataLoader:
    """
    Data loader class for GPA prediction dataset.
    Updated for the actual student performance dataset.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        
        # Define attributes to remove (from your requirements)
        self.attributes_to_remove = ['Attendance_Rate', 'Enrollment_Status']
        
        # Define identifier columns (should be excluded from features)
        self.identifier_columns = ['StudentID']
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load data from specified file path.
        
        Args:
            data_path (str): Path to data file. If None, uses self.data_path
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if data_path is not None:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("No data path provided")
            
        logger.info(f"Loading data from: {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data with shape: {self.data.shape}")
            logger.info(f"Columns: {self.data.columns.tolist()}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def remove_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove specified attributes from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with specified attributes removed
        """
        logger.info("Removing specified attributes...")
        
        # Check which attributes exist in the dataset
        existing_attributes = [attr for attr in self.attributes_to_remove if attr in df.columns]
        non_existing = [attr for attr in self.attributes_to_remove if attr not in df.columns]
        
        if existing_attributes:
            logger.info(f"Removing attributes: {existing_attributes}")
            df_cleaned = df.drop(columns=existing_attributes, errors='ignore')
        else:
            logger.info("No specified attributes found to remove")
            df_cleaned = df.copy()
            
        if non_existing:
            logger.warning(f"Attributes not found in dataset: {non_existing}")
            
        logger.info(f"Dataset shape after removal: {df_cleaned.shape}")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        logger.info("Handling missing values...")
        
        initial_shape = df.shape
        missing_counts = df.isnull().sum()
        
        if missing_counts.any():
            logger.info(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
            
            if strategy == 'drop':
                df_cleaned = df.dropna()
                logger.info(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            else:
                for column in df.select_dtypes(include=[np.number]).columns:
                    if df[column].isnull().any():
                        if strategy == 'mean':
                            fill_value = df[column].mean()
                        elif strategy == 'median':
                            fill_value = df[column].median()
                        else:
                            fill_value = 0
                            
                        df[column] = df[column].fillna(fill_value)
                        logger.info(f"Filled missing values in {column} with {strategy}: {fill_value}")
                
                # For categorical columns, fill with mode or 'Unknown'
                for column in df.select_dtypes(include=['object']).columns:
                    if df[column].isnull().any():
                        df[column] = df[column].fillna(df[column].mode()[0] if len(df[column].mode()) > 0 else 'Unknown')
                        logger.info(f"Filled missing categorical values in {column}")
                        
            df_cleaned = df
        else:
            logger.info("No missing values found")
            df_cleaned = df
            
        return df_cleaned
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA', 
                       handle_missing: bool = True, missing_strategy: str = 'mean') -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            handle_missing (bool): Whether to handle missing values
            missing_strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")
        
        # Step 1: Remove specified attributes
        df_processed = self.remove_attributes(df)
        
        # Step 2: Handle missing values if requested
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed, strategy=missing_strategy)
        
        # Step 3: Validate target column exists
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Step 4: Store feature information (exclude identifiers and target)
        self.feature_names = [col for col in df_processed.columns 
                            if col != target_column and col not in self.identifier_columns]
        self.target = target_column
        
        logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target: {self.target}")
        logger.info(f"Identifier columns excluded: {self.identifier_columns}")
        
        return df_processed
    
    def get_feature_target_split(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Split data into features and target.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            target_column (str): Name of the target column
            
        Returns:
            Tuple: (X_features, y_target, feature_names)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Remove target column and identifier columns
        columns_to_remove = [target_column] + self.identifier_columns
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        X = df.drop(columns=columns_to_remove)
        y = df[target_column]
        feature_names = X.columns.tolist()
        
        logger.info(f"Feature-target split completed. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Features used: {feature_names}")
        
        return X, y, feature_names
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Data summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        }
        
        return summary

# Utility functions
def load_dataset(file_path: str, target_column: str = 'Final_Year_GPA') -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Utility function to quickly load and preprocess dataset.
    
    Args:
        file_path (str): Path to data file
        target_column (str): Name of the target column
        
    Returns:
        Tuple: (X_features, y_target, feature_names)
    """
    loader = GPADataLoader(file_path)
    df = loader.load_data()
    df_processed = loader.preprocess_data(df, target_column=target_column)
    X, y, feature_names = loader.get_feature_target_split(df_processed, target_column)
    
    return X, y, feature_names

def validate_dataset(df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> bool:
    """
    Validate dataset for GPA prediction.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        target_column (str): Target column name
        
    Returns:
        bool: True if dataset is valid
    """
    # Check if target column exists
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found")
        return False
    
    # Check if there are enough samples
    if len(df) < 10:
        logger.error("Dataset too small for meaningful analysis")
        return False
    
    # Check if target column has sufficient variance
    if df[target_column].nunique() < 2:
        logger.error("Target column has insufficient variance")
        return False
    
    logger.info("Dataset validation passed")
    return True

# Example usage
if __name__ == "__main__":
    # Example of how to use the data loader
    try:
        # Initialize loader
        loader = GPADataLoader("data/cleaned_student_performance.csv")
        
        # Load data
        df = loader.load_data()
        
        # Preprocess data
        df_processed = loader.preprocess_data(df, target_column='Final_Year_GPA')
        
        # Get feature-target split
        X, y, features = loader.get_feature_target_split(df_processed)
        
        # Print summary
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {features}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"Target mean: {y.mean():.2f}")
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
'''       
# Data head
print("\n--- 3.1 Data Head (First 5 Rows) ---")
print(df.head())

# DataFrame info
print("\n--- 3.2 DataFrame Information (Data Types and Missing Values) ---")
df.info()

# Statistics
print("\n--- 3.3 Descriptive Statistics (Mean, Std, Min, Max, Quartiles) ---")
print(df.describe().T)

'''