"""
Model Pipeline for GPA Predictor Project
Combined with data loading for complete workflow.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import pickle
import os
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== DATA LOADER ====================

class GPADataLoader:
    """Data loader class for GPA prediction dataset."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.attributes_to_remove = ['Attendance_Rate', 'Enrollment_Status']
        self.identifier_columns = ['StudentID']
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load data from specified file path."""
        if data_path is not None:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("No data path provided")
            
        logger.info(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Successfully loaded data with shape: {self.data.shape}")
        return self.data
    
    def remove_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove specified attributes from the dataset."""
        existing_attributes = [attr for attr in self.attributes_to_remove if attr in df.columns]
        if existing_attributes:
            logger.info(f"Removing attributes: {existing_attributes}")
            return df.drop(columns=existing_attributes, errors='ignore')
        return df.copy()
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> pd.DataFrame:
        """Main preprocessing pipeline."""
        # Remove specified attributes
        df_processed = self.remove_attributes(df)
        
        # Handle missing values
        for column in df_processed.columns:
            if df_processed[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[column]):
                    fill_value = df_processed[column].mean()
                else:
                    mode_vals = df_processed[column].mode()
                    fill_value = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                df_processed[column] = df_processed[column].fillna(fill_value)
        
        # Validate target column
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")
        return df_processed
    
    def get_feature_target_split(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> Tuple[pd.DataFrame, pd.Series, list]:
        """Split data into features and target."""
        # Remove target column and identifier columns
        columns_to_remove = [target_column] + [col for col in self.identifier_columns if col in df.columns]
        X = df.drop(columns=columns_to_remove)
        y = df[target_column]
        feature_names = X.columns.tolist()
        
        logger.info(f"Feature-target split: X shape: {X.shape}, y shape: {y.shape}")
        return X, y, feature_names

def load_dataset(file_path: str, target_column: str = 'Final_Year_GPA') -> Tuple[pd.DataFrame, pd.Series, list]:
    """Utility function to quickly load and preprocess dataset."""
    loader = GPADataLoader(file_path)
    df = loader.load_data()
    df_processed = loader.preprocess_data(df, target_column=target_column)
    return loader.get_feature_target_split(df_processed, target_column)

# ==================== MISSING VALUE HANDLER ====================

class MissingValueHandler:
    """Missing value handler for GPA prediction data."""
    
    def __init__(self):
        self.imputation_values = {}
        
    def handle_missing_values(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> pd.DataFrame:
        """Handle missing values for small dataset."""
        df_processed = df.copy()
        
        if df.isnull().sum().sum() > 0:
            logger.info("Applying imputation...")
            
            # Drop rows with missing target
            if target_column in df_processed.columns:
                df_processed = df_processed.dropna(subset=[target_column])
            
            # Simple imputation
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        fill_value = df_processed[col].mean()
                        strategy = 'mean'
                    else:
                        fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                        strategy = 'mode'
                    
                    df_processed[col] = df_processed[col].fillna(fill_value)
                    self.imputation_values[col] = {'strategy': strategy, 'value': fill_value}
        
        return df_processed.dropna()

# ==================== MODEL PIPELINE ====================

class GPAModelPipeline:
    """Main pipeline for GPA prediction with small dataset optimizations."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.missing_handler = MissingValueHandler()
        self.feature_names = None
        
        # Available models for small dataset
        self.model_dict = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=random_state),
            'random_forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state)
        }

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data with missing value handling."""
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            X = self.missing_handler.handle_missing_values(X)
        
        # Scale features
        if fit:
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        else:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        # Align y with X
        if y is not None and len(y) != len(X_scaled):
            common_idx = X_scaled.index.intersection(y.index)
            X_scaled = X_scaled.loc[common_idx]
            y = y.loc[common_idx]
        
        return X_scaled, y

    def train_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> Dict[str, Any]:
        """Train model with simplified workflow."""
        self.feature_names = X.columns.tolist()
        
        # Split and preprocess
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        X_train_processed, y_train = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train
        self.model = self.model_dict.get(self.model_type, self.model_dict['random_forest'])
        self.model.fit(X_train_processed, y_train)
        
        # Evaluate
        train_pred = np.clip(self.model.predict(X_train_processed), 0.0, 4.0)
        test_pred = np.clip(self.model.predict(self.preprocess_data(X_test, fit=False)[0]), 0.0, 4.0)
        
        train_metrics = {
            'mae': mean_absolute_error(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred),
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'r2': r2_score(y_test, test_pred),
        }
        
        # Cross-validation
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        r2_scores = cross_val_score(self.model, X_processed, y_processed, cv=3, scoring='r2')
        
        return {
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean_r2': r2_scores.mean(),
            'feature_names': self.feature_names
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X_processed, _ = self.preprocess_data(X, fit=False)
        return np.clip(self.model.predict(X_processed), 0.0, 4.0)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'missing_handler': self.missing_handler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        for key, value in model_data.items():
            setattr(self, key, value)
        logger.info(f"Model loaded from: {filepath}")

# ==================== MODEL COMPARISON ====================

def compare_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> pd.DataFrame:
    """Compare multiple models."""
    models = ['linear_regression', 'ridge', 'random_forest', 'gradient_boosting']
    results = []
    
    for model_type in models:
        try:
            pipeline = GPAModelPipeline(model_type=model_type)
            results_dict = pipeline.train_model(X, y, test_size=test_size)
            
            results.append({
                'model': model_type,
                'test_r2': results_dict['test_metrics']['r2'],
                'test_rmse': results_dict['test_metrics']['rmse'],
                'test_mae': results_dict['test_metrics']['mae'],
                'cv_mean_r2': results_dict['cv_mean_r2']
            })
            logger.info(f"Trained {model_type}: R²={results[-1]['test_r2']:.3f}")
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
    
    if not results:
        raise ValueError("No models were successfully trained")
    
    return pd.DataFrame(results).sort_values('test_r2', ascending=False)

# ==================== MAIN WORKFLOW ====================

def run_full_pipeline(data_file: str = 'data/cleaned_student_performance.csv'):
    """Complete workflow from data loading to model training."""
    logger.info("Starting full GPA prediction pipeline...")
    
    try:
        # Step 1: Load and preprocess data
        logger.info(f"Loading data from {data_file}")
        X, y, features = load_dataset(data_file)
        
        print(f"\nData Summary:")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]} ({', '.join(features[:3])}...)")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"  Target mean: {y.mean():.2f}")
        
        # Step 2: Compare models
        print("\nComparing models...")
        comparison_results = compare_models(X, y)
        print("\nModel Comparison Results:")
        print(comparison_results.to_string(index=False))
        
        # Step 3: Train best model
        best_model = comparison_results.iloc[0]['model']
        print(f"\nTraining best model: {best_model}")
        
        pipeline = GPAModelPipeline(model_type=best_model)
        results = pipeline.train_model(X, y)
        
        # Step 4: Show feature importance
        importance_df = pipeline.get_feature_importance()
        if importance_df is not None:
            print("\nTop 5 Important Features:")
            print(importance_df.head().to_string(index=False))
        
        # Step 5: Save model
        pipeline.save_model('models/best_gpa_model.pkl')
        
        # Step 6: Test predictions
        print(f"\nSample Predictions (first 5 students):")
        sample_preds = pipeline.predict(X.head(5))
        actual_values = y.head(5).values
        
        for i, (pred, actual) in enumerate(zip(sample_preds, actual_values)):
            print(f"  Student {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Diff={abs(pred-actual):.2f}")
        
        # Step 7: Final metrics
        print(f"\nFinal Model Performance:")
        print(f"  Test R²: {results['test_metrics']['r2']:.3f}")
        print(f"  Test RMSE: {results['test_metrics']['rmse']:.3f}")
        print(f"  Test MAE: {results['test_metrics']['mae']:.3f}")
        print(f"  Cross-validation R²: {results['cv_mean_r2']:.3f}")
        
        logger.info("Pipeline completed successfully!")
        return pipeline, results
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        print(f"\nError: Could not find data file at {data_file}")
        print("Please ensure the file exists or update the path.")
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        print(f"\nError: {str(e)}")

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Run the complete pipeline
    run_full_pipeline()
    
    # Alternatively, use custom data file:
    # run_full_pipeline('your_data.csv')