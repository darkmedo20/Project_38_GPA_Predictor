"""
Model Pipeline for GPA Predictor Project
Updated for the actual student performance dataset.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import pickle
import json
import os
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPAModelPipeline:
    """
    Main pipeline for GPA prediction model training and evaluation.
    Updated for student performance dataset.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the model pipeline.
        
        Args:
            model_type (str): Type of model to use
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.train_metrics = {}
        self.test_metrics = {}
        self.cv_scores = {}
        
        # GPA constraints
        self.min_gpa = 0.0
        self.max_gpa = 4.0
        
        # Available models
        self.model_dict = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=random_state),
            'lasso': Lasso(random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
            'decision_tree': DecisionTreeRegressor(random_state=random_state),
            'svr': SVR()
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'svr': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
        
        logger.info(f"Initialized GPA Model Pipeline with model: {model_type}")
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Preprocess features for the student performance dataset.
        
        Args:
            X (pd.DataFrame): Input features
            fit (bool): Whether to fit scaler or use existing one
            
        Returns:
            np.ndarray: Preprocessed features
        """
        logger.info("Preprocessing features...")
        
        # Create a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Ensure all data is numeric
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                logger.warning(f"Column {col} is not numeric. Attempting to convert...")
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                    # Fill any NaN values created during conversion
                    if X_processed[col].isnull().any():
                        X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
                except Exception as e:
                    logger.error(f"Could not convert column {col} to numeric: {e}")
                    raise
        
        # Scale numerical features
        if fit:
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        logger.info(f"Preprocessed features shape: {X_processed.shape}")
        return X_processed
    
    def cap_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Cap predictions to valid GPA range (0.0 - 4.0).
        
        Args:
            predictions (np.ndarray): Raw predictions from model
            
        Returns:
            np.ndarray: Capped predictions within valid range
        """
        capped_predictions = np.clip(predictions, self.min_gpa, self.max_gpa)
        
        # Log if any predictions were capped
        out_of_bounds = np.any((predictions < self.min_gpa) | (predictions > self.max_gpa))
        if out_of_bounds:
            logger.warning(f"Some predictions were outside valid GPA range [{self.min_gpa}, {self.max_gpa}]. Capping applied.")
            logger.info(f"Original range: {predictions.min():.3f} - {predictions.max():.3f}")
            logger.info(f"Capped range: {capped_predictions.min():.3f} - {capped_predictions.max():.3f}")
        
        return capped_predictions
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                            test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test size: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           cv: int = 5) -> Any:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv (int): Number of cross-validation folds
            
        Returns:
            Best estimator model
        """
        if self.model_type not in self.param_grids:
            logger.info(f"No hyperparameter tuning defined for {self.model_type}")
            return self.model_dict[self.model_type]
        
        logger.info(f"Tuning hyperparameters for {self.model_type}...")
        
        # Preprocess features for tuning
        X_train_processed = self.preprocess_features(X_train, fit=True)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.model_dict[self.model_type],
            param_grid=self.param_grids[self.model_type],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_processed, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, tune_hyperparams: bool = True,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the GPA prediction model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            tune_hyperparams (bool): Whether to tune hyperparameters
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results and metrics
        """
        logger.info("Starting model training...")
        
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        self.target_name = y.name if hasattr(y, 'name') else 'Final_Year_GPA'
        
        logger.info(f"Features to be used: {self.feature_names}")
        logger.info(f"Data types: {X.dtypes.to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y, test_size)
        
        # Preprocess training features
        X_train_processed = self.preprocess_features(X_train, fit=True)
        
        # Train model with or without hyperparameter tuning
        if tune_hyperparams and self.model_type in self.param_grids:
            self.model = self.tune_hyperparameters(X_train, y_train, cv=cv_folds)
            # Retrain with best parameters on full training set
            self.model.fit(X_train_processed, y_train)
        else:
            self.model = self.model_dict[self.model_type]
            self.model.fit(X_train_processed, y_train)
        
        # Evaluate on training set
        train_predictions_raw = self.model.predict(X_train_processed)
        train_predictions = self.cap_predictions(train_predictions_raw)
        self.train_metrics = self.calculate_metrics(y_train, train_predictions, 'train')
        
        # Evaluate on test set
        X_test_processed = self.preprocess_features(X_test, fit=False)
        test_predictions_raw = self.model.predict(X_test_processed)
        test_predictions = self.cap_predictions(test_predictions_raw)
        self.test_metrics = self.calculate_metrics(y_test, test_predictions, 'test')
        
        # Perform cross-validation
        self.cv_scores = self.cross_validate_model(X, y, cv=cv_folds)
        
        # Prepare results
        results = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'cv_scores': self.cv_scores,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }
        
        logger.info("Model training completed successfully")
        return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, set_name: str) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            set_name (str): Name of the dataset (train/test)
            
        Returns:
            Dict: Calculated metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }
        
        logger.info(f"{set_name.capitalize()} Metrics - "
                   f"MAE: {metrics['mae']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, "
                   f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the entire dataset.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Calculate cross-validation scores
        cv_scores = {
            'neg_mse': cross_val_score(self.model, X_processed, y, cv=cv, scoring='neg_mean_squared_error'),
            'r2': cross_val_score(self.model, X_processed, y, cv=cv, scoring='r2'),
        }
        
        cv_results = {
            'mean_rmse': np.sqrt(-cv_scores['neg_mse'].mean()),
            'std_rmse': np.sqrt(-cv_scores['neg_mse'].std()),
            'mean_r2': cv_scores['r2'].mean(),
            'std_r2': cv_scores['r2'].std(),
        }
        
        logger.info(f"CV Results - Mean RMSE: {cv_results['mean_rmse']:.4f}, "
                   f"Mean R²: {cv_results['mean_r2']:.4f}")
        
        return cv_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predictions (capped to valid GPA range)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X_processed = self.preprocess_features(X, fit=False)
        predictions_raw = self.model.predict(X_processed)
        predictions = self.cap_predictions(predictions_raw)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self.model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            
            return importance_df
        else:
            logger.warning("Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model and preprocessing objects.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'cv_scores': self.cv_scores,
            'min_gpa': self.min_gpa,
            'max_gpa': self.max_gpa,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model and preprocessing objects.
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.train_metrics = model_data.get('train_metrics', {})
        self.test_metrics = model_data.get('test_metrics', {})
        self.cv_scores = model_data.get('cv_scores', {})
        self.min_gpa = model_data.get('min_gpa', 0.0)
        self.max_gpa = model_data.get('max_gpa', 4.0)
        
        logger.info(f"Model loaded from: {filepath}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"GPA range: [{self.min_gpa}, {self.max_gpa}]")

# Model comparison utility
def compare_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                  models: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple models and return performance metrics.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Test set proportion
        models (List[str]): List of models to compare
        
    Returns:
        pd.DataFrame: Comparison results
    """
    if models is None:
        models = ['linear_regression', 'ridge', 'random_forest', 'gradient_boosting']
    
    results = []
    
    for model_type in models:
        logger.info(f"Training and evaluating {model_type}...")
        
        try:
            pipeline = GPAModelPipeline(model_type=model_type)
            results_dict = pipeline.train_model(X, y, test_size=test_size, tune_hyperparams=True)
            
            result = {
                'model': model_type,
                'train_r2': results_dict['train_metrics']['r2'],
                'test_r2': results_dict['test_metrics']['r2'],
                'test_rmse': results_dict['test_metrics']['rmse'],
                'test_mae': results_dict['test_metrics']['mae'],
                'cv_mean_r2': results_dict['cv_scores']['mean_r2'],
                'cv_mean_rmse': results_dict['cv_scores']['mean_rmse']
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No models were successfully trained")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
    
    logger.info("Model comparison completed")
    return comparison_df

# Example usage
if __name__ == "__main__":
    # Example usage of the model pipeline
    from data_loader import load_dataset
    
    try:
        # Load data
        X, y, features = load_dataset('data/cleaned_student_performance.csv')
        print(f"Features: {features}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"Data types:\n{X.dtypes}")
        
        # Compare multiple models
        comparison_results = compare_models(X, y)
        print("\nModel Comparison Results:")
        print(comparison_results.to_string(index=False))
        
        # Train best model
        best_model_type = comparison_results.iloc[0]['model']
        print(f"\nTraining best model: {best_model_type}")
        
        pipeline = GPAModelPipeline(model_type=best_model_type)
        results = pipeline.train_model(X, y, tune_hyperparams=True)
        
        # Get feature importance
        importance_df = pipeline.get_feature_importance()
        if importance_df is not None:
            print("\nFeature Importance:")
            print(importance_df.to_string(index=False))
        
        # Save model
        pipeline.save_model('models/best_gpa_model.pkl')
        
        # Make sample predictions
        sample_data = X.head(5)
        predictions = pipeline.predict(sample_data)
        actual_values = y.head(5).values
        print(f"\nSample predictions vs actual:")
        for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
            print(f"Student {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Difference={abs(pred-actual):.2f}")
        
        # Test prediction capping
        print(f"\nTesting prediction capping:")
        print(f"GPA range enforced: [{pipeline.min_gpa}, {pipeline.max_gpa}]")
        
    except Exception as e:
        logger.error(f"Error in model pipeline example: {str(e)}")
        import traceback
        traceback.print_exc()