"""
Model Pipeline for GPA Predictor Project
Updated with comprehensive missing value handling.
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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingValueHandler:
    """
    Comprehensive missing value handling for GPA prediction data.
    """
    
    def __init__(self, strategy: str = 'contextual'):
        """
        Initialize missing value handler.
        
        Args:
            strategy (str): Imputation strategy
                Options: 'mean', 'median', 'knn', 'mice', 'contextual', 'drop'
        """
        self.strategy = strategy
        self.imputation_values = {}
        self.imputer = None
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing value patterns in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Missing value analysis report
        """
        logger.info("Analyzing missing value patterns...")
        
        missing_summary = {
            'total_records': len(df),
            'total_cells': df.size,
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'missing_by_column': df.isnull().sum().to_dict(),
            'percentage_by_column': (df.isnull().sum() / len(df) * 100).to_dict(),
            'columns_with_missing': [col for col in df.columns if df[col].isnull().any()],
            'complete_cases': df.dropna().shape[0],
            'complete_cases_percentage': (df.dropna().shape[0] / len(df)) * 100
        }
        
        # Log missing analysis
        logger.info(f"Missing cells: {missing_summary['missing_cells']} "
                   f"({missing_summary['missing_percentage']:.2f}%)")
        logger.info(f"Complete cases: {missing_summary['complete_cases']} "
                   f"({missing_summary['complete_cases_percentage']:.2f}%)")
        
        if missing_summary['columns_with_missing']:
            logger.info("Columns with missing values:")
            for col in missing_summary['columns_with_missing']:
                missing_pct = missing_summary['percentage_by_column'][col]
                logger.info(f"  - {col}: {missing_summary['missing_by_column'][col]} "
                           f"({missing_pct:.2f}%)")
        
        return missing_summary
    
    def select_best_strategy(self, df: pd.DataFrame, target_column: str = 'Final_Year_GPA') -> Dict[str, str]:
        """
        Select optimal imputation strategy for each column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target variable column name
            
        Returns:
            Dict: Column-specific imputation strategies
        """
        strategies = {}
        
        for column in df.columns:
            if not df[column].isnull().any():
                strategies[column] = 'none'
                continue
            
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            
            # Strategy selection rules
            if column == target_column:
                strategies[column] = 'drop_rows'  # Cannot impute target
                
            elif missing_pct > 40:  # Too much missing data
                strategies[column] = 'drop_column' if column != 'StudentID' else 'drop_rows'
                
            elif column == 'StudentID':
                strategies[column] = 'drop_rows'  # Essential identifier
                
            elif column == 'Gender_F':
                strategies[column] = 'mode'  # Categorical variable
                
            elif 'GPA' in column:
                # GPA-specific contextual imputation
                strategies[column] = 'contextual_gpa'
                
            elif df[column].dtype in ['int64', 'float64']:
                # Numerical columns
                if missing_pct < 5:
                    strategies[column] = 'mean'  # Small amount of missing
                elif abs(df[column].skew()) > 1:
                    strategies[column] = 'median'  # Skewed distribution
                else:
                    strategies[column] = 'knn'  # Moderate missing, use KNN
                    
            else:  # Categorical columns
                strategies[column] = 'mode'
        
        logger.info(f"Selected strategies: {strategies}")
        return strategies
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategies: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Handle missing values based on selected strategies.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategies (Dict): Column-specific strategies. If None, auto-select.
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {self.strategy}")
        
        df_processed = df.copy()
        
        # Auto-select strategies if not provided
        if strategies is None:
            strategies = self.select_best_strategy(df_processed)
        
        # Apply strategies
        for column, strategy in strategies.items():
            if column not in df_processed.columns:
                continue
                
            if not df_processed[column].isnull().any():
                continue
            
            logger.info(f"Processing {column} with {strategy} strategy")
            
            if strategy == 'drop_rows':
                df_processed = df_processed.dropna(subset=[column])
                
            elif strategy == 'drop_column':
                df_processed = df_processed.drop(columns=[column])
                
            elif strategy == 'mean':
                fill_value = df_processed[column].mean()
                df_processed[column] = df_processed[column].fillna(fill_value)
                self.imputation_values[column] = {'strategy': 'mean', 'value': fill_value}
                
            elif strategy == 'median':
                fill_value = df_processed[column].median()
                df_processed[column] = df_processed[column].fillna(fill_value)
                self.imputation_values[column] = {'strategy': 'median', 'value': fill_value}
                
            elif strategy == 'mode':
                mode_vals = df_processed[column].mode()
                fill_value = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                df_processed[column] = df_processed[column].fillna(fill_value)
                self.imputation_values[column] = {'strategy': 'mode', 'value': fill_value}
                
            elif strategy == 'contextual_gpa':
                fill_value = self._contextual_gpa_imputation(df_processed, column)
                df_processed[column] = df_processed[column].fillna(fill_value)
                self.imputation_values[column] = {'strategy': 'contextual_gpa', 'value': fill_value}
                
            elif strategy == 'knn':
                df_processed = self._knn_imputation(df_processed, column)
                
            elif strategy == 'mice':
                df_processed = self._mice_imputation(df_processed, column)
        
        # Handle remaining missing values with global strategy
        if self.strategy == 'global':
            df_processed = self._global_imputation(df_processed)
        
        # Final check
        remaining_missing = df_processed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Still {remaining_missing} missing values after imputation")
            # Drop any remaining missing values
            df_processed = df_processed.dropna()
        
        logger.info(f"Missing value handling complete. Final shape: {df_processed.shape}")
        return df_processed
    
    def _contextual_gpa_imputation(self, df: pd.DataFrame, gpa_column: str) -> float:
        """
        Contextual imputation for GPA columns using academic progression.
        
        Args:
            df (pd.DataFrame): Input dataframe
            gpa_column (str): GPA column name
            
        Returns:
            float: Imputation value
        """
        # Get all GPA columns
        gpa_columns = [col for col in df.columns if 'GPA' in col and col != 'Final_Year_GPA']
        gpa_columns.sort()  # Sort by year
        
        if gpa_column not in gpa_columns:
            # If not a standard GPA column, use mean
            return df[gpa_column].mean()
        
        # Find position in academic progression
        col_index = gpa_columns.index(gpa_column)
        
        # Try to use neighboring years
        if col_index > 0:
            prev_col = gpa_columns[col_index - 1]
            if not df[prev_col].isnull().all():
                # Use previous year's GPA (assuming improvement)
                return df[prev_col].mean() * 1.05
        
        if col_index < len(gpa_columns) - 1:
            next_col = gpa_columns[col_index + 1]
            if not df[next_col].isnull().all():
                # Use next year's GPA (assuming it was worse)
                return df[next_col].mean() * 0.95
        
        # Fallback: average of all available GPA columns
        available_gpas = [df[col].mean() for col in gpa_columns 
                         if col != gpa_column and not df[col].isnull().all()]
        
        if available_gpas:
            return np.mean(available_gpas)
        
        # Ultimate fallback: institutional average
        return 3.0
    
    def _knn_imputation(self, df: pd.DataFrame, target_column: str, k: int = 5) -> pd.DataFrame:
        """
        K-Nearest Neighbors imputation for a specific column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Column to impute
            k (int): Number of neighbors
            
        Returns:
            pd.DataFrame: Imputed dataframe
        """
        from sklearn.impute import KNNImputer
        
        # Select only numerical columns for KNN
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column not in numeric_cols:
            logger.warning(f"Cannot apply KNN to non-numeric column {target_column}")
            return df
        
        # Create KNN imputer
        imputer = KNNImputer(n_neighbors=k)
        
        # Fit and transform
        imputed_data = imputer.fit_transform(df[numeric_cols])
        
        # Update dataframe
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputed_data
        
        return df_imputed
    
    def _mice_imputation(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """
        Multiple Imputation by Chained Equations (MICE).
        
        Args:
            df (pd.DataFrame): Input dataframe
            max_iter (int): Maximum iterations
            
        Returns:
            pd.DataFrame: Imputed dataframe
        """
        # Select only numerical columns for MICE
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numerical columns for MICE imputation")
            return df
        
        # Create MICE imputer
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=42,
            initial_strategy='mean'
        )
        
        # Fit and transform
        imputed_data = imputer.fit_transform(df[numeric_cols])
        
        # Update dataframe
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputed_data
        
        return df_imputed
    
    def _global_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply global imputation strategy to all missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Imputed dataframe
        """
        df_imputed = df.copy()
        
        if self.strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif self.strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif self.strategy == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            return df_imputed
        
        # Apply to numerical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputed_numeric = imputer.fit_transform(df_imputed[numeric_cols])
            df_imputed[numeric_cols] = imputed_numeric
        
        return df_imputed
    
    def get_imputation_report(self) -> Dict[str, Any]:
        """
        Generate imputation report.
        
        Returns:
            Dict: Imputation report
        """
        return {
            'strategy': self.strategy,
            'imputation_values': self.imputation_values,
            'summary': f"Applied {self.strategy} strategy to {len(self.imputation_values)} columns"
        }

class GPAModelPipeline:
    """
    Main pipeline for GPA prediction model training and evaluation.
    Updated with comprehensive missing value handling.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42,
                 missing_strategy: str = 'contextual'):
        """
        Initialize the model pipeline.
        
        Args:
            model_type (str): Type of model to use
            random_state (int): Random state for reproducibility
            missing_strategy (str): Strategy for handling missing values
        """
        self.model_type = model_type
        self.random_state = random_state
        self.missing_strategy = missing_strategy
        self.model = None
        self.scaler = StandardScaler()
        self.missing_handler = MissingValueHandler(strategy=missing_strategy)
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
        
        logger.info(f"Initialized GPA Model Pipeline with model: {model_type}, "
                   f"missing strategy: {missing_strategy}")
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                       fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Comprehensive preprocessing including missing value handling.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target (optional)
            fit (bool): Whether to fit preprocessing objects
            
        Returns:
            Tuple: Preprocessed (X, y)
        """
        logger.info("Starting comprehensive data preprocessing...")
        
        X_processed = X.copy()
        
        if y is not None:
            y_processed = y.copy()
        else:
            y_processed = None
        
        # Step 1: Analyze missing values
        missing_report = self.missing_handler.analyze_missing_patterns(X_processed)
        
        # Step 2: Handle missing values in features
        if missing_report['missing_cells'] > 0:
            logger.info(f"Handling {missing_report['missing_cells']} missing values...")
            X_processed = self.missing_handler.handle_missing_values(X_processed)
        
        # Step 3: Ensure all data is numeric
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
        
        # Step 4: Scale numerical features
        if fit:
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns
            )
        else:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns
            )
        
        # Step 5: Align y with X (if rows were dropped during imputation)
        if y_processed is not None and len(y_processed) != len(X_processed):
            # This shouldn't happen if we handle missing values properly
            # But just in case, take the intersection
            logger.warning(f"Length mismatch between X ({len(X_processed)}) and y ({len(y_processed)})")
            min_len = min(len(X_processed), len(y_processed))
            X_processed = X_processed.iloc[:min_len]
            y_processed = y_processed.iloc[:min_len]
        
        logger.info(f"Preprocessing completed. Final shape: {X_processed.shape}")
        return X_processed, y_processed
    
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
            logger.warning(f"Some predictions were outside valid GPA range "
                          f"[{self.min_gpa}, {self.max_gpa}]. Capping applied.")
            logger.info(f"Original range: {predictions.min():.3f} - {predictions.max():.3f}")
            logger.info(f"Capped range: {capped_predictions.min():.3f} - "
                       f"{capped_predictions.max():.3f}")
        
        return capped_predictions
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                            test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets with missing value consideration.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test size: {test_size}")
        
        # First handle missing values in the combined dataset
        X_combined = X.copy()
        y_combined = y.copy()
        
        # Check for missing values before split
        if X_combined.isnull().any().any() or y_combined.isnull().any():
            logger.info("Missing values detected before train/test split")
            # Handle missing values
            X_combined = self.missing_handler.handle_missing_values(X_combined)
            # Align y with X (rows might have been dropped)
            y_combined = y_combined.loc[X_combined.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, tune_hyperparams: bool = True,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the GPA prediction model with missing value handling.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            tune_hyperparams (bool): Whether to tune hyperparameters
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results and metrics
        """
        logger.info("Starting model training with missing value handling...")
        
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        self.target_name = y.name if hasattr(y, 'name') else 'Final_Year_GPA'
        
        logger.info(f"Features to be used: {self.feature_names}")
        
        # Get imputation report
        imputation_report = self.missing_handler.get_imputation_report()
        logger.info(f"Missing value handling: {imputation_report['summary']}")
        
        # Split data (handles missing values internally)
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y, test_size)
        
        # Preprocess training features
        X_train_processed, y_train = self.preprocess_data(X_train, y_train, fit=True)
        
        # Train model with or without hyperparameter tuning
        if tune_hyperparams and self.model_type in self.param_grids:
            self.model = self._tune_hyperparameters(X_train_processed, y_train, cv=cv_folds)
            # Retrain with best parameters on full training set
            self.model.fit(X_train_processed.values, y_train)
        else:
            self.model = self.model_dict[self.model_type]
            self.model.fit(X_train_processed.values, y_train)
        
        # Evaluate on training set
        train_predictions_raw = self.model.predict(X_train_processed.values)
        train_predictions = self.cap_predictions(train_predictions_raw)
        self.train_metrics = self.calculate_metrics(y_train, train_predictions, 'train')
        
        # Evaluate on test set
        X_test_processed, y_test = self.preprocess_data(X_test, y_test, fit=False)
        test_predictions_raw = self.model.predict(X_test_processed.values)
        test_predictions = self.cap_predictions(test_predictions_raw)
        self.test_metrics = self.calculate_metrics(y_test, test_predictions, 'test')
        
        # Perform cross-validation
        self.cv_scores = self.cross_validate_model(X, y, cv=cv_folds)
        
        # Prepare results
        results = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'missing_value_strategy': self.missing_strategy,
            'imputation_report': imputation_report,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'cv_scores': self.cv_scores,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }
        
        logger.info("Model training completed successfully")
        return results
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
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
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.model_dict[self.model_type],
            param_grid=self.param_grids[self.model_type],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train.values, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         set_name: str) -> Dict[str, float]:
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
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict[str, Any]:
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
        
        # Handle missing values before CV
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Calculate cross-validation scores
        cv_scores = {
            'neg_mse': cross_val_score(self.model, X_processed.values, y_processed, 
                                      cv=cv, scoring='neg_mean_squared_error'),
            'r2': cross_val_score(self.model, X_processed.values, y_processed, 
                                 cv=cv, scoring='r2'),
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
        
        # Handle missing values in new data
        if X.isnull().any().any():
            logger.info("Missing values detected in prediction data")
            X_processed = self.missing_handler.handle_missing_values(X)
        else:
            X_processed = X.copy()
        
        # Preprocess (using fitted scaler)
        X_processed, _ = self.preprocess_data(X_processed, fit=False)
        
        # Make predictions
        predictions_raw = self.model.predict(X_processed.values)
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
            'missing_handler': self.missing_handler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'missing_strategy': self.missing_strategy,
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
        self.missing_handler = model_data.get('missing_handler', 
                                            MissingValueHandler())
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.missing_strategy = model_data.get('missing_strategy', 'contextual')
        self.train_metrics = model_data.get('train_metrics', {})
        self.test_metrics = model_data.get('test_metrics', {})
        self.cv_scores = model_data.get('cv_scores', {})
        self.min_gpa = model_data.get('min_gpa', 0.0)
        self.max_gpa = model_data.get('max_gpa', 4.0)
        
        logger.info(f"Model loaded from: {filepath}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Missing value strategy: {self.missing_strategy}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"GPA range: [{self.min_gpa}, {self.max_gpa}]")

# Model comparison utility
def compare_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                  models: List[str] = None, missing_strategy: str = 'contextual') -> pd.DataFrame:
    """
    Compare multiple models and return performance metrics.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Test set proportion
        models (List[str]): List of models to compare
        missing_strategy (str): Strategy for handling missing values
        
    Returns:
        pd.DataFrame: Comparison results
    """
    if models is None:
        models = ['linear_regression', 'ridge', 'random_forest', 'gradient_boosting']
    
    results = []
    
    for model_type in models:
        logger.info(f"Training and evaluating {model_type} with {missing_strategy} "
                   f"missing value strategy...")
        
        try:
            pipeline = GPAModelPipeline(model_type=model_type, 
                                       missing_strategy=missing_strategy)
            results_dict = pipeline.train_model(X, y, test_size=test_size, 
                                               tune_hyperparams=True)
            
            result = {
                'model': model_type,
                'missing_strategy': missing_strategy,
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
        
        # Check for missing values
        missing_total = X.isnull().sum().sum() + y.isnull().sum()
        if missing_total > 0:
            print(f"Missing values detected: {missing_total} total")
        
        # Compare multiple models with missing value handling
        print("\nComparing models with contextual missing value handling:")
        comparison_results = compare_models(X, y, missing_strategy='contextual')
        print(comparison_results.to_string(index=False))
        
        # Train best model
        best_model_type = comparison_results.iloc[0]['model']
        print(f"\nTraining best model: {best_model_type}")
        
        pipeline = GPAModelPipeline(model_type=best_model_type, 
                                  missing_strategy='contextual')
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
            print(f"Student {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, "
                  f"Difference={abs(pred-actual):.2f}")
        
        # Test prediction capping
        print(f"\nTesting prediction capping:")
        print(f"GPA range enforced: [{pipeline.min_gpa}, {pipeline.max_gpa}]")
        
    except Exception as e:
        logger.error(f"Error in model pipeline example: {str(e)}")
        import traceback
        traceback.print_exc()
        
# Add these imports at the top
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Add this function after training
def generate_visualizations(pipeline, X_test, y_test, X_train, y_train):
    """
    Generate comprehensive visualizations
    """
    predictions = pipeline.predict(X_test)
    
    # Create visualization directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual GPA')
    plt.ylabel('Predicted GPA')
    plt.title('Actual vs Predicted GPA')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance Plot
    importance_df = pipeline.get_feature_importance()
    if importance_df is not None:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Error Distribution Histogram
    errors = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=errors.mean(), color='red', linestyle='--', label=f'Mean Error: {errors.mean():.3f}')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Performance Comparison (if comparing multiple models)
    print("Visualizations saved to 'visualizations/' directory")
    return ['actual_vs_predicted.png', 'feature_importance.png', 'error_distribution.png']

