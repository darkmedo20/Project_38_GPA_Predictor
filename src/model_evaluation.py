"""
Model Performance Evaluation for GPA Prediction
Regression model evaluation with comprehensive metrics
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive evaluation of GPA prediction model performance.
    """
    
    def __init__(self, model_path: str, test_data_path: str = None):
        """
        Initialize evaluator with model and test data.
        
        Args:
            model_path (str): Path to trained model
            test_data_path (str): Path to test data CSV
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.feature_names = []
        self.scaler = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
    def load_model_and_data(self) -> bool:
        """
        Load model and test data.
        
        Returns:
            bool: True if successful
        """
        try:
            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Try joblib first
            try:
                model_data = joblib.load(self.model_path)
            except:
                # Try pickle with workaround
                class DummyMissingValueHandler:
                    def __init__(self, *args, **kwargs):
                        pass
                import __main__
                __main__.MissingValueHandler = DummyMissingValueHandler
                
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.feature_names = model_data.get('feature_names', [])
            self.scaler = model_data.get('scaler', None)
            
            print(f"✅ Model loaded: {type(self.model).__name__}")
            print(f"   Features: {len(self.feature_names)}")
            
            # Load test data if provided
            if self.test_data_path and os.path.exists(self.test_data_path):
                self._load_test_data()
            else:
                print("⚠️ No test data provided. Using training metrics from model file.")
                # Try to get metrics from saved model
                self.train_metrics = model_data.get('train_metrics', {})
                self.test_metrics = model_data.get('test_metrics', {})
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading: {str(e)}")
            return False
    
    def _load_test_data(self):
        """Load and prepare test data."""
        df = pd.read_csv(self.test_data_path)
        
        # Check if target column exists
        target_col = 'Final_Year_GPA'
        if target_col not in df.columns:
            # Try to guess target column
            possible_targets = ['GPA', 'Final_GPA', 'Final_Year_GPA', 'Target']
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
        
        # Prepare features
        if self.feature_names:
            # Use model's feature names
            X = df[[col for col in self.feature_names if col in df.columns]].copy()
            
            # Add missing features
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        else:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            X = df[[col for col in numeric_cols if col != target_col]].copy()
            self.feature_names = X.columns.tolist()
        
        # Prepare target
        y = df[target_col] if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(0)
        if y is not None:
            y = y.fillna(y.mean())
        
        # Apply scaler if available
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        self.X_test = X
        self.y_test = y
        
        print(f"✅ Test data loaded: {len(X)} samples, {len(X.columns)} features")
        
        # Make predictions
        if y is not None:
            self.y_pred = self.model.predict(X.values)
            print(f"✅ Predictions made on test data")
    
    def calculate_regression_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Returns:
            Dict: All calculated metrics
        """
        if self.y_test is None or self.y_pred is None:
            print("⚠️ No test predictions available. Using saved metrics if available.")
            return self.test_metrics if hasattr(self, 'test_metrics') else {}
        
        y_true = self.y_test.values
        y_pred = self.y_pred
        
        # Cap predictions to valid GPA range (0-4)
        y_pred = np.clip(y_pred, 0.0, 4.0)
        
        metrics = {
            # Basic error metrics
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # as percentage
            
            # Goodness of fit metrics
            'R²': r2_score(y_true, y_pred),
            'Adjusted_R²': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(self.feature_names) - 1),
            'Explained_Variance': explained_variance_score(y_true, y_pred),
            
            # Distribution metrics
            'Max_Error': max_error(y_true, y_pred),
            'Mean_Absolute_Deviation': np.mean(np.abs(y_true - np.mean(y_true))),
            'Median_Absolute_Error': np.median(np.abs(y_true - y_pred)),
            
            # Custom GPA-specific metrics
            'Within_0.1_GPA': np.mean(np.abs(y_true - y_pred) <= 0.1) * 100,
            'Within_0.25_GPA': np.mean(np.abs(y_true - y_pred) <= 0.25) * 100,
            'Within_0.5_GPA': np.mean(np.abs(y_true - y_pred) <= 0.5) * 100,
            
            # Direction accuracy
            'Direction_Accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
            if len(y_true) > 1 else 0,
        }
        
        return metrics
    
    def calculate_classification_metrics(self, threshold: float = 2.0) -> Dict[str, float]:
        """
        Convert regression to classification for academic performance levels.
        
        Args:
            threshold (float): GPA threshold for classification
            
        Returns:
            Dict: Classification metrics
        """
        if self.y_test is None or self.y_pred is None:
            return {}
        
        y_true = self.y_test.values
        y_pred = self.y_pred
        
        # Define classes based on GPA
        # Class 0: GPA < threshold (At Risk)
        # Class 1: GPA >= threshold (Passing)
        y_true_class = (y_true >= threshold).astype(int)
        y_pred_class = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix components
        TP = np.sum((y_true_class == 1) & (y_pred_class == 1))
        TN = np.sum((y_true_class == 0) & (y_pred_class == 0))
        FP = np.sum((y_true_class == 0) & (y_pred_class == 1))
        FN = np.sum((y_true_class == 1) & (y_pred_class == 0))
        
        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (True Negative Rate)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Balanced Accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        metrics = {
            'Threshold_GPA': threshold,
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1_Score': f1_score * 100,
            'Specificity': specificity * 100,
            'Balanced_Accuracy': balanced_accuracy * 100,
            'Confusion_Matrix': {
                'TP': int(TP),
                'TN': int(TN),
                'FP': int(FP),
                'FN': int(FN)
            }
        }
        
        return metrics
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Dict: Complete evaluation report
        """
        report = {
            'model_info': {
                'model_type': str(type(self.model).__name__),
                'features_count': len(self.feature_names),
                'features': self.feature_names
            },
            'data_info': {
                'test_samples': len(self.X_test) if self.X_test is not None else 0,
                'has_target': self.y_test is not None
            },
            'regression_metrics': self.calculate_regression_metrics(),
            'classification_metrics': {
                'threshold_2.0': self.calculate_classification_metrics(threshold=2.0),
                'threshold_2.5': self.calculate_classification_metrics(threshold=2.5),
                'threshold_3.0': self.calculate_classification_metrics(threshold=3.0)
            },
            'error_analysis': self._analyze_errors(),
            'feature_importance': self._get_feature_importance()
        }
        
        return report
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        if self.y_test is None or self.y_pred is None:
            return {}
        
        errors = self.y_test.values - self.y_pred
        abs_errors = np.abs(errors)
        
        analysis = {
            'error_distribution': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors))
            },
            'absolute_error_distribution': {
                'mean': float(np.mean(abs_errors)),
                'std': float(np.std(abs_errors)),
                'median': float(np.median(abs_errors)),
                'q25': float(np.percentile(abs_errors, 25)),
                'q75': float(np.percentile(abs_errors, 75))
            },
            'worst_predictions': self._get_worst_predictions(5)
        }
        
        return analysis
    
    def _get_worst_predictions(self, n: int = 5) -> List[Dict]:
        """Get the n worst predictions."""
        if self.y_test is None or self.y_pred is None:
            return []
        
        errors = np.abs(self.y_test.values - self.y_pred)
        worst_indices = np.argsort(errors)[-n:][::-1]
        
        worst = []
        for idx in worst_indices:
            worst.append({
                'index': int(idx),
                'actual_gpa': float(self.y_test.iloc[idx]),
                'predicted_gpa': float(self.y_pred[idx]),
                'error': float(errors[idx]),
                'error_percentage': float(abs(errors[idx] / self.y_test.iloc[idx] * 100)) if self.y_test.iloc[idx] != 0 else 0
            })
        
        return worst
    
    def _get_feature_importance(self) -> Dict:
        """Extract feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importance))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            importance_dict = dict(zip(self.feature_names, np.abs(coef)))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        else:
            return {'note': 'Feature importance not available for this model type'}
    
    def create_visualizations(self, save_dir: str = "evaluation_plots"):
        """
        Create comprehensive visualizations.
        
        Args:
            save_dir (str): Directory to save plots
        """
        if self.y_test is None or self.y_pred is None:
            print("⚠️ Cannot create visualizations without test data")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        y_true = self.y_test.values
        y_pred = self.y_pred
        errors = y_true - y_pred
        
        # 1. Actual vs Predicted Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
        
        plt.xlabel('Actual GPA', fontsize=12)
        plt.ylabel('Predicted GPA', fontsize=12)
        plt.title('Actual vs Predicted GPA', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Error Distribution Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Residual Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted GPA', fontsize=12)
        plt.ylabel('Residuals (Error)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance Plot (if available)
        importance = self._get_feature_importance()
        if isinstance(importance, dict) and len(importance) > 0 and 'note' not in importance:
            plt.figure(figsize=(12, 6))
            features = list(importance.keys())[:10]  # Top 10 features
            values = list(importance.values())[:10]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
            bars = plt.barh(features, values, color=colors)
            
            plt.xlabel('Importance Score', fontsize=12)
            plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to: {save_dir}/")
    
    def save_report(self, report: Dict, output_path: str = "model_evaluation_report.json"):
        """Save evaluation report to JSON file."""
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_to_serializable(report)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"✅ Report saved to: {output_path}")
        return output_path

def display_evaluation_summary(report: Dict):
    """
    Display evaluation results in a clean, readable format.
    
    Args:
        report (Dict): Evaluation report
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n📋 MODEL INFORMATION")
    print(f"   Model Type: {report['model_info']['model_type']}")
    print(f"   Features Used: {report['model_info']['features_count']}")
    
    print(f"\n📊 REGRESSION METRICS (GPA Prediction Accuracy)")
    reg_metrics = report['regression_metrics']
    
    print(f"\n   Error Metrics:")
    print(f"   • Mean Absolute Error (MAE): {reg_metrics.get('MAE', 0):.3f} GPA points")
    print(f"   • Root Mean Squared Error (RMSE): {reg_metrics.get('RMSE', 0):.3f} GPA points")
    print(f"   • Mean Absolute Percentage Error (MAPE): {reg_metrics.get('MAPE', 0):.1f}%")
    
    print(f"\n   Goodness of Fit:")
    r2 = reg_metrics.get('R²', 0)
    print(f"   • R² Score: {r2:.3f} ({(r2*100):.1f}% variance explained)")
    print(f"   • Adjusted R²: {reg_metrics.get('Adjusted_R²', 0):.3f}")
    print(f"   • Explained Variance: {reg_metrics.get('Explained_Variance', 0):.3f}")
    
    print(f"\n   Practical Accuracy (GPA Scale 0-4):")
    print(f"   • Predictions within 0.1 GPA: {reg_metrics.get('Within_0.1_GPA', 0):.1f}%")
    print(f"   • Predictions within 0.25 GPA: {reg_metrics.get('Within_0.25_GPA', 0):.1f}%")
    print(f"   • Predictions within 0.5 GPA: {reg_metrics.get('Within_0.5_GPA', 0):.1f}%")
    print(f"   • Direction Accuracy: {reg_metrics.get('Direction_Accuracy', 0):.1f}%")
    
    print(f"\n🎯 CLASSIFICATION METRICS (Academic Performance Levels)")
    
    for threshold, metrics in report['classification_metrics'].items():
        if metrics:  # Only display if metrics exist
            print(f"\n   Threshold: GPA ≥ {threshold.split('_')[1]}")
            print(f"   • Accuracy: {metrics.get('Accuracy', 0):.1f}%")
            print(f"   • Precision: {metrics.get('Precision', 0):.1f}%")
            print(f"   • Recall: {metrics.get('Recall', 0):.1f}%")
            print(f"   • F1-Score: {metrics.get('F1_Score', 0):.1f}%")
            
            # Confusion Matrix
            cm = metrics.get('Confusion_Matrix', {})
            if cm:
                print(f"   • Confusion Matrix:")
                print(f"        True Positives: {cm.get('TP', 0)}")
                print(f"        True Negatives: {cm.get('TN', 0)}")
                print(f"        False Positives: {cm.get('FP', 0)}")
                print(f"        False Negatives: {cm.get('FN', 0)}")
    
    print(f"\n📈 PERFORMANCE INTERPRETATION")
    
    r2 = reg_metrics.get('R²', 0)
    rmse = reg_metrics.get('RMSE', 0)
    
    if r2 >= 0.9:
        print("   ✅ EXCELLENT: Model explains over 90% of GPA variance")
    elif r2 >= 0.8:
        print("   ✅ VERY GOOD: Model explains 80-90% of GPA variance")
    elif r2 >= 0.7:
        print("   ✅ GOOD: Model explains 70-80% of GPA variance")
    elif r2 >= 0.6:
        print("   ⚠️  MODERATE: Model explains 60-70% of GPA variance")
    else:
        print("   ❌ POOR: Model explains less than 60% of GPA variance")
    
    if rmse <= 0.15:
        print("   ✅ HIGH PRECISION: Average error ≤ 0.15 GPA points")
    elif rmse <= 0.25:
        print("   ✅ GOOD PRECISION: Average error ≤ 0.25 GPA points")
    elif rmse <= 0.4:
        print("   ⚠️  MODERATE PRECISION: Average error ≤ 0.4 GPA points")
    else:
        print("   ❌ LOW PRECISION: Average error > 0.4 GPA points")
    
    within_25 = reg_metrics.get('Within_0.25_GPA', 0)
    if within_25 >= 90:
        print("   ✅ EXCELLENT PRACTICAL ACCURACY: Over 90% within 0.25 GPA")
    elif within_25 >= 80:
        print("   ✅ GOOD PRACTICAL ACCURACY: 80-90% within 0.25 GPA")
    elif within_25 >= 70:
        print("   ⚠️  MODERATE PRACTICAL ACCURACY: 70-80% within 0.25 GPA")
    else:
        print("   ❌ LOW PRACTICAL ACCURACY: Less than 70% within 0.25 GPA")
    
    print("\n" + "="*80)

def main():
    """
    Main function to run model evaluation.
    """
    print("\n" + "="*80)
    print("GPA PREDICTION MODEL EVALUATION")
    print("="*80)
    
    # Configuration
    MODEL_PATH = "models/best_gpa_model.pkl"  # Update this path
    TEST_DATA_PATH = "data/test_students.csv"  # Optional: Test data with actual GPAs
    
    # If test data doesn't exist, check for alternatives
    if not os.path.exists(TEST_DATA_PATH):
        possible_test_paths = [
            "data/student_performance_test.csv",
            "data/cleaned_student_performance.csv",  # Might need to split this
            "../data/test_students.csv"
        ]
        
        for path in possible_test_paths:
            if os.path.exists(path):
                TEST_DATA_PATH = path
                break
    
    # Initialize evaluator
    evaluator = ModelEvaluator(MODEL_PATH, TEST_DATA_PATH)
    
    if not evaluator.load_model_and_data():
        print("❌ Failed to load model and data")
        return
    
    # Generate comprehensive report
    print("\n📊 Calculating performance metrics...")
    report = evaluator.generate_detailed_report()
    
    # Display summary
    display_evaluation_summary(report)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    evaluator.create_visualizations("model_evaluation_plots")
    
    # Save detailed report
    print("\n💾 Saving detailed report...")
    report_path = evaluator.save_report(report, "model_evaluation_report.json")
    
    # Display worst predictions for analysis
    if 'error_analysis' in report and 'worst_predictions' in report['error_analysis']:
        worst_preds = report['error_analysis']['worst_predictions']
        if worst_preds:
            print(f"\n⚠️  TOP 5 WORST PREDICTIONS (For Model Improvement):")
            for i, pred in enumerate(worst_preds, 1):
                print(f"   {i}. Index {pred['index']}: "
                      f"Actual={pred['actual_gpa']:.2f}, "
                      f"Predicted={pred['predicted_gpa']:.2f}, "
                      f"Error={pred['error']:.2f} "
                      f"({pred['error_percentage']:.1f}%)")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   • Summary displayed above")
    print(f"   • Visualizations saved to: model_evaluation_plots/")
    print(f"   • Detailed report saved to: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()