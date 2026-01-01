"""
New Student GPA Prediction Module
Updated for the actual student performance dataset structure.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
from typing import Dict, List, Any, Optional, Union
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipeline import GPAModelPipeline
from data_loader import GPADataLoader

# class missing handler
# Re-define MissingValueHandler class to match the pickled version
class MissingValueHandler:
    """Dummy class to allow unpickling."""
    def __init__(self, strategy: str = 'contextual'):
        self.strategy = strategy
        self.imputation_values = {}
        self.imputer = None
    
    def analyze_missing_patterns(self, df):
        return {}
    
    def handle_missing_values(self, df):
        return df
    
    # Add other methods as needed, or leave as pass

# Now import the actual class
from model_pipeline import GPAModelPipeline


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewStudentPredictor:
    """
    Class for predicting GPA of new students using trained models.
    Updated for student performance dataset.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model_pipeline = None
        self.feature_names = None
        self.expected_features = None
        self.is_loaded = False
        
        # Define attributes that should NOT be present (removed features)
        self.removed_attributes = ['Attendance_Rate', 'Enrollment_Status']
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Initialize pipeline and load model
            self.model_pipeline = GPAModelPipeline()
            self.model_pipeline.load_model(model_path)
            
            self.feature_names = self.model_pipeline.feature_names
            self.expected_features = set(self.feature_names)
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully from: {model_path}")
            logger.info(f"Expected features: {self.feature_names}")
            logger.info(f"Model type: {self.model_pipeline.model_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def load_new_students_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load new students data from CSV file.
        
        Args:
            csv_path (str): Path to the CSV file with new student data
            
        Returns:
            pd.DataFrame: DataFrame containing new student data
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} new students from: {csv_path}")
            logger.info(f"Columns in CSV: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def validate_new_student_data(self, student_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate new student data against expected features.
        
        Args:
            student_data (Union[Dict, pd.DataFrame]): New student data
            
        Returns:
            Dict: Validation results
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Call load_model() first.")
        
        validation_result = {
            'is_valid': True,
            'missing_features': [],
            'extra_features': [],
            'has_removed_attributes': False,
            'removed_attributes_found': [],
            'warnings': [],
            'errors': []
        }
        
        # Convert to DataFrame for easier processing
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()
        
        # Check for removed attributes
        current_columns = set(df.columns)
        removed_found = [attr for attr in self.removed_attributes if attr in current_columns]
        
        if removed_found:
            validation_result['has_removed_attributes'] = True
            validation_result['removed_attributes_found'] = removed_found
            validation_result['warnings'].append(
                f"Found removed attributes in data: {removed_found}. "
                f"These will be automatically excluded from prediction."
            )
        
        # Check for missing expected features
        missing_features = self.expected_features - current_columns
        if missing_features:
            validation_result['missing_features'] = list(missing_features)
            validation_result['errors'].append(
                f"Missing expected features: {list(missing_features)}"
            )
            validation_result['is_valid'] = False
        
        # Check for extra features (not in expected features)
        extra_features = current_columns - self.expected_features
        if extra_features:
            # Exclude StudentID from extra features (it's allowed as identifier)
            extra_features = extra_features - {'StudentID'}
            if extra_features:
                validation_result['extra_features'] = list(extra_features)
                validation_result['warnings'].append(
                    f"Extra features found: {list(extra_features)}. These will be ignored."
                )
        
        # Validate data types
        for feature in self.expected_features:
            if feature in df.columns:
                # Check for null values
                if df[feature].isnull().any():
                    validation_result['warnings'].append(
                        f"Feature '{feature}' contains null values"
                    )
                
                # Check if feature is numeric
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    validation_result['errors'].append(
                        f"Feature '{feature}' must be numeric, but found {df[feature].dtype}"
                    )
                    validation_result['is_valid'] = False
        
        logger.info(f"Data validation completed. Valid: {validation_result['is_valid']}")
        return validation_result
    
    def preprocess_new_student_data(self, student_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess new student data for prediction.
        
        Args:
            student_data (Union[Dict, pd.DataFrame]): New student data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        logger.info("Preprocessing new student data...")
        
        # Convert to DataFrame
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()
        
        # Remove any excluded attributes if present
        columns_to_remove = [attr for attr in self.removed_attributes if attr in df.columns]
        if columns_to_remove:
            logger.info(f"Removing excluded attributes: {columns_to_remove}")
            df = df.drop(columns=columns_to_remove, errors='ignore')
        
        # Ensure only expected features are present (keep StudentID for reference)
        extra_columns = set(df.columns) - self.expected_features - {'StudentID'}
        if extra_columns:
            logger.info(f"Removing extra columns: {list(extra_columns)}")
            df = df.drop(columns=list(extra_columns), errors='ignore')
        
        # Add missing features with default values
        missing_columns = self.expected_features - set(df.columns)
        if missing_columns:
            logger.warning(f"Adding missing features with default values: {list(missing_columns)}")
            for column in missing_columns:
                # Use appropriate default values based on feature characteristics
                if 'GPA' in column:
                    df[column] = 3.0  # Default GPA
                elif 'Hours' in column:
                    df[column] = 15.0  # Default for credit hours
                elif 'Gender' in column:
                    df[column] = 0  # Default gender (Male)
                else:
                    df[column] = 0.0  # Default for other numerical features
        
        # Reorder columns to match training data (exclude StudentID for prediction)
        prediction_features = [col for col in self.feature_names if col in df.columns]
        df_for_prediction = df[prediction_features]
        
        logger.info(f"Preprocessed data shape: {df_for_prediction.shape}")
        return df_for_prediction
    
    def predict_single_student(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict Final Year GPA for a single student.
        
        Args:
            student_data (Dict): Student features as dictionary
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info("Predicting Final Year GPA for single student...")
        
        # Validate data
        validation = self.validate_new_student_data(student_data)
        
        if not validation['is_valid']:
            raise ValueError(f"Invalid student data: {validation['errors']}")
        
        # Preprocess data
        processed_data = self.preprocess_new_student_data(student_data)
        
        # Make prediction
        try:
            prediction = self.model_pipeline.predict(processed_data)
            gpa_prediction = float(prediction[0])
            
            # Generate insights
            insights = self._generate_insights(processed_data, gpa_prediction)
            
            result = {
                'student_id': student_data.get('StudentID', 'Unknown'),
                'predicted_final_gpa': round(gpa_prediction, 2),
                'insights': insights,
                'status': 'success'
            }
            
            logger.info(f"Prediction successful: Final GPA = {result['predicted_final_gpa']}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'student_id': student_data.get('StudentID', 'Unknown'),
                'predicted_final_gpa': None,
                'error': str(e),
                'status': 'error'
            }
    
    def predict_multiple_students(self, students_data: List[Dict]) -> Dict[str, Any]:
        """
        Predict Final Year GPA for multiple students.
        
        Args:
            students_data (List[Dict]): List of student data dictionaries
            
        Returns:
            Dict: Batch prediction results
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Predicting Final Year GPA for {len(students_data)} students...")
        
        results = {
            'total_students': len(students_data),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'predictions': []
        }
        
        for i, student_data in enumerate(students_data):
            try:
                prediction_result = self.predict_single_student(student_data)
                results['predictions'].append(prediction_result)
                
                if prediction_result['status'] == 'success':
                    results['successful_predictions'] += 1
                else:
                    results['failed_predictions'] += 1
                    
            except Exception as e:
                logger.error(f"Prediction failed for student {i}: {str(e)}")
                error_result = {
                    'student_id': student_data.get('StudentID', f'Student_{i}'),
                    'predicted_final_gpa': None,
                    'error': str(e),
                    'status': 'error'
                }
                results['predictions'].append(error_result)
                results['failed_predictions'] += 1
        
        # Calculate summary statistics
        successful_predictions = [p for p in results['predictions'] if p['status'] == 'success']
        if successful_predictions:
            gpas = [p['predicted_final_gpa'] for p in successful_predictions]
            results['summary'] = {
                'average_predicted_gpa': round(np.mean(gpas), 2),
                'min_predicted_gpa': round(min(gpas), 2),
                'max_predicted_gpa': round(max(gpas), 2),
                'std_predicted_gpa': round(np.std(gpas), 2)
            }
        
        logger.info(f"Batch prediction completed: {results['successful_predictions']} successful, "
                   f"{results['failed_predictions']} failed")
        
        return results
    
    def predict_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Predict Final Year GPA for students from a CSV file.
        
        Args:
            csv_path (str): Path to CSV file with new student data
            
        Returns:
            Dict: Batch prediction results
        """
        logger.info(f"Predicting from CSV file: {csv_path}")
        
        # Load data from CSV
        df = self.load_new_students_from_csv(csv_path)
        
        # Convert DataFrame to list of dictionaries
        students_data = df.to_dict('records')
        
        # Make predictions
        results = self.predict_multiple_students(students_data)
        
        # Add CSV file info to results
        results['source_file'] = csv_path
        results['students_processed'] = len(students_data)
        
        return results
    
    def _generate_insights(self, features: pd.DataFrame, predicted_gpa: float) -> List[str]:
        """
        Generate insights and recommendations based on prediction.
        
        Args:
            features (pd.DataFrame): Student features
            predicted_gpa (float): Predicted final GPA
            
        Returns:
            List[str]: List of insights and recommendations
        """
        insights = []
        
        # GPA range insights
        if predicted_gpa >= 3.5:
            insights.append("Predicted excellent academic performance (GPA ≥ 3.5)")
            insights.append("Strong candidate for honors or advanced programs")
        elif predicted_gpa >= 3.0:
            insights.append("Predicted good academic performance (GPA 3.0-3.49)")
            insights.append("Maintains solid academic standing")
        elif predicted_gpa >= 2.5:
            insights.append("Predicted average academic performance (GPA 2.5-2.99)")
            insights.append("May benefit from additional academic support")
        else:
            insights.append("Predicted below average performance - may need academic intervention")
            insights.append("Consider academic advising and tutoring services")
        
        # Feature-based insights
        feature_insights = self._generate_feature_insights(features, predicted_gpa)
        insights.extend(feature_insights)
        
        return insights
    
    def _generate_feature_insights(self, features: pd.DataFrame, predicted_gpa: float) -> List[str]:
        """
        Generate insights based on specific feature values.
        
        Args:
            features (pd.DataFrame): Student features
            predicted_gpa (float): Predicted final GPA
            
        Returns:
            List[str]: Feature-based insights
        """
        insights = []
        
        try:
            # Check previous GPA trends
            if 'Year3_GPA' in features.columns and 'Year1_GPA' in features.columns:
                year3_gpa = features['Year3_GPA'].iloc[0]
                year1_gpa = features['Year1_GPA'].iloc[0]
                
                if year3_gpa > year1_gpa + 0.5:
                    insights.append("Strong academic improvement trend detected")
                elif year3_gpa < year1_gpa - 0.3:
                    insights.append("Declining academic performance trend - may need support")
            
            # Check credit hours
            if 'Credit_Hours_Avg' in features.columns:
                credit_hours = features['Credit_Hours_Avg'].iloc[0]
                if credit_hours > 18:
                    insights.append("High course load detected - ensure manageable schedule")
                elif credit_hours < 12:
                    insights.append("Light course load - consider adding courses if appropriate")
            
            # Gender-based insights (if relevant)
            if 'Gender_F' in features.columns:
                gender = "Female" if features['Gender_F'].iloc[0] == 1 else "Male"
                insights.append(f"Gender: {gender}")
            
        except Exception as e:
            logger.warning(f"Could not generate feature insights: {str(e)}")
        
        return insights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        if not self.is_loaded:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': self.model_pipeline.model_type,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'removed_attributes': self.removed_attributes,
            'training_metrics': {
                'train_r2': self.model_pipeline.train_metrics.get('r2', 'N/A'),
                'test_r2': self.model_pipeline.test_metrics.get('r2', 'N/A'),
                'test_rmse': self.model_pipeline.test_metrics.get('rmse', 'N/A'),
                'test_mae': self.model_pipeline.test_metrics.get('mae', 'N/A')
            },
            'cv_scores': {
                'mean_r2': self.model_pipeline.cv_scores.get('mean_r2', 'N/A'),
                'mean_rmse': self.model_pipeline.cv_scores.get('mean_rmse', 'N/A')
            },
            'model_path': self.model_path
        }

# Utility functions
def create_sample_student() -> Dict[str, Any]:
    """
    Create a sample student data dictionary for testing.
    
    Returns:
        Dict: Sample student data
    """
    return {
        'StudentID': '2024ITIBB0001',
        'Gender_F': 1,
        'Year1_GPA': 3.2,
        'Year2_GPA': 3.4,
        'Year3_GPA': 3.5,
        'Credit_Hours_Avg': 16.0,
        # Note: 'Attendance_Rate' and 'Enrollment_Status' are intentionally excluded
    }

def display_model_accuracy(model_info: Dict[str, Any]) -> None:
    """
    Display model accuracy metrics in a clean format.
    
    Args:
        model_info (Dict): Model information dictionary
    """
    metrics = model_info['training_metrics']
    cv_scores = model_info['cv_scores']
    
    print("\n" + "="*60)
    print("MODEL ACCURACY METRICS")
    print("="*60)
    
    print(f"\nAlgorithm: {model_info['model_type'].replace('_', ' ').title()}")
    
    print(f"\n📊 Performance on Training Data:")
    print(f"   R² Score: {metrics['train_r2']:.3f} ({(metrics['train_r2']*100):.1f}% variance explained)")
    
    print(f"\n🎯 Performance on Test Data (Real-world accuracy):")
    print(f"   R² Score: {metrics['test_r2']:.3f} ({(metrics['test_r2']*100):.1f}% variance explained)")
    print(f"   RMSE: {metrics['test_rmse']:.3f} GPA points (average error)")
    print(f"   MAE: {metrics['test_mae']:.3f} GPA points (average absolute error)")
    
    print(f"\n🔄 Cross-Validation Performance:")
    print(f"   Average R²: {cv_scores['mean_r2']:.3f}")
    print(f"   Average RMSE: {cv_scores['mean_rmse']:.3f} GPA points")
    
    print(f"\n📈 Interpretation:")
    if metrics['test_r2'] >= 0.9:
        print("   ✅ Excellent predictive accuracy")
    elif metrics['test_r2'] >= 0.8:
        print("   ✅ Very good predictive accuracy") 
    elif metrics['test_r2'] >= 0.7:
        print("   ✅ Good predictive accuracy")
    elif metrics['test_r2'] >= 0.6:
        print("   ⚠️  Moderate predictive accuracy")
    else:
        print("   ❌ Low predictive accuracy")
    
    if metrics['test_rmse'] <= 0.15:
        print("   ✅ High precision (low error)")
    elif metrics['test_rmse'] <= 0.25:
        print("   ✅ Good precision")
    else:
        print("   ⚠️  Moderate precision")
    
    print("="*60)

def display_predictions(results: Dict[str, Any]) -> None:
    """
    Display prediction results in a clean format in the console.
    
    Args:
        results (Dict): Prediction results
    """
    print("\n" + "="*80)
    print("FINAL YEAR GPA PREDICTION RESULTS")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"Total students processed: {results['total_students']}")
    print(f"Successful predictions: {results['successful_predictions']}")
    print(f"Failed predictions: {results['failed_predictions']}")
    
    if 'summary' in results:
        print(f"\nOverall Statistics:")
        print(f"Average Predicted GPA: {results['summary']['average_predicted_gpa']}")
        print(f"Minimum Predicted GPA: {results['summary']['min_predicted_gpa']}")
        print(f"Maximum Predicted GPA: {results['summary']['max_predicted_gpa']}")
        print(f"Standard Deviation: {results['summary']['std_predicted_gpa']}")
    
    print(f"\nIndividual Student Predictions:")
    print("-" * 80)
    
    for i, prediction in enumerate(results['predictions']):
        print(f"\nStudent #{i+1}:")
        print(f"  Student ID: {prediction['student_id']}")
        
        if prediction['status'] == 'success':
            print(f"  Predicted Final Year GPA: {prediction['predicted_final_gpa']}")
            print(f"  Insights:")
            for insight in prediction['insights']:
                print(f"    • {insight}")
        else:
            print(f"  ❌ PREDICTION FAILED")
            print(f"  Error: {prediction['error']}")
    
    print("-" * 80)
    print("End of predictions")
    print("="*80)

# Main execution
if __name__ == "__main__":
    """
    Example usage of the New Student Predictor with CSV file input.
    """
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model is in models/ directory which is at the same level as src/
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    MODEL_PATH = os.path.join(project_root, "models", "best_gpa_model.pkl")
    NEW_STUDENTS_CSV = os.path.join(project_root, "data", "new_students_for_prediction.csv")

    # Debug print (you can remove this later)
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}") '''
    
    # Configuration
    MODEL_PATH = "models/best_gpa_model.pkl"
    NEW_STUDENTS_CSV = "data/new_students_for_prediction.csv"
    
    try:
        # Initialize predictor
        predictor = NewStudentPredictor(MODEL_PATH)
        
        if not predictor.is_loaded:
            print("❌ Failed to load model. Please train a model first by running model_pipeline.py")
            exit(1)
        
        # Display model information and accuracy
        model_info = predictor.get_model_info()
        display_model_accuracy(model_info)
        
        # Check if CSV file exists
        if not os.path.exists(NEW_STUDENTS_CSV):
            print(f"❌ CSV file not found: {NEW_STUDENTS_CSV}")
            print("Please ensure the file exists in the data directory.")
            exit(1)
        
        # Predict from CSV file
        print(f"\n📊 Predicting from CSV file: {NEW_STUDENTS_CSV}")
        
        results = predictor.predict_from_csv(NEW_STUDENTS_CSV)
        
        # Display results in console
        display_predictions(results)
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()