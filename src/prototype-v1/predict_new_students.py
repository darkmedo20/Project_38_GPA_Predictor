"""
Simplified New Student GPA Predictor
Direct model loading and prediction without complex pipeline dependencies
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import sys
from typing import Dict, List, Any

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGPAPredictor:
    """
    Simplified GPA predictor that loads models directly.
    Handles both pickle (.pkl) and joblib (.joblib) formats.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.scaler = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from file.
        Supports both pickle and joblib formats.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Try joblib first (preferred for sklearn models)
            if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
                try:
                    model_data = joblib.load(model_path)
                    logger.info("Loaded with joblib")
                except:
                    # Try pickle with MissingValueHandler workaround
                    logger.info("Trying pickle with workaround...")
                    model_data = self._load_pickle_with_workaround(model_path)
            else:
                raise ValueError("Unsupported model format. Use .pkl or .joblib")
            
            # Extract model and metadata
            self.model = model_data.get('model')
            if self.model is None:
                # Try direct model if wrapped differently
                self.model = model_data
            
            self.feature_names = model_data.get('feature_names', [])
            self.scaler = model_data.get('scaler', None)
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Model type: {type(self.model).__name__}")
            logger.info(f"   Features expected: {len(self.feature_names)}")
            if self.feature_names:
                logger.info(f"   Feature list: {self.feature_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def _load_pickle_with_workaround(self, model_path: str) -> Any:
        """
        Load pickle file with MissingValueHandler workaround.
        """
        # Define a dummy MissingValueHandler class to allow unpickling
        class DummyMissingValueHandler:
            def __init__(self, *args, **kwargs):
                self.strategy = kwargs.get('strategy', 'contextual')
                self.imputation_values = {}
            
            def analyze_missing_patterns(self, df):
                return {'missing_cells': 0}
            
            def handle_missing_values(self, df):
                return df.fillna(0) if df is not None else df
        
        # Register the dummy class
        import __main__
        __main__.MissingValueHandler = DummyMissingValueHandler
        
        # Load the pickle file
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def load_students_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load new student data from CSV.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Student data
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            logger.info(f"📊 Loaded {len(df)} students from: {csv_path}")
            logger.info(f"   Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def prepare_features(self, student_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            student_data (pd.DataFrame): Raw student data
            
        Returns:
            pd.DataFrame: Prepared features ready for prediction
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create a copy to avoid modifying original
        df = student_data.copy()
        
        # Store StudentID if present
        if 'StudentID' in df.columns:
            student_ids = df['StudentID']
            df_features = df.drop('StudentID', axis=1)
        else:
            student_ids = pd.Series([f"Student_{i}" for i in range(len(df))])
            df_features = df
        
        # If no feature names specified, use all numeric columns
        if not self.feature_names:
            self.feature_names = [col for col in df_features.columns 
                                 if pd.api.types.is_numeric_dtype(df_features[col])]
            logger.info(f"Auto-detected features: {self.feature_names}")
        
        # Handle missing features
        missing_features = [f for f in self.feature_names if f not in df_features.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
            for feature in missing_features:
                # Use sensible defaults
                if 'GPA' in feature:
                    df_features[feature] = 3.0  # Average GPA
                elif 'Gender' in feature:
                    df_features[feature] = 0  # Default to male
                elif 'Credit' in feature or 'Hours' in feature:
                    df_features[feature] = 15.0  # Average credit hours
                else:
                    df_features[feature] = 0.0
        
        # Remove extra features not used by model
        extra_features = [f for f in df_features.columns if f not in self.feature_names]
        if extra_features:
            logger.info(f"Removing extra features: {extra_features}")
            df_features = df_features.drop(columns=extra_features)
        
        # Reorder to match training order
        df_features = df_features[self.feature_names]
        
        # Handle missing values
        if df_features.isnull().any().any():
            logger.info(f"Filling {df_features.isnull().sum().sum()} missing values")
            df_features = df_features.fillna(0)
        
        # Apply scaler if available
        if self.scaler is not None:
            logger.info("Applying feature scaling")
            df_features = pd.DataFrame(
                self.scaler.transform(df_features),
                columns=df_features.columns
            )
        
        return df_features, student_ids
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make GPA predictions.
        
        Args:
            features (pd.DataFrame): Prepared features
            
        Returns:
            np.ndarray: GPA predictions (capped 0.0-4.0)
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Make predictions
        predictions = self.model.predict(features.values)
        
        # Cap to valid GPA range
        predictions = np.clip(predictions, 0.0, 4.0)
        
        return predictions
    
    def predict_single(self, student_data: Dict) -> Dict:
        """
        Predict GPA for a single student.
        
        Args:
            student_data (Dict): Student features as dictionary
            
        Returns:
            Dict: Prediction result
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([student_data])
        
        # Prepare features and get prediction
        features, student_ids = self.prepare_features(df)
        prediction = self.predict(features)[0]
        
        # Generate insights
        insights = self._generate_insights(prediction)
        
        return {
            'student_id': student_data.get('StudentID', student_ids.iloc[0]),
            'predicted_gpa': round(float(prediction), 2),
            'insights': insights,
            'status': 'success'
        }
    
    def predict_batch(self, csv_path: str) -> Dict:
        """
        Predict GPA for all students in a CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            Dict: Batch prediction results
        """
        logger.info(f"Starting batch prediction from: {csv_path}")
        
        # Load student data
        df_students = self.load_students_from_csv(csv_path)
        
        # Prepare features
        features, student_ids = self.prepare_features(df_students)
        
        # Make predictions
        predictions = self.predict(features)
        
        # Compile results
        results = {
            'total_students': len(df_students),
            'successful_predictions': len(predictions),
            'failed_predictions': 0,
            'predictions': [],
            'summary': {}
        }
        
        # Add individual predictions
        for i, (student_id, pred) in enumerate(zip(student_ids, predictions)):
            insights = self._generate_insights(pred)
            
            results['predictions'].append({
                'student_id': str(student_id),
                'predicted_gpa': round(float(pred), 2),
                'insights': insights,
                'status': 'success'
            })
        
        # Add summary statistics
        if predictions.size > 0:
            results['summary'] = {
                'average_gpa': round(float(np.mean(predictions)), 2),
                'min_gpa': round(float(np.min(predictions)), 2),
                'max_gpa': round(float(np.max(predictions)), 2),
                'std_gpa': round(float(np.std(predictions)), 2)
            }
        
        logger.info(f"✅ Batch prediction complete: {len(predictions)} students")
        return results
    
    def _generate_insights(self, gpa: float) -> List[str]:
        """
        Generate insights based on predicted GPA.
        
        Args:
            gpa (float): Predicted GPA
            
        Returns:
            List[str]: Insights and recommendations
        """
        insights = []
        
        if gpa >= 3.5:
            insights.append("🎓 Excellent academic performance expected")
            insights.append("Consider honors programs or research opportunities")
        elif gpa >= 3.0:
            insights.append("📚 Good academic performance expected")
            insights.append("Maintain current study habits")
        elif gpa >= 2.5:
            insights.append("📖 Average performance - room for improvement")
            insights.append("Consider academic advising or tutoring")
        elif gpa >= 2.0:
            insights.append("⚠️ Below average - academic support recommended")
            insights.append("Schedule meeting with academic advisor")
        else:
            insights.append("❌ At risk - immediate intervention needed")
            insights.append("Urgent: Contact academic support services")
        
        # Add GPA-specific note
        insights.append(f"Predicted Final GPA: {gpa:.2f}/4.0")
        
        return insights
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'model_type': type(self.model).__name__,
            'features_count': len(self.feature_names),
            'features': self.feature_names,
            'model_path': self.model_path,
            'has_scaler': self.scaler is not None
        }

def display_results(results: Dict) -> None:
    """
    Display prediction results in a clean format.
    
    Args:
        results (Dict): Prediction results
    """
    print("\n" + "="*80)
    print("FINAL YEAR GPA PREDICTION RESULTS")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"Total students: {results['total_students']}")
    print(f"Successful predictions: {results['successful_predictions']}")
    print(f"Failed predictions: {results['failed_predictions']}")
    
    if 'summary' in results and results['summary']:
        print(f"\n📈 Overall Statistics:")
        print(f"Average Predicted GPA: {results['summary']['average_gpa']}")
        print(f"Minimum GPA: {results['summary']['min_gpa']}")
        print(f"Maximum GPA: {results['summary']['max_gpa']}")
        print(f"Standard Deviation: {results['summary']['std_gpa']}")
    
    print(f"\n👤 Individual Predictions:")
    print("-" * 80)
    
    for i, pred in enumerate(results['predictions']):
        print(f"\nStudent #{i+1}: {pred['student_id']}")
        print(f"  Predicted GPA: {pred['predicted_gpa']}/4.0")
        
        if pred['status'] == 'success':
            print(f"  Insights:")
            for insight in pred['insights']:
                print(f"    • {insight}")
        else:
            print(f"  ❌ Error: {pred.get('error', 'Unknown error')}")
    
    print("-" * 80)
    print("✅ Prediction complete")
    print("="*80)

def create_sample_csv(output_path: str = "data/sample_students.csv") -> None:
    """
    Create a sample CSV file for testing if none exists.
    
    Args:
        output_path (str): Path to save sample CSV
    """
    sample_data = {
        'StudentID': ['2024IT001', '2024IT002', '2024IT003', '2024IT004'],
        'Gender_F': [1, 0, 1, 0],
        'Year1_GPA': [3.2, 3.5, 2.8, 3.0],
        'Year2_GPA': [3.4, 3.6, 2.9, 3.1],
        'Year3_GPA': [3.5, 3.7, 3.0, 3.2],
        'Credit_Hours_Avg': [16.0, 18.0, 12.0, 15.0]
    }
    
    df = pd.DataFrame(sample_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample CSV at: {output_path}")
    return output_path

def main():
    """
    Main function to run predictions.
    """
    print("\n" + "="*80)
    print("SIMPLIFIED GPA PREDICTOR")
    print("="*80)
    
    # Configuration - adjust these paths as needed
    MODEL_PATHS = [
        "models/best_gpa_model.pkl",
        "models/best_gpa_model.joblib",
        "models/final_gpa_predictor_rf.pkl",
        "../models/best_gpa_model.pkl"
    ]
    
    CSV_PATHS = [
        "data/new_students_for_prediction.csv",
        "../data/new_students_for_prediction.csv",
        "new_students_for_prediction.csv"
    ]
    
    # Find model file
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model: {model_path}")
            break
    
    if model_path is None:
        print("❌ No model file found. Please ensure:")
        print("   1. You have trained a model using model_pipeline.py")
        print("   2. The model file exists in the models/ directory")
        print("\nTried:")
        for path in MODEL_PATHS:
            print(f"   - {os.path.abspath(path)}")
        return
    
    # Find CSV file
    csv_path = None
    for path in CSV_PATHS:
        if os.path.exists(path):
            csv_path = path
            print(f"✅ Found student data: {csv_path}")
            break
    
    if csv_path is None:
        print("⚠️ No student CSV found. Creating sample data...")
        csv_path = create_sample_csv("data/sample_students.csv")
    
    # Initialize predictor
    print(f"\n🔧 Initializing predictor...")
    predictor = SimpleGPAPredictor(model_path)
    
    if not predictor.is_loaded:
        print("❌ Failed to load model")
        return
    
    # Display model info
    model_info = predictor.get_model_info()
    print(f"\n📋 Model Information:")
    print(f"   Algorithm: {model_info['model_type']}")
    print(f"   Features: {model_info['features_count']}")
    if model_info['features']:
        print(f"   Feature list: {model_info['features']}")
    
    # Make predictions
    print(f"\n🎯 Making predictions...")
    results = predictor.predict_batch(csv_path)
    
    # Display results
    display_results(results)
    
    # Optional: Save results to CSV
    output_csv = "data/prediction_results.csv"
    results_df = pd.DataFrame([
        {**pred, 'insights': ' | '.join(pred['insights'])} 
        for pred in results['predictions']
    ])
    results_df.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to: {output_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure model_pipeline.py has been run successfully")
        print("   2. Check that the model file exists in models/ directory")
        print("   3. Verify your CSV file has the required columns")