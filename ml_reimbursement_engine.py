#!/usr/bin/env python3
"""
Enhanced ML-based Reimbursement Engine
Uses machine learning to predict reimbursement amounts based on trip details.

This implementation includes robust feature engineering and handles edge cases.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import warnings
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMLReimbursementEngine:
    """
    Enhanced Machine Learning engine for travel reimbursement calculations.
    
    This implementation uses gradient boosting with robust feature engineering,
    cross-validation, and hyperparameter tuning to predict reimbursement amounts
    based on trip duration, miles traveled, and receipt amounts.
    """
    
    def __init__(self, model_type: str = 'gb'):
        """
        Initialize the EnhancedMLReimbursementEngine.
        
        Args:
            model_type (str): Type of model to use (gb, rf, dt)
        """
        self.model_type = model_type
        self.feature_columns = [
            'days', 'miles', 'receipts', 'days_sq', 'miles_sq', 'receipts_sq', 
            'days_cubed', 'miles_cubed', 'receipts_cubed', 'days_x_miles', 
            'days_x_receipts', 'miles_x_receipts', 'days_x_miles_x_receipts', 
            'log_receipts', 'sqrt_miles', 'log_miles', 'exp_receipts', 
            'reciprocal_days', 'miles_per_day', 'receipts_per_day', 
            'receipts_per_mile', 'miles_per_dollar', 'days_per_mile', 
            'is_5_day_trip', 'has_high_mileage', 'has_high_receipts', 
            'is_weekend_trip', 'is_extended_trip', 'days_bin', 'miles_bin', 
            'receipts_bin', 'miles_per_day_bin', 'receipts_per_day_bin', 
            'is_peak_season', 'is_holiday', 'total_expense_ratio', 
            'mileage_ratio', 'receipt_ratio'
        ]
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def _create_features(self, df):
        """
        Create comprehensive features from input data with robust error handling.
        
        Args:
            df (pd.DataFrame): Input DataFrame with columns: days, miles, receipts
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # 1. Basic features with robust type conversion and validation
            df['days'] = pd.to_numeric(df['days'], errors='coerce').fillna(0).clip(lower=0)
            df['miles'] = pd.to_numeric(df['miles'], errors='coerce').fillna(0).clip(lower=0)
            df['receipts'] = pd.to_numeric(df['receipts'], errors='coerce').fillna(0).clip(lower=0)
            
            # 2. Polynomial features
            df['days_sq'] = np.square(df['days'])
            df['miles_sq'] = np.square(df['miles'])
            df['receipts_sq'] = np.square(df['receipts'])
            df['days_cubed'] = np.power(df['days'], 3)
            df['miles_cubed'] = np.power(df['miles'], 3)
            df['receipts_cubed'] = np.power(df['receipts'], 3)
            
            # 3. Interaction terms
            df['days_x_miles'] = df['days'] * df['miles']
            df['days_x_receipts'] = df['days'] * df['receipts']
            df['miles_x_receipts'] = df['miles'] * df['receipts']
            df['days_x_miles_x_receipts'] = df['days'] * df['miles'] * df['receipts']
            
            # 4. Non-linear transforms (with protection for edge cases)
            df['log_receipts'] = np.log1p(np.abs(df['receipts']))
            df['sqrt_miles'] = np.sqrt(np.abs(df['miles']) + 1e-6)
            df['log_miles'] = np.log1p(np.abs(df['miles']))
            df['exp_receipts'] = np.exp(np.minimum(df['receipts'], 10))  # Cap to avoid overflow
            df['reciprocal_days'] = np.where(df['days'] > 0, 1 / (df['days'] + 1e-6), 0)
            
            # 5. Rate features with protection against division by zero
            df['miles_per_day'] = np.where(
                df['days'] > 0, 
                df['miles'] / (df['days'] + 1e-6), 
                df['miles']
            )
            
            df['receipts_per_day'] = np.where(
                df['days'] > 0,
                df['receipts'] / (df['days'] + 1e-6),
                df['receipts']
            )
            
            df['receipts_per_mile'] = np.where(
                df['miles'] > 0,
                df['receipts'] / (df['miles'] + 1e-6),
                df['receipts']
            )
            
            df['miles_per_dollar'] = np.where(
                df['receipts'] > 0,
                df['miles'] / (df['receipts'] + 1e-6),
                0
            )
            
            df['days_per_mile'] = np.where(
                df['miles'] > 0,
                df['days'] / (df['miles'] + 1e-6),
                0
            )
            
            # 6. Special flags based on domain knowledge
            df['is_5_day_trip'] = (df['days'] == 5).astype(int)
            df['has_high_mileage'] = (df['miles'] > 500).astype(int)
            df['has_high_receipts'] = (df['receipts'] > 1000).astype(int)
            df['is_weekend_trip'] = ((df['days'] >= 2) & (df['days'] <= 3)).astype(int)
            df['is_extended_trip'] = (df['days'] > 7).astype(int)
            
            # 7. Binning with fixed edges to avoid duplicate edges issue
            days_bins = [-1, 1, 3, 5, 7, 10, 14, 21, 30, 60, float('inf')]
            miles_bins = [-1, 50, 100, 200, 400, 600, 1000, float('inf')]
            receipts_bins = [-1, 100, 300, 600, 1000, 1500, 2000, float('inf')]
            
            df['days_bin'] = pd.cut(df['days'], bins=days_bins, labels=False).fillna(0).astype(int)
            df['miles_bin'] = pd.cut(df['miles'], bins=miles_bins, labels=False).fillna(0).astype(int)
            df['receipts_bin'] = pd.cut(df['receipts'], bins=receipts_bins, labels=False).fillna(0).astype(int)
            
            # 8. Additional derived features
            df['miles_per_day_bin'] = pd.cut(
                df['miles_per_day'], 
                bins=[-1, 50, 100, 200, 400, float('inf')],
                labels=False
            ).fillna(0).astype(int)
            
            df['receipts_per_day_bin'] = pd.cut(
                df['receipts_per_day'],
                bins=[-1, 50, 100, 200, 500, float('inf')],
                labels=False
            ).fillna(0).astype(int)
            
            # 9. Temporal features (simplified examples)
            # In a real implementation, you'd use actual dates
            df['is_peak_season'] = 0  # Placeholder
            df['is_holiday'] = 0  # Placeholder
            
            # 10. Derived ratios
            total_expense = df['days'] * 100 + df['miles'] * 0.5 + df['receipts']
            df['total_expense_ratio'] = np.where(
                total_expense > 0,
                (df['days'] * 100 + df['miles'] * 0.5 + df['receipts']) / total_expense,
                1.0
            )
            
            df['mileage_ratio'] = np.where(
                df['miles'] > 0,
                (df['miles'] * 0.5) / (df['days'] * 100 + 1e-6),
                0
            )
            
            df['receipt_ratio'] = np.where(
                df['receipts'] > 0,
                df['receipts'] / (df['days'] * 100 + df['miles'] * 0.5 + 1e-6),
                0
            )
            
            # 11. Fill any remaining NaN values that might have been created
            df = df.fillna(0)
            
            # 12. Ensure all feature columns exist (fill with 0 if not created)
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # 13. Ensure consistent column order
            result = df.reindex(columns=self.feature_columns, fill_value=0)
            
            # 14. Final validation of feature matrix
            if result.isnull().values.any():
                logger.warning("NaN values detected in feature matrix. Replacing with zeros.")
                result = result.fillna(0)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in _create_features: {e}", exc_info=True)
            # Return a DataFrame with all zeros if feature creation fails
            return pd.DataFrame(0, index=df.index, columns=self.feature_columns)
    
    def train(self, X, y, test_size=0.2, random_state=42, n_splits=5, model_type=None, fast=False):
        """
        Train the enhanced ML model. If fast=True and model_type=='dt', fit a DecisionTreeRegressor with default params (no tuning, no CV).
        Args:
            X (pd.DataFrame): Input features (must contain 'days', 'miles', 'receipts')
            y (pd.Series): Target variable (reimbursement amount)
            test_size (float): Proportion of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            n_splits (int): Number of cross-validation folds (default: 5)
            model_type (str): 'gb' for GradientBoosting, 'rf' for RandomForest, 'dt' for DecisionTree. Defaults to self.model_type.
            fast (bool): If True and model_type=='dt', fit a DecisionTreeRegressor directly (no tuning, no CV).
        Returns:
            dict: Dictionary containing training metrics and model information
        Raises:
            ValueError: If input data is invalid or training fails
        """
        try:
            model_type = model_type or self.model_type
            X_features = self._create_features(X)
            X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=test_size, random_state=random_state)
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            if fast and model_type == 'dt':
                from sklearn.tree import DecisionTreeRegressor
                self.model = DecisionTreeRegressor(random_state=random_state)
                self.model.fit(X_train_scaled, y_train)
                self.is_trained = True
                y_pred_train = self.model.predict(X_train_scaled)
                y_pred_test = self.model.predict(X_test_scaled)
                metrics = {
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'best_params': 'default',
                }
                logger.info(f"Fast DecisionTree training complete. Metrics: {metrics}")
                self.training_metrics_ = metrics
                if hasattr(self.model, 'feature_importances_'):
                    self.feature_importances_ = self.model.feature_importances_
                else:
                    self.feature_importances_ = None
                self.best_params_ = 'default'
                return metrics

            # Otherwise, do full hyperparameter search as before
            if model_type == 'gb':
                model = GradientBoostingRegressor(loss='huber', random_state=random_state)
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            elif model_type == 'rf':
                model = RandomForestRegressor(random_state=random_state)
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8, 12, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            elif model_type == 'dt':
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=random_state)
                param_dist = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                }
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=20,
                cv=n_splits,
                scoring='neg_mean_absolute_error',
                random_state=random_state,
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(X_train_scaled, y_train)
            self.model = random_search.best_estimator_
            self.is_trained = True

            # Evaluate
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Get feature importances
            feature_importances = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            # Store metrics
            metrics = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': self.best_params_,
                'feature_importances': feature_importances,
                'cv_results': random_search.cv_results_
            }
            
            # Store feature importances and metrics
            self.feature_importances_ = feature_importances
            self.training_metrics_ = metrics
            self.is_trained = True
            
            # Log feature importances
            logger.info("Top 10 most important features:")
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                logger.info(f"  {feature}: {importance:.4f}")
            
            # Log metrics
            logger.info(f"Training MAE: ${train_mae:.2f}")
            logger.info(f"Test MAE: ${test_mae:.2f}")
            logger.info(f"Training RMSE: ${train_rmse:.2f}")
            logger.info(f"Test RMSE: ${test_rmse:.2f}")
            logger.info(f"Training R²: {train_r2:.4f}")
            logger.info(f"Test R²: {test_r2:.4f}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def predict(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Predict the reimbursement amount for a trip.
        
        Args:
            trip_duration_days (float): Duration of the trip in days
            miles_traveled (float): Total miles traveled
            total_receipts_amount (float): Total amount from receipts
                
        Returns:
            float: Predicted reimbursement amount (non-negative)
            
        Raises:
            ValueError: If input values are invalid
            RuntimeError: If model is not trained or prediction fails
        """
        try:
            if not self.is_trained or self.model is None:
                raise RuntimeError("Model has not been trained. Call train() first.")
                
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [trip_duration_days, miles_traveled, total_receipts_amount]):
                raise ValueError("All input values must be numbers")
            
            # Ensure non-negative inputs
            trip_duration_days = max(0, float(trip_duration_days))
            miles_traveled = max(0.0, float(miles_traveled))
            total_receipts_amount = max(0.0, float(total_receipts_amount))
            
            # Create input DataFrame
            input_data = pd.DataFrame({
                'days': [trip_duration_days],
                'miles': [miles_traveled],
                'receipts': [total_receipts_amount]
            })
            
            # Create features
            X = self._create_features(input_data)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)
            
            # Ensure non-negative prediction and round to 2 decimal places
            return round(max(0, float(prediction[0])), 2)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fall back to a simple rule-based prediction if ML prediction fails
            logger.warning("Falling back to rule-based prediction")
            return self._fallback_prediction(trip_duration_days, miles_traveled, total_receipts_amount)
    
    def _fallback_prediction(self, days, miles, receipts):
        """
        Simple fallback prediction when ML model fails.
        
        Args:
            days (float): Trip duration in days
            miles (float): Miles traveled
            receipts (float): Total receipts amount
            
        Returns:
            float: Estimated reimbursement amount
        """
        try:
            # Basic per diem + mileage + receipts (capped at 3x per diem)
            daily_rate = 100.0
            mileage_rate = 0.5
            
            base = daily_rate * max(1, days)
            mileage = mileage_rate * miles
            receipt_portion = min(receipts, daily_rate * 3 * max(1, days))
            
            return round(base + mileage + receipt_portion, 2)
        except Exception as e:
            # If even the fallback fails, return a conservative estimate
            logger.error(f"Fallback prediction failed: {e}")
            return round(max(100, days * 75), 2)

    def save_model(self, path):
        """
        Save the trained model, scaler, and metadata to disk.
        
        Args:
            path (str): File path to save the model to
            
        Raises:
            RuntimeError: If model is not trained or save fails
        """
        try:
            if not self.is_trained or self.model is None:
                raise RuntimeError("Cannot save an untrained model")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save model and metadata
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'training_metrics': getattr(self, 'training_metrics_', None),
                'feature_importances': getattr(self, 'feature_importances_', None),
                'best_params': getattr(self, 'best_params_', None)
            }, path)
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def load_model(self, path):
        """
        Load a trained model, scaler, and metadata from disk.
        
        Args:
            path (str): File path to load the model from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails or is incompatible
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
                
            # Load model and metadata
            data = joblib.load(path)
            
            # Validate loaded data
            required_keys = ['model', 'scaler', 'feature_columns']
            if not all(key in data for key in required_keys):
                raise RuntimeError("Invalid model file format")
            
            # Set model attributes
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.is_trained = True
            
            # Set optional attributes if they exist
            if 'training_metrics' in data:
                self.training_metrics_ = data['training_metrics']
            if 'feature_importances' in data:
                self.feature_importances_ = data['feature_importances']
            if 'best_params' in data:
                self.best_params_ = data['best_params']
            
            logger.info(f"Successfully loaded model from {path}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if isinstance(e, (FileNotFoundError, RuntimeError)):
                raise
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def load_training_data(filepath='public_cases.json'):
        """
        Load training data from JSON file.
        
        Args:
            filepath (str): Path to the JSON file containing test cases
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        try:
            with open(filepath, 'r') as f:
                test_cases = json.load(f)
            
            # Convert to DataFrame
            data = []
            for case in test_cases:
                data.append({
                    'days': case['input']['trip_duration_days'],
                    'miles': case['input']['miles_traveled'],
                    'receipts': case['input']['total_receipts_amount'],
                    'output': case['expected_output']
                })
            
            df = pd.DataFrame(data)
            X = df[['days', 'miles', 'receipts']]
            y = df['output']
            
            logger.info(f"Successfully loaded {len(df)} training examples from {filepath}")
            return X, y
            
        except Exception as e:
            error_msg = f"Failed to load training data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


def train_and_evaluate():
    """Train and evaluate the ML model."""
    try:
        # Load data
        X, y = EnhancedMLReimbursementEngine.load_training_data()
        
        # Initialize and train model
        engine = EnhancedMLReimbursementEngine()
        metrics = engine.train(X, y)
        
        # Print metrics
        print("\n=== Model Training Results ===")
        print(f"Training MAE: ${metrics['train_mae']:.2f}" if 'train_mae' in metrics else "No training MAE available")
        print(f"Test MAE: ${metrics['test_mae']:.2f}" if 'test_mae' in metrics else "No test MAE available")
        print(f"Training RMSE: ${metrics['train_rmse']:.2f}" if 'train_rmse' in metrics else "No training RMSE available")
        print(f"Test RMSE: ${metrics['test_rmse']:.2f}" if 'test_rmse' in metrics else "No test RMSE available")
        
        # Save model
        model_path = 'reimbursement_model.joblib'
        engine.save_model(model_path)
        print(f"\nModel saved to '{model_path}'")
        
        return engine
        
    except Exception as e:
        logger.error(f"Training and evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    train_and_evaluate()
