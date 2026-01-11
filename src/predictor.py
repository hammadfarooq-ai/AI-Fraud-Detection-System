"""
Real-time Fraud Prediction Module
Handles prediction for new transactions
"""

import pandas as pd
import numpy as np
import pickle
import os

class FraudPredictor:
    """Class to make real-time fraud predictions"""
    
    def __init__(self, model_path='models/random_forest.pkl', scaler_path='models/scaler.pkl'):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the trained model
            scaler_path (str): Path to the fitted scaler
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns = None
        
    def load_model(self, model_path=None):
        """
        Load the trained model
        
        Args:
            model_path (str): Path to the model file
        """
        if model_path is None:
            model_path = self.model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"[OK] Model loaded from {model_path}")
    
    def load_scaler(self, scaler_path=None):
        """
        Load the fitted scaler
        
        Args:
            scaler_path (str): Path to the scaler file
        """
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"[OK] Scaler loaded from {scaler_path}")
    
    def prepare_transaction(self, transaction_data):
        """
        Prepare a single transaction for prediction
        
        Args:
            transaction_data (dict or pd.Series): Transaction data
            
        Returns:
            pd.DataFrame: Prepared transaction features
        """
        # Convert to DataFrame if dict or Series
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, pd.Series):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Remove target and time columns if present
        exclude_cols = ['Time', 'Class', 'Hour']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure we have the right columns
        if self.feature_columns is None:
            # Infer from data (should match training features)
            self.feature_columns = feature_cols
        else:
            # Use the feature columns from training
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                # Add missing columns with zeros
                for col in missing_cols:
                    df[col] = 0
        
        # Select only the feature columns in the correct order
        X = df[self.feature_columns].copy()
        
        return X
    
    def predict(self, transaction_data, return_proba=True):
        """
        Predict if a transaction is fraudulent
        
        Args:
            transaction_data (dict or pd.DataFrame): Transaction data
            return_proba (bool): Whether to return probability scores
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.load_model()
        
        if self.scaler is None:
            self.load_scaler()
        
        # Prepare transaction
        X = self.prepare_transaction(transaction_data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        fraud_probability = prediction_proba[1]
        confidence = max(prediction_proba)
        
        result = {
            'is_fraud': bool(prediction == 1),
            'fraud_probability': float(fraud_probability),
            'normal_probability': float(prediction_proba[0]),
            'confidence': float(confidence),
            'prediction': 'FRAUD' if prediction == 1 else 'NORMAL'
        }
        
        return result
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions_df (pd.DataFrame): DataFrame of transactions
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        if self.model is None:
            self.load_model()
        
        if self.scaler is None:
            self.load_scaler()
        
        # Prepare transactions
        X = self.prepare_transaction(transactions_df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        prediction_probas = self.model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'is_fraud': predictions == 1,
            'fraud_probability': prediction_probas[:, 1],
            'normal_probability': prediction_probas[:, 0],
            'confidence': np.max(prediction_probas, axis=1)
        })
        
        results['prediction_label'] = results['is_fraud'].map({True: 'FRAUD', False: 'NORMAL'})
        
        return results
    
    def explain_prediction(self, transaction_data, top_n=5):
        """
        Explain prediction using feature importance (for tree-based models)
        
        Args:
            transaction_data (dict or pd.DataFrame): Transaction data
            top_n (int): Number of top features to show
            
        Returns:
            dict: Explanation with top contributing features
        """
        if self.model is None:
            self.load_model()
        
        # Check if model has feature_importances_ attribute
        if not hasattr(self.model, 'feature_importances_'):
            return {
                'explanation': 'Feature importance not available for this model type',
                'model_type': type(self.model).__name__
            }
        
        # Prepare transaction
        X = self.prepare_transaction(transaction_data)
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = X.columns
        
        # Get top contributing features
        top_indices = np.argsort(importances)[::-1][:top_n]
        
        explanation = {
            'top_features': [
                {
                    'feature': feature_names[i],
                    'importance': float(importances[i]),
                    'value': float(X.iloc[0, i])
                }
                for i in top_indices
            ],
            'model_type': type(self.model).__name__
        }
        
        return explanation


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from data_loader import DataLoader
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'creditcard.csv'
    model_path = project_root / 'models' / 'random_forest.pkl'
    scaler_path = project_root / 'models' / 'scaler.pkl'
    
    # Load sample data
    loader = DataLoader(data_path=str(data_path))
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data. Exiting.")
        exit(1)
    
    # Get a sample transaction
    sample_transaction = df.iloc[0].to_dict()
    
    # Make prediction
    predictor = FraudPredictor()
    predictor.load_model(str(model_path))
    predictor.load_scaler(str(scaler_path))
    result = predictor.predict(sample_transaction)
    
    print("\n" + "="*60)
    print("FRAUD PREDICTION RESULT")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Actual Class: {'FRAUD' if sample_transaction.get('Class') == 1 else 'NORMAL'}")
