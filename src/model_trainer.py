"""
Model Training Module
Trains multiple ML models for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class to train and save fraud detection models"""
    
    def __init__(self, fast_mode=False):
        """
        Initialize the model trainer
        
        Args:
            fast_mode (bool): If True, use faster training with reduced parameters
        """
        self.models = {}
        self.best_params = {}
        self.fast_mode = fast_mode
        
    def train_logistic_regression(self, X_train, y_train, use_class_weight=True):
        """
        Train Logistic Regression model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            use_class_weight (bool): Whether to use class weights for imbalance
            
        Returns:
            LogisticRegression: Trained model
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression...")
        print("-"*60)
        
        # Calculate class weights if needed
        class_weight = None
        if use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))
            print(f"  Using class weights: {class_weight}")
        
        if self.fast_mode:
            # Fast mode: skip grid search, use default parameters
            print("  Fast mode: Using default parameters (no grid search)")
            lr = LogisticRegression(
                class_weight=class_weight,
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=500,
                random_state=42
            )
            lr.fit(X_train, y_train)
            self.models['logistic_regression'] = lr
            self.best_params['logistic_regression'] = {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}
            print(f"  Training completed")
        else:
            # Normal mode: use grid search
            lr = LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            )
            
            print("  Performing grid search...")
            grid_search = GridSearchCV(
                lr, 
                param_grid={'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
                cv=3,
                scoring='roc_auc',
                n_jobs=1,  # Set to 1 for Windows compatibility
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            
            self.models['logistic_regression'] = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_
        
        return self.models['logistic_regression']
    
    def train_random_forest(self, X_train, y_train, use_class_weight=True):
        """
        Train Random Forest model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            use_class_weight (bool): Whether to use class weights
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training Random Forest...")
        print("-"*60)
        
        # Calculate class weights if needed
        class_weight = None
        if use_class_weight:
            class_weight = 'balanced'
            print("  Using balanced class weights")
        
        if self.fast_mode:
            # Fast mode: use fewer trees, no grid search
            print("  Fast mode: Using reduced parameters (no grid search)")
            rf = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100-200
                max_depth=15,  # Reduced depth
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=42,
                n_jobs=1
            )
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf
            self.best_params['random_forest'] = {'n_estimators': 50, 'max_depth': 15}
            print(f"  Training completed")
        else:
            # Normal mode: use grid search
            rf = RandomForestClassifier(
                class_weight=class_weight,
                random_state=42,
                n_jobs=1  # Set to 1 for Windows compatibility
            )
            
            print("  Performing grid search...")
            grid_search = GridSearchCV(
                rf,
                param_grid={'n_estimators': [100, 200], 'max_depth': [20, None], 
                           'min_samples_split': [2, 5]},
                cv=3,
                scoring='roc_auc',
                n_jobs=1,  # Set to 1 for Windows compatibility
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            
            self.models['random_forest'] = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
        
        return self.models['random_forest']
    
    def train_isolation_forest(self, X_train, y_train):
        """
        Train Isolation Forest (unsupervised anomaly detection)
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target (for evaluation only)
            
        Returns:
            IsolationForest: Trained model
        """
        print("\n" + "-"*60)
        print("Training Isolation Forest (Anomaly Detection)...")
        print("-"*60)
        
        # Isolation Forest is unsupervised, so we use all data
        # contamination is the expected proportion of outliers
        fraud_ratio = y_train.mean()
        contamination = max(0.001, min(0.5, fraud_ratio * 2))  # Slightly higher than actual
        
        print(f"  Estimated contamination: {contamination:.4f}")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': [contamination, contamination * 1.5, contamination * 2],
            'max_features': [0.5, 0.75, 1.0]
        }
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        print("  Performing grid search...")
        grid_search = GridSearchCV(
            iso_forest,
            param_grid={'n_estimators': [100, 200], 
                       'contamination': [contamination, contamination * 1.5],
                       'max_features': [0.75, 1.0]},
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        # For Isolation Forest, we need to convert predictions to match y_train format
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # We'll use a wrapper to convert this
        
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class IsolationForestWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, n_estimators=100, contamination=0.1, max_features=1.0):
                self.n_estimators = n_estimators
                self.contamination = contamination
                self.max_features = max_features
                self.model = None
                
            def fit(self, X, y):
                self.model = IsolationForest(
                    n_estimators=self.n_estimators,
                    contamination=self.contamination,
                    max_features=self.max_features,
                    random_state=42,
                    n_jobs=1  # Set to 1 for Windows compatibility
                )
                self.model.fit(X)
                return self
                
            def predict(self, X):
                predictions = self.model.predict(X)
                # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0
                return (predictions == -1).astype(int)
            
            def predict_proba(self, X):
                # Get anomaly scores
                scores = self.model.score_samples(X)
                # Convert scores to probabilities (lower score = higher fraud probability)
                # Normalize scores to [0, 1] range
                scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
                # Invert so lower scores (anomalies) have higher fraud probability
                fraud_proba = 1 - scores_normalized
                # Stack probabilities
                proba = np.column_stack([1 - fraud_proba, fraud_proba])
                return proba
        
        wrapper = IsolationForestWrapper()
        param_grid_wrapper = {
            'n_estimators': [100, 200],
            'contamination': [contamination, contamination * 1.5],
            'max_features': [0.75, 1.0]
        }
        
        grid_search = GridSearchCV(
            wrapper,
            param_grid=param_grid_wrapper,
            cv=3,
            scoring='roc_auc',
            n_jobs=1,  # Set to 1 for Windows compatibility
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        self.models['isolation_forest'] = grid_search.best_estimator_
        self.best_params['isolation_forest'] = grid_search.best_params_
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, include_isolation=True):
        """
        Train all models
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            include_isolation (bool): Whether to include Isolation Forest
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS" + (" (FAST MODE)" if self.fast_mode else ""))
        print("="*60)
        
        # Train supervised models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        # Train unsupervised model if requested (skip in fast mode as it's slow)
        if include_isolation and not self.fast_mode:
            self.train_isolation_forest(X_train, y_train)
        elif include_isolation and self.fast_mode:
            print("\n" + "-"*60)
            print("Skipping Isolation Forest in fast mode (too slow)")
            print("-"*60)
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        return self.models
    
    def save_model(self, model, model_name, filepath=None):
        """
        Save a trained model
        
        Args:
            model: Trained model object
            model_name (str): Name of the model
            filepath (str): Path to save the model
        """
        if filepath is None:
            filepath = f'models/{model_name}.pkl'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"[OK] Model saved to {filepath}")
    
    def save_all_models(self, base_path='models'):
        """Save all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = f'{base_path}/{model_name}.pkl'
            self.save_model(model, model_name, filepath)
    
    def load_model(self, model_name, filepath=None):
        """
        Load a saved model
        
        Args:
            model_name (str): Name of the model
            filepath (str): Path to the model file
            
        Returns:
            Loaded model object
        """
        if filepath is None:
            filepath = f'models/{model_name}.pkl'
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"[OK] Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage - FAST MODE (completes in < 1 minute)
    import time
    from pathlib import Path
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    
    start_time = time.time()
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'creditcard.csv'
    
    print("\n" + "="*80)
    print("FAST MODE - MODEL TRAINING PIPELINE (Target: < 1 minute)")
    print("="*80)
    
    # Load data
    print("\n[STEP 1/4] Loading data...")
    loader = DataLoader(data_path=str(data_path))
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data. Exiting.")
        exit(1)
    
    # Sample data for faster processing (use 20% of data)
    print("\n[STEP 2/4] Sampling data for fast processing...")
    np.random.seed(42)
    sample_size = int(len(df) * 0.2)
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    df = df.iloc[sample_indices].reset_index(drop=True)
    print(f"  Using {len(df):,} samples ({sample_size/len(df)*100:.1f}% of original)")
    
    # Preprocess
    print("\n[STEP 3/4] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, 
        use_smote=True, 
        smote_ratio=0.1  # Lower ratio for faster SMOTE
    )
    
    # Sample training data for faster training (use 10% of training data)
    print("\n  Sampling training data for faster training...")
    np.random.seed(42)
    train_sample_size = int(len(X_train) * 0.1)
    train_sample_indices = np.random.choice(len(X_train), size=train_sample_size, replace=False)
    X_train = X_train.iloc[train_sample_indices].reset_index(drop=True)
    y_train = y_train.iloc[train_sample_indices].reset_index(drop=True)
    print(f"  Training on {len(X_train):,} samples")
    
    # Train models in fast mode
    print("\n[STEP 4/4] Training models (FAST MODE)...")
    trainer = ModelTrainer(fast_mode=True)  # Enable fast mode
    models = trainer.train_all_models(X_train, y_train, include_isolation=False)
    
    # Save models
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    trainer.save_all_models(base_path=str(models_dir))
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"⏱️  TOTAL TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*80)