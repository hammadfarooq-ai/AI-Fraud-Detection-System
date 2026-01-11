"""
Data Preprocessing Module
Handles feature scaling, class imbalance, and train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os

class DataPreprocessor:
    """Class to preprocess the credit card fraud dataset"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.smote = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        
    def prepare_features(self, df):
        """
        Prepare features and target variable
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        # Separate features and target
        # Exclude Time, Class, and Hour (if exists) from features
        exclude_cols = ['Time', 'Class', 'Hour']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].copy()
        y = df['Class'].copy()
        
        print(f"[OK] Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"  Feature columns: {len(self.feature_columns)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n[OK] Data split completed:")
        print(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"  Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"  Training fraud cases: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
        print(f"  Test fraud cases: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, fit=True):
        """
        Scale features using StandardScaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            fit (bool): Whether to fit the scaler or use existing
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("[OK] Scaler fitted on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        print(f"[OK] Features scaled (mean~0, std~1)")
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance_smote(self, X_train, y_train, sampling_strategy=0.1, random_state=42):
        """
        Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            sampling_strategy (float): Ratio of minority to majority class after resampling
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\nBefore SMOTE:")
        print(f"  Class 0 (Normal): {(y_train == 0).sum():,}")
        print(f"  Class 1 (Fraud): {(y_train == 1).sum():,}")
        print(f"  Ratio: {(y_train == 0).sum()/(y_train == 1).sum():.2f}:1")
        
        # Apply SMOTE
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
        
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X_train.columns
        )
        y_resampled = pd.Series(y_resampled)
        
        print(f"\nAfter SMOTE:")
        print(f"  Class 0 (Normal): {(y_resampled == 0).sum():,}")
        print(f"  Class 1 (Fraud): {(y_resampled == 1).sum():,}")
        print(f"  Ratio: {(y_resampled == 0).sum()/(y_resampled == 1).sum():.2f}:1")
        print(f"  Total samples: {len(y_resampled):,}")
        
        return X_resampled, y_resampled
    
    def handle_imbalance_undersample(self, X_train, y_train, sampling_strategy=0.5, random_state=42):
        """
        Handle class imbalance using Random Under-sampling
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            sampling_strategy (float): Ratio of minority to majority class
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\nBefore Undersampling:")
        print(f"  Class 0 (Normal): {(y_train == 0).sum():,}")
        print(f"  Class 1 (Fraud): {(y_train == 1).sum():,}")
        
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=X_train.columns
        )
        y_resampled = pd.Series(y_resampled)
        
        print(f"\nAfter Undersampling:")
        print(f"  Class 0 (Normal): {(y_resampled == 0).sum():,}")
        print(f"  Class 1 (Fraud): {(y_resampled == 1).sum():,}")
        print(f"  Total samples: {len(y_resampled):,}")
        
        return X_resampled, y_resampled
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[OK] Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        """Load a saved scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"[OK] Scaler loaded from {filepath}")
    
    def preprocess_pipeline(self, df, use_smote=True, smote_ratio=0.1, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            use_smote (bool): Whether to use SMOTE
            smote_ratio (float): SMOTE sampling strategy
            test_size (float): Test set proportion
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # Handle imbalance
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalance_smote(
                X_train_scaled, y_train, smote_ratio, random_state
            )
        else:
            # Use class weights instead (no resampling)
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
            print("\n[WARNING] Using original imbalanced data (class weights will be used in models)")
        
        # Update stored data
        self.X_train = X_train_balanced
        self.X_test = X_test_scaled
        self.y_train = y_train_balanced
        self.y_test = y_test
        
        # Save scaler
        self.save_scaler()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from data_loader import DataLoader
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'creditcard.csv'
    
    # Load data
    loader = DataLoader(data_path=str(data_path))
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data. Exiting.")
        exit(1)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df, use_smote=True)
