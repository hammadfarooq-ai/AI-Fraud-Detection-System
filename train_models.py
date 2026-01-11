"""
Main Script to Train Fraud Detection Models
Run this script to train all models and generate evaluation reports
FAST MODE - Completes in < 1 minute
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

def main():
    """Main training pipeline - FAST MODE"""
    
    start_time = time.time()
    
    print("\n" + "="*80)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE (FAST MODE - Target: < 1 minute)")
    print("="*80)
    
    # Get project root
    project_root = Path(__file__).parent
    data_path = project_root / 'data' / 'creditcard.csv'
    models_dir = project_root / 'models'
    notebooks_dir = project_root / 'notebooks'
    
    # Step 1: Load data
    print("\n[STEP 1/5] Loading data...")
    loader = DataLoader(data_path=str(data_path))
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    # Sample data for faster processing (use 20% of data)
    print("\n  Sampling data for fast processing...")
    original_size = len(df)
    np.random.seed(42)
    sample_size = int(len(df) * 0.2)
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    df = df.iloc[sample_indices].reset_index(drop=True)
    print(f"  Using {len(df):,} samples ({len(df)/original_size*100:.1f}% of original)")
    
    # Step 2: Preprocess data
    print("\n[STEP 2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, 
        use_smote=True, 
        smote_ratio=0.1,
        test_size=0.2,
        random_state=42
    )
    
    # Sample training data for faster training (use 10% of training data)
    print("\n  Sampling training data for faster training...")
    np.random.seed(42)
    train_sample_size = int(len(X_train) * 0.1)
    train_sample_indices = np.random.choice(len(X_train), size=train_sample_size, replace=False)
    X_train = X_train.iloc[train_sample_indices].reset_index(drop=True)
    y_train = y_train.iloc[train_sample_indices].reset_index(drop=True)
    print(f"  Training on {len(X_train):,} samples")
    
    # Sample test data for faster evaluation (use 20% of test data)
    print("\n  Sampling test data for faster evaluation...")
    np.random.seed(42)
    test_sample_size = int(len(X_test) * 0.2)
    test_sample_indices = np.random.choice(len(X_test), size=test_sample_size, replace=False)
    X_test = X_test.iloc[test_sample_indices].reset_index(drop=True)
    y_test = y_test.iloc[test_sample_indices].reset_index(drop=True)
    print(f"  Evaluating on {len(X_test):,} samples")
    
    # Step 3: Train models (FAST MODE)
    print("\n[STEP 3/5] Training models (FAST MODE)...")
    trainer = ModelTrainer(fast_mode=True)
    models = trainer.train_all_models(X_train, y_train, include_isolation=False)
    
    # Step 4: Evaluate models (without plots for speed)
    print("\n[STEP 4/5] Evaluating models (plots disabled for speed)...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(
        models, 
        X_test, 
        y_test, 
        save_plots=False,  # Disable plots for speed
        output_dir=str(notebooks_dir)
    )
    
    # Step 5: Save models
    print("\n[STEP 5/5] Saving models...")
    models_dir.mkdir(exist_ok=True)
    trainer.save_all_models(base_path=str(models_dir))
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nModels saved to: models/")
    print("Evaluation results available (plots disabled for speed)")
    print("\nTo use the models, run: streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    main()
