"""
Model Evaluation Module
Comprehensive evaluation of fraud detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import os

class ModelEvaluator:
    """Class to evaluate fraud detection models"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics
        
        Args:
            metrics (dict): Evaluation metrics dictionary
        """
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {metrics['model_name']}")
        print("="*60)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"Avg Precision: {metrics['average_precision']:.4f}")
        
        print("\n" + "-"*60)
        print("Confusion Matrix:")
        print("-"*60)
        cm = metrics['confusion_matrix']
        print(f"                Predicted")
        print(f"              Normal  Fraud")
        print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"       Fraud    {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        # Note: y_test should be passed separately or stored in metrics
        if 'y_test' in metrics:
            print(classification_report(
                metrics['y_test'],
                metrics['y_pred'],
                target_names=['Normal', 'Fraud']
            ))
        else:
            print("(y_test not available for classification report)")
        
        # Explain why recall is important
        print("\n" + "="*60)
        print("WHY RECALL IS CRITICAL IN FRAUD DETECTION:")
        print("="*60)
        print("""
        Recall (True Positive Rate) measures the proportion of actual fraud cases
        that were correctly identified. In fraud detection:
        
        - HIGH RECALL = Fewer fraud cases missed (fewer false negatives)
        - LOW RECALL = More fraud cases go undetected (more false negatives)
        
        Missing a fraud case (False Negative) is MUCH MORE COSTLY than:
        - Flagging a legitimate transaction as fraud (False Positive)
        - False positives can be reviewed manually
        - False negatives result in financial loss and customer impact
        
        Therefore, we prioritize RECALL over PRECISION in fraud detection.
        A model with high recall catches more fraud, even if it means more false alarms.
        """)
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            metrics (dict): Evaluation metrics
            save_path (str): Path to save the plot
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {metrics["model_name"]}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, metrics_list, y_test=None, save_path=None):
        """
        Plot ROC curves for multiple models
        
        Args:
            metrics_list (list): List of metrics dictionaries
            y_test (pd.Series): True labels (optional, can be in metrics)
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for metrics in metrics_list:
            # Get true labels from metrics or parameter
            y_true = metrics.get('y_test', y_test)
            if y_true is None or len(y_true) == 0:
                print(f"Warning: y_test not available for {metrics['model_name']}, skipping ROC curve")
                continue
                
            fpr, tpr, _ = roc_curve(y_true, metrics['y_pred_proba'])
            roc_auc = metrics['roc_auc']
            
            plt.plot(fpr, tpr, 
                    label=f"{metrics['model_name']} (AUC = {roc_auc:.4f})",
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, metrics_list, y_test=None, save_path=None):
        """
        Plot Precision-Recall curves
        
        Args:
            metrics_list (list): List of metrics dictionaries
            y_test (pd.Series): True labels (optional, can be in metrics)
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for metrics in metrics_list:
            # Get true labels from metrics or parameter
            y_true = metrics.get('y_test', y_test)
            if y_true is None or len(y_true) == 0:
                print(f"Warning: y_test not available for {metrics['model_name']}, skipping PR curve")
                continue
                
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_pred_proba'])
            avg_precision = metrics['average_precision']
            
            plt.plot(recall, precision,
                    label=f"{metrics['model_name']} (AP = {avg_precision:.4f})",
                    linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, metrics_list, save_path=None):
        """
        Compare multiple models side by side
        
        Args:
            metrics_list (list): List of metrics dictionaries
            save_path (str): Path to save the comparison table
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison_data = []
        for metrics in metrics_list:
            comparison_data.append({
                'Model': metrics['model_name'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Avg Precision': f"{metrics['average_precision']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Visual comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']
        
        for idx, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
            ax = axes[idx // 3, idx % 3]
            values = [m[metric] for m in metrics_list]
            names = [m['model_name'] for m in metrics_list]
            
            bars = ax.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(names)])
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Comparison chart saved to {save_path}")
        
        plt.show()
        
        if save_path:
            # Save comparison table to CSV
            csv_path = save_path.replace('.png', '.csv')
            comparison_df.to_csv(csv_path, index=False)
            print(f"[OK] Comparison table saved to {csv_path}")
    
    def evaluate_all_models(self, models, X_test, y_test, save_plots=True, output_dir=None):
        """
        Evaluate all models and generate comprehensive reports
        
        Args:
            models (dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_plots (bool): Whether to save plots
            output_dir (str, optional): Directory to save plots. If None, uses notebooks/ relative to project root
            
        Returns:
            dict: All evaluation results
        """
        from pathlib import Path
        
        # Determine output directory for plots
        if output_dir is None:
            # Try to find project root
            try:
                # If running from src directory, go up one level
                current_path = Path.cwd()
                if current_path.name == 'src':
                    project_root = current_path.parent
                else:
                    project_root = current_path
                output_dir = project_root / 'notebooks'
            except:
                output_dir = Path('notebooks')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        metrics_list = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            metrics['y_test'] = y_test  # Store for plotting
            metrics_list.append(metrics)
            self.print_metrics(metrics)
            
            if save_plots:
                self.plot_confusion_matrix(
                    metrics, 
                    str(output_dir / f'confusion_matrix_{model_name}.png')
                )
        
        # Compare all models
        if len(metrics_list) > 1:
            self.compare_models(
                metrics_list,
                str(output_dir / 'model_comparison.png') if save_plots else None
            )
            
            if save_plots:
                self.plot_roc_curve(metrics_list, y_test, str(output_dir / 'roc_curves.png'))
                self.plot_precision_recall_curve(
                    metrics_list, 
                    y_test,
                    str(output_dir / 'pr_curves.png')
                )
        
        return self.results


if __name__ == "__main__":
    # Example usage - FAST MODE (completes in < 1 minute)
    import time
    from pathlib import Path
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from model_trainer import ModelTrainer
    
    start_time = time.time()
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'creditcard.csv'
    
    print("\n" + "="*80)
    print("FAST MODE - COMPLETE PIPELINE (Target: < 1 minute)")
    print("="*80)
    
    # Load data
    print("\n[STEP 1/5] Loading data...")
    loader = DataLoader(data_path=str(data_path))
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data. Exiting.")
        exit(1)
    
    # Sample data for faster processing (use 20% of data)
    print("\n[STEP 2/5] Sampling data for fast processing...")
    np.random.seed(42)
    sample_size = int(len(df) * 0.2)
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    df = df.iloc[sample_indices].reset_index(drop=True)
    print(f"  Using {len(df):,} samples ({sample_size/len(df)*100:.1f}% of original)")
    
    # Preprocess
    print("\n[STEP 3/5] Preprocessing data...")
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
    
    # Sample test data for faster evaluation (use 20% of test data)
    print("\n  Sampling test data for faster evaluation...")
    np.random.seed(42)
    test_sample_size = int(len(X_test) * 0.2)
    test_sample_indices = np.random.choice(len(X_test), size=test_sample_size, replace=False)
    X_test = X_test.iloc[test_sample_indices].reset_index(drop=True)
    y_test = y_test.iloc[test_sample_indices].reset_index(drop=True)
    print(f"  Evaluating on {len(X_test):,} samples")
    
    # Train models in fast mode
    print("\n[STEP 4/5] Training models (FAST MODE)...")
    trainer = ModelTrainer(fast_mode=True)
    models = trainer.train_all_models(X_train, y_train, include_isolation=False)
    
    # Save models
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    trainer.save_all_models(base_path=str(models_dir))
    
    # Evaluate models (without plots for speed)
    print("\n[STEP 5/5] Evaluating models (plots disabled for speed)...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test, save_plots=False)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"⏱️  TOTAL TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*80)
