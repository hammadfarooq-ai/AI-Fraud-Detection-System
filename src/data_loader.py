"""
Data Loading and Exploration Module
Handles loading the credit card fraud dataset and performing initial EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class DataLoader:
    """Class to load and explore the credit card fraud dataset"""
    
    def __init__(self, data_path='data/creditcard.csv', output_dir=None):
        """
        Initialize the DataLoader
        
        Args:
            data_path (str): Path to the CSV file
            output_dir (str, optional): Directory to save output plots. If None, uses notebooks/ directory relative to project root
        """
        self.data_path = data_path
        self.df = None
        
        # Determine output directory for plots
        if output_dir is None:
            # Try to find project root (look for notebooks directory or go up from current location)
            current_path = Path.cwd()
            if 'notebooks' in current_path.parts:
                # We're in notebooks directory, go up one level
                project_root = current_path.parent
            else:
                # We're at project root
                project_root = current_path
            self.output_dir = project_root / 'notebooks'
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load the dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[OK] Dataset loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"[ERROR] File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Error loading data: {str(e)}")
            return None
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("DATASET BASIC INFORMATION")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1] - 1}")  # Excluding target
        print(f"Number of Samples: {self.df.shape[0]:,}")
        
        print("\n" + "-"*60)
        print("Column Names:")
        print("-"*60)
        print(self.df.columns.tolist())
        
        print("\n" + "-"*60)
        print("Data Types:")
        print("-"*60)
        print(self.df.dtypes)
        
        print("\n" + "-"*60)
        print("Missing Values:")
        print("-"*60)
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("[OK] No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\n" + "-"*60)
        print("First 5 Rows:")
        print("-"*60)
        print(self.df.head())
        
        print("\n" + "-"*60)
        print("Statistical Summary:")
        print("-"*60)
        print(self.df.describe())
    
    def explore_class_distribution(self):
        """Explore the distribution of fraud vs non-fraud transactions"""
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        class_counts = self.df['Class'].value_counts()
        class_percentages = self.df['Class'].value_counts(normalize=True) * 100
        
        print(f"\nNon-Fraud (Class 0): {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
        print(f"Fraud (Class 1): {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
        print(f"\nClass Imbalance Ratio: {class_counts[0]/class_counts[1]:.2f}:1")
        
        # Visualize class distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        plt.xlabel('Class (0=Normal, 1=Fraud)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        for i, v in enumerate(class_counts):
            plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        class_percentages.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        plt.xlabel('Class (0=Normal, 1=Fraud)', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks(rotation=0)
        for i, v in enumerate(class_percentages):
            plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'class_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to {output_path}")
        plt.show()
    
    def explore_amount_distribution(self):
        """Explore transaction amount distribution"""
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("TRANSACTION AMOUNT ANALYSIS")
        print("="*60)
        
        print(f"\nAmount Statistics:")
        print(f"  Mean: ${self.df['Amount'].mean():.2f}")
        print(f"  Median: ${self.df['Amount'].median():.2f}")
        print(f"  Std: ${self.df['Amount'].std():.2f}")
        print(f"  Min: ${self.df['Amount'].min():.2f}")
        print(f"  Max: ${self.df['Amount'].max():.2f}")
        
        # Compare amounts for fraud vs non-fraud
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount']
        normal_amounts = self.df[self.df['Class'] == 0]['Amount']
        
        print(f"\nFraud Transactions:")
        print(f"  Mean: ${fraud_amounts.mean():.2f}")
        print(f"  Median: ${fraud_amounts.median():.2f}")
        print(f"  Max: ${fraud_amounts.max():.2f}")
        
        print(f"\nNormal Transactions:")
        print(f"  Mean: ${normal_amounts.mean():.2f}")
        print(f"  Median: ${normal_amounts.median():.2f}")
        print(f"  Max: ${normal_amounts.max():.2f}")
        
        # Visualize amount distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.df['Amount'], bins=50, color='#3498db', edgecolor='black')
        plt.title('Transaction Amount Distribution (All)', fontsize=12, fontweight='bold')
        plt.xlabel('Amount ($)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.hist(normal_amounts, bins=50, color='#3498db', alpha=0.7, label='Normal', edgecolor='black')
        plt.hist(fraud_amounts, bins=50, color='#e74c3c', alpha=0.7, label='Fraud', edgecolor='black')
        plt.title('Amount Distribution by Class', fontsize=12, fontweight='bold')
        plt.xlabel('Amount ($)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.boxplot([normal_amounts, fraud_amounts], labels=['Normal', 'Fraud'])
        plt.title('Amount Distribution (Box Plot)', fontsize=12, fontweight='bold')
        plt.ylabel('Amount ($)', fontsize=10)
        plt.yscale('log')
        
        plt.tight_layout()
        output_path = self.output_dir / 'amount_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to {output_path}")
        plt.show()
    
    def explore_time_distribution(self):
        """Explore time-based patterns"""
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("TIME-BASED ANALYSIS")
        print("="*60)
        
        # Convert time to hours (assuming seconds from first transaction)
        self.df['Hour'] = (self.df['Time'] / 3600) % 24
        
        fraud_by_hour = self.df[self.df['Class'] == 1].groupby('Hour')['Class'].count()
        normal_by_hour = self.df[self.df['Class'] == 0].groupby('Hour')['Class'].count()
        
        plt.figure(figsize=(12, 5))
        plt.plot(fraud_by_hour.index, fraud_by_hour.values, 
                marker='o', color='#e74c3c', label='Fraud', linewidth=2)
        plt.plot(normal_by_hour.index, normal_by_hour.values, 
                marker='o', color='#3498db', label='Normal', alpha=0.5, linewidth=2)
        plt.title('Transaction Frequency by Hour of Day', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Transactions', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = self.output_dir / 'time_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to {output_path}")
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation with target
        correlations = self.df.corr()['Class'].sort_values(ascending=False)
        
        print("\nTop 10 Features Correlated with Fraud (Class):")
        print(correlations.head(11))  # Including Class itself
        
        # Visualize correlation heatmap for top features
        top_features = correlations.abs().nlargest(11).index.tolist()
        corr_matrix = self.df[top_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap (Top Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to {output_path}")
        plt.show()
    
    def full_exploration(self):
        """Run complete exploratory data analysis"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        self.load_data()
        self.basic_info()
        self.explore_class_distribution()
        self.explore_amount_distribution()
        self.explore_time_distribution()
        self.correlation_analysis()
        
        print("\n" + "="*80)
        print("EDA COMPLETE!")
        print("="*80)
        
        return self.df


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'creditcard.csv'
    output_dir = project_root / 'notebooks'
    
    loader = DataLoader(data_path=str(data_path), output_dir=str(output_dir))
    df = loader.full_exploration()
