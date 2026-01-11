# AI-Based Fraud Detection System

A comprehensive Machine Learning system for detecting fraudulent financial transactions in real-time using multiple ML algorithms.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Project Overview

This fraud detection system uses Machine Learning to identify anomalous or fraudulent transactions by learning patterns from historical data. Unlike traditional rule-based methods, this system continuously improves by learning from new fraud cases.

### Key Objectives

- **Real-time Detection**: Identify fraud as transactions occur
- **High Recall**: Minimize missed fraud cases (false negatives)
- **Multiple Models**: Compare supervised and unsupervised approaches
- **Production Ready**: Deployable web application

## ✨ Features

### 1. Data Loading & Exploration
- Automated data loading and validation
- Comprehensive Exploratory Data Analysis (EDA)
- Class distribution visualization
- Transaction amount analysis
- Time-based pattern detection
- Correlation analysis

### 2. Data Preprocessing
- Feature scaling (StandardScaler)
- Class imbalance handling (SMOTE)
- Train-test splitting with stratification
- Missing value handling

### 3. Model Building
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: High-accuracy ensemble method
- **Isolation Forest**: Unsupervised anomaly detection
- Hyperparameter tuning with GridSearchCV

### 4. Model Evaluation
- Comprehensive metrics:
  - Accuracy
  - Precision
  - Recall (critical for fraud detection)
  - F1-Score
  - ROC-AUC
  - Average Precision
- Confusion matrix visualization
- ROC curve comparison
- Precision-Recall curves

### 5. Real-time Prediction
- Single transaction prediction
- Batch transaction processing
- Confidence scores
- Feature importance explanation

### 6. Web Application
- Streamlit-based interactive UI
- Single transaction input
- CSV batch upload
- Visual results and recommendations

## 🛠 Technology Stack

- **Language**: Python 3.8+
- **Libraries**:
  - `numpy`, `pandas`: Data manipulation
  - `scikit-learn`: Machine Learning models
  - `imbalanced-learn`: SMOTE for class balancing
  - `matplotlib`, `seaborn`: Visualization
  - `streamlit`: Web application framework
- **ML Models**: Logistic Regression, Random Forest, Isolation Forest
- **Environment**: Jupyter Notebook + modular Python scripts

## 📁 Project Structure

```
fraud-detection-system/
│
├── data/
│   └── creditcard.csv          # Dataset file
│
├── notebooks/
│   ├── class_distribution.png   # EDA visualizations
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_*.png   # Model evaluation plots
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── model_comparison.png
│
├── models/
│   ├── logistic_regression.pkl  # Trained models
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl              # Feature scaler
│
├── src/
│   ├── data_loader.py          # Data loading and EDA
│   ├── preprocessor.py         # Data preprocessing
│   ├── model_trainer.py       # Model training
│   ├── model_evaluator.py     # Model evaluation
│   └── predictor.py            # Real-time prediction
│
├── app.py                      # Streamlit web application
├── train_models.py            # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Dataset

### Credit Card Fraud Dataset

The system uses the Credit Card Fraud Detection dataset, which contains:

- **Features**: 
  - `Time`: Seconds elapsed between transaction and first transaction
  - `V1-V28`: PCA-transformed features (anonymized for privacy)
  - `Amount`: Transaction amount
- **Target**: 
  - `Class`: 0 = Normal, 1 = Fraud
- **Characteristics**:
  - Highly imbalanced (fraud cases < 1%)
  - No missing values
  - Pre-processed features (PCA transformation)

### Dataset Source

The dataset can be obtained from:
- [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Or use the provided `creditcard.csv` file

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
# Navigate to project directory
cd fraud-detection-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

Ensure the `creditcard.csv` file is in the `data/` directory:

```bash
# If not already present, copy your dataset
# The dataset should be named creditcard.csv and placed in data/ folder
```

## 📖 Usage

### 1. Train Models

Run the main training script to train all models:

```bash
python train_models.py
```

This will:
- Load and explore the data
- Preprocess features
- Train Logistic Regression, Random Forest, and Isolation Forest
- Evaluate all models
- Save trained models to `models/` directory
- Generate evaluation plots in `notebooks/` directory

### 2. Run Web Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the App:

**Single Transaction:**
1. Navigate to "Single Transaction" page
2. Enter transaction details (Amount, V1-V28 features)
3. Click "Check for Fraud"
4. View prediction, probability scores, and recommendations

**Batch Upload:**
1. Navigate to "Batch Upload" page
2. Upload a CSV file with transaction data
3. Click "Predict Fraud for All Transactions"
4. Download results as CSV

### 3. Use Python API

You can also use the prediction module directly in Python:

```python
from src.predictor import FraudPredictor

# Initialize predictor
predictor = FraudPredictor()
predictor.load_model('models/random_forest.pkl')
predictor.load_scaler('models/scaler.pkl')

# Make prediction
transaction = {
    'Time': 0,
    'Amount': 100.0,
    'V1': -1.36,
    'V2': -0.07,
    # ... other V features
}

result = predictor.predict(transaction)
print(f"Prediction: {result['prediction']}")
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
```

### 4. Run Exploratory Data Analysis

```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.full_exploration()  # Runs complete EDA
```

## 🧠 Model Explanation

### Why Multiple Models?

1. **Logistic Regression**
   - Fast and interpretable
   - Good baseline for comparison
   - Handles class imbalance with class weights

2. **Random Forest**
   - High accuracy through ensemble learning
   - Captures non-linear patterns
   - Provides feature importance

3. **Isolation Forest**
   - Unsupervised anomaly detection
   - Doesn't require labeled fraud cases
   - Good for detecting novel fraud patterns

### Why Recall is Critical

**Recall (True Positive Rate)** measures the proportion of actual fraud cases correctly identified.

- **High Recall** = Fewer fraud cases missed (fewer false negatives)
- **Low Recall** = More fraud cases go undetected (more false negatives)

**In fraud detection:**
- Missing a fraud case (False Negative) is **MUCH MORE COSTLY** than:
  - Flagging a legitimate transaction as fraud (False Positive)
  - False positives can be reviewed manually
  - False negatives result in financial loss

Therefore, we **prioritize RECALL over PRECISION** in fraud detection.

### Model Selection

The best model is typically selected based on:
1. **High Recall** (primary)
2. **Good ROC-AUC** (overall performance)
3. **Balanced Precision** (manageable false positives)
4. **Inference Speed** (for real-time applications)

## 📈 Results

### Expected Performance

After training, you should see results similar to:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.98 | ~0.85 | ~0.75 | ~0.80 | ~0.95 |
| Random Forest | ~0.99 | ~0.90 | ~0.85 | ~0.87 | ~0.98 |
| Isolation Forest | ~0.95 | ~0.60 | ~0.70 | ~0.65 | ~0.85 |

*Note: Actual results may vary based on dataset and hyperparameters*

### Key Insights

1. **Random Forest** typically performs best overall
2. **Logistic Regression** provides good balance of speed and accuracy
3. **Isolation Forest** is useful for detecting novel patterns
4. All models struggle with highly imbalanced data without resampling

## 🔄 Model Improvement

### Continuous Learning

The system can be improved through:

1. **Retraining Strategy**
   - Retrain models periodically with new fraud cases
   - Use incremental learning for large datasets
   - Monitor model performance over time

2. **Concept Drift Handling**
   - Fraud patterns change over time
   - Implement drift detection
   - Retrain when performance degrades

3. **Feedback Loop**
   - Collect false positive/negative cases
   - Use feedback to improve future predictions
   - Update training data regularly

4. **Feature Engineering**
   - Create new features from transaction patterns
   - Time-based features (hour, day of week)
   - Customer behavior features
   - Transaction velocity features

5. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use voting or stacking
   - Improve robustness

## 🚧 Future Improvements

### Short-term
- [ ] Add XGBoost model for better performance
- [ ] Implement model ensemble (voting/stacking)
- [ ] Add more feature engineering
- [ ] Improve Streamlit UI/UX
- [ ] Add model versioning

### Long-term
- [ ] Deep Learning models (Neural Networks)
- [ ] Real-time streaming prediction
- [ ] Database integration
- [ ] API endpoints (FastAPI/Flask)
- [ ] Model monitoring dashboard
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] Explainable AI (SHAP values)

## 📝 Notes

- The dataset features (V1-V28) are PCA-transformed and anonymized
- In production, you would need to implement the same feature engineering pipeline
- Model performance depends heavily on data quality and feature engineering
- Always validate models on unseen data before deployment

## 🤝 Contributing

This is a demonstration project. For production use:
- Add comprehensive testing
- Implement proper logging
- Add error handling
- Include security measures
- Set up CI/CD pipeline

## 📄 License

This project is for educational and demonstration purposes.

## 👤 Author

AI/ML Engineer - Fraud Detection System

---

**Built with ❤️ using Python and scikit-learn**
