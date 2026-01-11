"""
Streamlit App for Fraud Detection System
Interactive web interface for real-time fraud prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from predictor import FraudPredictor

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud-box {
        background-color: #ffebee;
        border: 2px solid #e74c3c;
    }
    .normal-box {
        background-color: #e8f5e9;
        border: 2px solid #2ecc71;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the fraud predictor (cached)"""
    try:
        predictor = FraudPredictor()
        predictor.load_model()
        predictor.load_scaler()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔒 AI Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Single Transaction", "Batch Upload", "About"]
    )
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("""
        ⚠️ **Model not found!** 
        
        Please train the models first by running:
        ```bash
        python train_models.py
        ```
        """)
        return
    
    if page == "Single Transaction":
        single_transaction_page(predictor)
    elif page == "Batch Upload":
        batch_upload_page(predictor)
    elif page == "About":
        about_page()

def single_transaction_page(predictor):
    """Single transaction prediction page"""
    
    st.header("📊 Single Transaction Prediction")
    st.markdown("Enter transaction details to check for fraud")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        # Create input form
        with st.form("transaction_form"):
            # Amount
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=100.0,
                step=0.01,
                help="Enter the transaction amount"
            )
            
            # V1-V28 features (simplified - in real app, these would come from feature engineering)
            st.markdown("**Feature Values (V1-V28)**")
            st.caption("These are PCA-transformed features. Enter values or use defaults.")
            
            # Create a simplified input - in practice, you'd have all V1-V28
            # For demo, we'll use a few key features
            feature_inputs = {}
            
            # Key features that are most important
            key_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V10', 'V11', 'V12', 'V14', 'V17']
            
            cols = st.columns(5)
            for idx, feat in enumerate(key_features):
                with cols[idx % 5]:
                    feature_inputs[feat] = st.number_input(
                        feat,
                        value=0.0,
                        step=0.01,
                        key=f"input_{feat}"
                    )
            
            # Time (optional)
            time = st.number_input(
                "Time (seconds from first transaction)",
                min_value=0.0,
                value=0.0,
                step=1.0
            )
            
            submitted = st.form_submit_button("🔍 Check for Fraud", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        if submitted:
            # Prepare transaction data
            transaction_data = {
                'Time': time,
                'Amount': amount,
                **{k: v for k, v in feature_inputs.items()}
            }
            
            # Fill missing V features with 0
            for i in range(1, 29):
                v_name = f'V{i}'
                if v_name not in transaction_data:
                    transaction_data[v_name] = 0.0
            
            try:
                # Make prediction
                with st.spinner("Analyzing transaction..."):
                    result = predictor.predict(transaction_data)
                
                # Display result
                is_fraud = result['is_fraud']
                fraud_prob = result['fraud_probability']
                confidence = result['confidence']
                
                # Styling based on prediction
                if is_fraud:
                    st.markdown(f"""
                    <div class="prediction-box fraud-box">
                        <h2 style="color: #e74c3c;">⚠️ FRAUD DETECTED</h2>
                        <p style="font-size: 1.2rem;">This transaction has been flagged as potentially fraudulent.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box normal-box">
                        <h2 style="color: #2ecc71;">✓ NORMAL TRANSACTION</h2>
                        <p style="font-size: 1.2rem;">This transaction appears to be legitimate.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                st.markdown("### Prediction Metrics")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_prob:.2%}",
                        delta=None
                    )
                
                with metric_col2:
                    st.metric(
                        "Normal Probability",
                        f"{result['normal_probability']:.2%}",
                        delta=None
                    )
                
                with metric_col3:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}",
                        delta=None
                    )
                
                # Progress bar for fraud probability
                st.markdown("### Fraud Risk Level")
                st.progress(fraud_prob)
                
                # Explanation (if available)
                if hasattr(predictor.model, 'feature_importances_'):
                    with st.expander("🔍 View Prediction Explanation"):
                        explanation = predictor.explain_prediction(transaction_data)
                        if 'top_features' in explanation:
                            st.markdown("**Top Contributing Features:**")
                            for feat in explanation['top_features']:
                                st.write(f"- **{feat['feature']}**: {feat['value']:.4f} (importance: {feat['importance']:.4f})")
                
                # Recommendations
                st.markdown("### Recommendations")
                if is_fraud or fraud_prob > 0.7:
                    st.warning("""
                    **Recommended Actions:**
                    - Review transaction details manually
                    - Contact customer for verification
                    - Consider blocking the transaction
                    - Flag account for additional monitoring
                    """)
                elif fraud_prob > 0.3:
                    st.info("""
                    **Recommended Actions:**
                    - Monitor this transaction
                    - Review customer's transaction history
                    - Consider additional verification
                    """)
                else:
                    st.success("""
                    **Transaction Status:**
                    - Low fraud risk detected
                    - Proceed with normal processing
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Fill in the transaction details and click 'Check for Fraud' to get a prediction")

def batch_upload_page(predictor):
    """Batch transaction prediction page"""
    
    st.header("📁 Batch Transaction Prediction")
    st.markdown("Upload a CSV file with multiple transactions to check for fraud")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should contain transaction data with columns: Time, Amount, V1-V28"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File loaded successfully! ({len(df)} transactions)")
            
            # Display preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10))
            
            # Check required columns
            required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
                st.info("Missing columns will be filled with default values (0)")
            else:
                st.success("✓ All required columns present")
            
            # Predict button
            if st.button("🔍 Predict Fraud for All Transactions", use_container_width=True):
                with st.spinner("Processing transactions..."):
                    # Make predictions
                    predictions = predictor.predict_batch(df)
                    
                    # Combine with original data
                    results_df = pd.concat([df, predictions], axis=1)
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(results_df))
                    with col2:
                        fraud_count = results_df['is_fraud'].sum()
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        fraud_rate = fraud_count / len(results_df) * 100
                        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                    with col4:
                        avg_confidence = results_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    # Results table
                    st.dataframe(
                        results_df[['Amount', 'prediction_label', 'fraud_probability', 'confidence']],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.markdown("### Visualizations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        results_df['prediction_label'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                        ax.set_title('Fraud vs Normal Predictions')
                        ax.set_xlabel('Prediction')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=0)
                        st.pyplot(fig)
                    
                    with viz_col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(results_df['fraud_probability'], bins=50, color='#3498db', edgecolor='black')
                        ax.set_title('Fraud Probability Distribution')
                        ax.set_xlabel('Fraud Probability')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Sample file format
        with st.expander("📝 Sample CSV Format"):
            st.code("""
Time,Amount,V1,V2,V3,...,V28
0,149.62,-1.36,-0.07,2.54,...,-0.02
1,2.69,1.19,0.27,0.17,...,0.01
...
            """)

def about_page():
    """About page"""
    
    st.header("📖 About This System")
    
    st.markdown("""
    ## AI-Based Fraud Detection System
    
    This system uses Machine Learning to detect fraudulent financial transactions in real-time.
    It employs multiple ML algorithms to identify patterns and anomalies that indicate fraud.
    
    ### Features
    
    - **Real-time Prediction**: Instant fraud detection for individual transactions
    - **Batch Processing**: Analyze multiple transactions at once
    - **Multiple ML Models**: Uses Logistic Regression, Random Forest, and Isolation Forest
    - **High Accuracy**: Trained on historical transaction data
    - **Explainable**: Provides feature importance and confidence scores
    
    ### Models Used
    
    1. **Logistic Regression**: Fast, interpretable baseline model
    2. **Random Forest**: Ensemble method with high accuracy
    3. **Isolation Forest**: Unsupervised anomaly detection
    
    ### Key Metrics
    
    - **Recall**: Critical for fraud detection - minimizes missed fraud cases
    - **Precision**: Balances false positives
    - **ROC-AUC**: Overall model performance
    - **F1-Score**: Harmonic mean of precision and recall
    
    ### How It Works
    
    1. **Data Preprocessing**: Features are scaled and balanced
    2. **Model Training**: Multiple models are trained and compared
    3. **Prediction**: New transactions are analyzed using the best model
    4. **Risk Assessment**: Probability scores indicate fraud likelihood
    
    ### Model Improvement
    
    The system can continuously learn from new data:
    - **Retraining**: Models can be retrained with new fraud cases
    - **Concept Drift**: Handles changing fraud patterns over time
    - **Feedback Loop**: False positives/negatives can improve future predictions
    
    ### Technology Stack
    
    - Python 3.8+
    - scikit-learn
    - pandas, numpy
    - Streamlit (Web Interface)
    - SMOTE (Class Balancing)
    
    ### Usage
    
    1. **Single Transaction**: Enter transaction details manually
    2. **Batch Upload**: Upload CSV file with multiple transactions
    3. **Review Results**: Check predictions and confidence scores
    4. **Take Action**: Follow recommendations based on risk level
    
    ---
    
    **Note**: This is a demonstration system. In production, additional security measures
    and validation would be required.
    """)

if __name__ == "__main__":
    main()
