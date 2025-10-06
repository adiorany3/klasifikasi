import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from joblib import Parallel, delayed
from fpdf import FPDF
import io
import base64
from datetime import datetime

@st.cache_data
def load_and_prepare_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

@st.cache_data
def process_features(df, selected_features):
    # Memilih fitur dan melakukan preprocessing
    X = df[selected_features]
    y = df['BIP']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Categorize target
    bins = [float('-inf'), 300, 325, 350, 400, float('inf')]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    y_cat = pd.cut(y, bins=bins, labels=labels, right=True)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cat)
    
    return X_scaled, y_encoded, y_cat, le

@st.cache_data
def calculate_correlations(df, selected_features):
    correlations = pd.DataFrame({
        'Feature': selected_features,
        'Correlation with BIP': [df[feature].corr(df['BIP']) 
                                     for feature in selected_features]
    }).sort_values('Correlation with BIP', key=abs, ascending=False)
    return correlations

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, X, y, cv_folds):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Cross-validation
    acc_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    prec_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='precision_weighted')
    rec_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='recall_weighted')
    f1_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted')
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "Accuracy": acc,
        "Accuracy Std": acc_cv.std(),
        "Precision": prec,
        "Precision Std": prec_cv.std(),
        "Recall": rec,
        "Recall Std": rec_cv.std(),
        "F1-score": f1,
        "F1-score Std": f1_cv.std()
    }, cm

def evaluate_predictions(y_true, y_pred, le):
    """Evaluasi detail untuk prediksi model"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Convert numeric predictions back to labels
    y_true_labels = le.inverse_transform(y_true)
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual IP': y_true_labels,
        'Predicted IP': y_pred_labels,
        'Correct': y_true_labels == y_pred_labels
    })
    
    return {
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        },
        'comparison': comparison_df
    }

# Modifikasi fungsi create_pdf_report
def create_pdf_report(results_df, comprehensive_results, best_model_name, best_predictions, 
                     class_metrics_df, feature_importance=None, figures=None):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Menggunakan font default
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'IP Classification Analysis Report', 0, 1, 'C')
        pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        
        # Model Evaluation Results
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Model Evaluation Results', 0, 1, 'L')
        
        # Convert DataFrame to table
        pdf.set_font('Arial', '', 8)
        
        # Menambahkan hasil evaluasi model
        pdf.cell(0, 10, "Model Evaluation Summary", 0, 1, 'L')
        cols_width = pdf.w / len(results_df.columns)
        
        # Header
        for col in results_df.columns:
            pdf.cell(cols_width, 7, str(col), 1)
        pdf.ln()
        
        # Data
        for i in range(len(results_df)):
            for col in results_df.columns:
                value = results_df.iloc[i][col]
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                pdf.cell(cols_width, 7, text, 1)
            pdf.ln()
        
        # Best Model Results
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Best Model: {best_model_name}', 0, 1, 'L')
        
        # Summary statistics
        pdf.set_font('Arial', '', 10)
        correct = best_predictions['Correct'].sum()
        total = len(best_predictions)
        accuracy = (correct / total) * 100
        
        pdf.cell(0, 7, f'Total Predictions: {total}', 0, 1)
        pdf.cell(0, 7, f'Correct Predictions: {correct}', 0, 1)
        pdf.cell(0, 7, f'Accuracy: {accuracy:.2f}%', 0, 1)
        
        # Class Metrics
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Class Performance Metrics', 0, 1, 'L')
        
        # Add class metrics table
        cols_width = pdf.w / len(class_metrics_df.columns)
        
        for col in class_metrics_df.columns:
            pdf.cell(cols_width, 7, str(col), 1)
        pdf.ln()
        
        for i in range(len(class_metrics_df)):
            for col in class_metrics_df.columns:
                value = class_metrics_df.iloc[i][col]
                if isinstance(value, float):  # Fixed the missing parenthesis
                    text = f"{value:.2f}%"
                else:
                    text = str(value)
                pdf.cell(cols_width, 7, text, 1)
            pdf.ln()
        
        # Feature Importance if available
        if feature_importance is not None:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Feature Importance Analysis', 0, 1, 'L')
            
            cols_width = pdf.w / 2  # Two columns: Feature and Importance
            
            # Header
            pdf.cell(cols_width, 7, 'Feature', 1)
            pdf.cell(cols_width, 7, 'Importance', 1)
            pdf.ln()
            
            # Data
            for i in range(len(feature_importance)):
                pdf.cell(cols_width, 7, str(feature_importance.iloc[i]['Feature']), 1)
                pdf.cell(cols_width, 7, f"{feature_importance.iloc[i]['Importance']:.4f}", 1)
                pdf.ln()
        
        return pdf
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def main():
    st.title("IP Classification Analysis")
    
    uploaded_file = st.file_uploader("Upload gabungan.xlsx file", type=["xlsx"])
    if uploaded_file is None:
        st.info("Please upload the gabungan.xlsx file to start the analysis.")
        return
    
    # Load data dengan caching
    df = load_and_prepare_data(uploaded_file)
    
    # Data cleaning info
    st.write("Data Info Before Cleaning:")
    st.write(f"Total rows before cleaning: {len(df)}")
    st.write("Missing values before cleaning:", df.isnull().sum())
    
    # Clean data
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)
    
    # Selected features
    selected_features = [
        'Age', 'FeedIntake', 'Culling', 'Death', 
        'Depletion', 'LiveBirdAfter', '%Live', 
        'LiveWeightBird', 'ADG', 'FCR', 'Temp', 
        'Hum', 'THI', 'WCI'
    ]
    
    # Validate features
    missing_columns = [col for col in selected_features if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return
    
    # Process features with caching
    X_scaled, y_encoded, y_cat, le = process_features(df, selected_features)
    
    # Calculate correlations with caching
    correlations = calculate_correlations(df, selected_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', n_jobs=-1),
        'CatBoost': CatBoostClassifier(verbose=0, thread_count=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'HGB': HistGradientBoostingClassifier()
    }
    
    # Parallel model training and evaluation
    cv_folds = min(5, pd.Series(y_encoded).value_counts().min())
    results = []
    confusion_matrices = {}
    
    with st.spinner('Training models in parallel...'):
        model_results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_model)(
                model, X_train, X_test, y_train, y_test, X_scaled, y_encoded, cv_folds
            ) for model in models.values()
        )
    
    # Process results
    for (name, _), (metrics, cm) in zip(models.items(), model_results):
        metrics["Model"] = name
        results.append(metrics)
        confusion_matrices[name] = cm
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df["Rank"] = results_df["Accuracy"].rank(ascending=False, method="min").astype(int)
    results_df = results_df.sort_values("Rank")
    
    # Display results and visualizations
    st.subheader("Model Evaluation Results")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1-score": "{:.4f}"
    }))
    
    # Evaluasi dan ranking model
    model_predictions = {}
    model_scores = {}
    
    st.subheader("Model Performance Analysis")
    
    for name, model in models.items():
        # Train model
        with st.spinner(f'Training {name}...'):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            eval_results = evaluate_predictions(y_test, y_pred, le)
            
            model_predictions[name] = eval_results['comparison']
            model_scores[name] = eval_results['metrics']
    
    # Create comprehensive results DataFrame
    comprehensive_results = pd.DataFrame(model_scores).T
    comprehensive_results.columns = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    # Calculate average score for ranking
    comprehensive_results['Average Score'] = comprehensive_results.mean(axis=1)
    comprehensive_results['Rank'] = comprehensive_results['Average Score'].rank(ascending=False, method='min')
    comprehensive_results = comprehensive_results.sort_values('Rank')
    
    # Display results
    st.write("### Model Rankings")
    st.dataframe(comprehensive_results.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-score': '{:.4f}',
        'Average Score': '{:.4f}',
        'Rank': '{:.0f}'
    }))
    
    # Get best model
    best_model_name = comprehensive_results.index[0]
    best_model = models[best_model_name]
    
    # Detailed analysis of best model
    st.write(f"### Best Model Analysis: {best_model_name}")
    best_predictions = model_predictions[best_model_name]
    
    # Confusion Matrix for best model
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(
        y_test, 
        best_model.predict(X_test),
        labels=range(len(le.classes_))
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Prediction Analysis
    st.write("#### Prediction Analysis")
    prediction_stats = best_predictions['Correct'].value_counts()
    
    plt.figure(figsize=(8, 6))
    prediction_stats.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Prediction Accuracy Distribution - {best_model_name}')
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Detailed Predictions Table
    st.write("#### Detailed Predictions")
    
    # Tampilkan prediksi tanpa styling terlebih dahulu
    st.dataframe(best_predictions)
    
    # Tampilkan ringkasan prediksi dalam format yang lebih sederhana
    st.write("#### Prediction Results Summary")
    summary_df = pd.DataFrame({
        'Category': ['Correct Predictions', 'Incorrect Predictions', 'Total Predictions'],
        'Count': [
            best_predictions['Correct'].sum(),
            len(best_predictions) - best_predictions['Correct'].sum(),
            len(best_predictions)
        ],
        'Percentage': [
            best_predictions['Correct'].mean() * 100,
            (1 - best_predictions['Correct'].mean()) * 100,
            100
        ]
    })
    
    st.dataframe(summary_df.style.format({
        'Count': '{:,.0f}',
        'Percentage': '{:.2f}%'
    }))

    # Confusion Matrix Visualization
    st.write("#### Confusion Matrix Visualization")
    confusion = pd.crosstab(
        best_predictions['Actual IP'],
        best_predictions['Predicted IP'],
        margins=True
    )
    st.dataframe(confusion)

    # Class Distribution Plot
    st.write("#### Class Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Actual Distribution
    sns.countplot(data=best_predictions, x='Actual IP', ax=ax1)
    ax1.set_title('Actual Class Distribution')
    ax1.tick_params(axis='x', rotation=45)

    # Predicted Distribution
    sns.countplot(data=best_predictions, x='Predicted IP', ax=ax2)
    ax2.set_title('Predicted Class Distribution')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # Accuracy Metrics per Class
    st.write("#### Performance Metrics per Class")
    class_metrics = []
    
    for class_name in best_predictions['Actual IP'].unique():
        class_data = best_predictions[best_predictions['Actual IP'] == class_name]
        metrics = {
            'Class': class_name,
            'Total Samples': len(class_data),
            'Correct Predictions': class_data['Correct'].sum(),
            'Accuracy': class_data['Correct'].mean() * 100
        }
        class_metrics.append(metrics)
    
    class_metrics_df = pd.DataFrame(class_metrics)
    st.dataframe(class_metrics_df.style.format({
        'Total Samples': '{:,.0f}',
        'Correct Predictions': '{:,.0f}',
        'Accuracy': '{:.2f}%'
    }))

    # Feature Importance Analysis (if available)
    if hasattr(best_model, "feature_importances_"):
        st.write("#### Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Feature importance table
        st.dataframe(feature_importance.style.format({
            'Importance': '{:.4f}'
        }))

    # Correlation Analysis
    st.write("#### Feature Correlation Analysis")
    correlation_matrix = pd.DataFrame(X_train, columns=selected_features).corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Add download button for PDF report
    st.write("### Download Complete Report")
    
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf = create_pdf_report(
                    results_df=results_df,
                    comprehensive_results=comprehensive_results,
                    best_model_name=best_model_name,
                    best_predictions=best_predictions,
                    class_metrics_df=class_metrics_df,
                    feature_importance=feature_importance if hasattr(best_model, "feature_importances_") else None
                )
                
                if pdf is not None:
                    try:
                        # Convert PDF to bytes using a binary string format
                        pdf_data = pdf.output(dest='S').encode('latin1')
                        
                        # Create download button with properly formatted binary data
                        st.download_button(
                            label="Download PDF Report",
                            data=bytes(pdf_data),  # Convert to bytes explicitly
                            file_name=f"ip_classification_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="pdf_download"  # Add a unique key
                        )
                        
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error saving PDF: {str(e)}")
            except Exception as e:
                st.error(f"Error in PDF generation: {str(e)}")
    
    # Add download buttons for individual data tables
    st.write("### Download Individual Data")
    
    # Model Evaluation Results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Model Evaluation Results",
        data=csv,
        file_name=f"model_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Best Model Predictions
    csv = best_predictions.to_csv(index=False)
    st.download_button(
        label="Download Best Model Predictions",
        data=csv,
        file_name=f"best_model_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Class Metrics
    csv = class_metrics_df.to_csv(index=False)
    st.download_button(
        label="Download Class Metrics",
        data=csv,
        file_name=f"class_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    if hasattr(best_model, "feature_importances_"):
        # Feature Importance
        csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance",
            data=csv,
            file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()