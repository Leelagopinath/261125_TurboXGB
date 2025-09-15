# utils.py
import pandas as pd
import streamlit as st
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Function to load data from various formats
def load_data(uploaded_file):
    """Loads data from an uploaded file into a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, JSON, or Parquet.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to get the next available version number for a file
def get_next_version(directory, prefix="dataset_v"):
    """Finds the next version number for a file in a given directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        return 1
    
    max_version = 0
    for f in files:
        try:
            version = int(f.replace(prefix, '').replace('.csv', ''))
            if version > max_version:
                max_version = version
        except ValueError:
            continue
    return max_version + 1

# Functions to calculate evaluation metrics
def calculate_classification_metrics(y_true, y_pred, y_prob):
    """Calculates and returns a dictionary of classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-Score": f1_score(y_true, y_pred, average='weighted')
    }
    # AUC requires probabilities and is only for binary classification
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of regression metrics."""
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R-squared": r2_score(y_true, y_pred)
    }
    return metrics