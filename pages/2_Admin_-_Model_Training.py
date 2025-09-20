import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import json
from datetime import datetime

# Import all model classes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from xgboost.callback import EarlyStopping

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from utils import calculate_classification_metrics, calculate_regression_metrics
import auth
import seaborn as sns

# --- Page Setup and Authentication ---
auth.login()
st.set_page_config(page_title="Universal Model Trainer", layout="wide")

if st.session_state.get('authenticated'):
    if st.session_state.get('role') != 'admin':
        st.error("You do not have permission to view this page.")
        st.stop()

    st.title("Step 2: Universal Model Trainer (Admin)")

    MODELS_DIR = "./project_data/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    if 'X_train' not in st.session_state:
        st.error("Data not prepared. Please go to the 'Data Upload' page first.")
        st.stop()

    # Define task_type from session_state or default to Classification
    task_type = st.session_state.get('task_type', "Classification")

    # --- Primary Algorithm Selection ---
    st.sidebar.header("Select an Algorithm")
    
    model_options = [
        "XGBoost", "Random Forest", "LightGBM", "CatBoost", 
        "Logistic Regression", "K-Nearest Neighbors", "Decision Tree", 
        "Support Vector Machine", "Naive Bayes"
    ]
    selected_model_name = st.sidebar.selectbox(
        "Choose the classification model you want to train:",
        model_options,
        key="model_selector"
    )
    
    # stable separator for sidebar
    st.sidebar.markdown("---")
    
    # --- Dynamic Hyperparameter Sidebar ---
    st.sidebar.header(f"{selected_model_name} Hyperparameters")
    
    model_params = {}
    fit_params = {} # Parameters specific to the .fit() method

    # --- XGBoost Manual Parameters (Retained as per requirement) ---
    if selected_model_name == "XGBoost":
        st.sidebar.info("Adjust the parameters below. Un-set parameters will use XGBoost defaults.")
        with st.sidebar.expander("1. General Parameters", expanded=True):
            model_params['booster'] = st.selectbox("Booster", ['gbtree', 'gblinear', 'dart'], index=0)
            model_params['nthread'] = st.number_input("n_threads / n_jobs", min_value=1, value=os.cpu_count(), help="Number of parallel threads.")
            model_params['verbosity'] = st.select_slider("Verbosity", options=[0, 1, 2, 3], value=1, help="0 (silent), 1 (warning), 2 (info), 3 (debug).")
            model_params['seed'] = st.number_input("random_state (seed)", value=42)
        with st.sidebar.expander("2. Core Training Parameters", expanded=True):
            model_params['n_estimators'] = st.number_input("n_estimators", min_value=1, max_value=5000, value=100, step=10)
            model_params['learning_rate'] = st.slider("learning_rate (eta)", 0.0, 1.0, 0.3, 0.01, help="Step size shrinkage. Range: [0, 1]")
        with st.sidebar.expander("3. Booster Parameters", expanded=True):
            if model_params.get('booster') in ['gbtree', 'dart']:
                st.subheader("Tree Growth")
                model_params['max_depth'] = st.slider("max_depth", 3, 15, 6, 1)
                model_params['min_child_weight'] = st.number_input("min_child_weight", min_value=0.0, value=1.0, step=0.1)
                model_params['gamma'] = st.number_input("gamma (min_split_loss)", min_value=0.0, value=0.0, step=0.1)
                model_params['grow_policy'] = st.selectbox("grow_policy", ['depthwise', 'lossguide'], help="How new nodes are added.")
                model_params['max_delta_step'] = st.number_input("max_delta_step", min_value=0.0, value=0.0, step=0.1, help="Max delta step for tree weight estimation. Range: [0, ∞)")
                model_params['max_bin'] = st.number_input("max_bin", min_value=2, value=256, help="Max discrete bins for continuous features.")
                model_params['tree_method'] = st.selectbox("tree_method", ['auto', 'exact', 'approx', 'hist', 'gpu_hist'])
                updaters = st.multiselect("updater (tree plugins)",
                                        options=['grow_colmaker', 'grow_histmaker', 'grow_gpu_hist', 'refresh', 'prune'],
                                        help="List of tree updaters to run.")
                if updaters:
                    model_params['updater'] = ','.join(updaters)
                
                st.subheader("Sampling")
                model_params['subsample'] = st.slider("subsample", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of training instances. Range: (0, 1]")
                model_params['colsample_bytree'] = st.slider("colsample_bytree", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per tree. Range: (0, 1]")
                model_params['colsample_bylevel'] = st.slider("colsample_bylevel", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per level. Range: (0, 1]")
                model_params['colsample_bynode'] = st.slider("colsample_bynode", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per node. Range: (0, 1]")

                st.subheader("Regularization")
                model_params['lambda'] = st.number_input("lambda (L2 regularization)", min_value=0.0, value=1.0, step=0.1, help="Range: [0, ∞)")
                model_params['alpha'] = st.number_input("alpha (L1 regularization)", min_value=0.0, value=0.0, step=0.1, help="Range: [0, ∞)")
                if task_type == 'Classification':
                    y_train_series = pd.Series(st.session_state['y_train'])
                    if y_train_series.nunique() == 2:
                        scale_pos_weight_default = (y_train_series == 0).sum() / max(1, (y_train_series == 1).sum())
                        model_params['scale_pos_weight'] = st.number_input("scale_pos_weight", min_value=0.0, value=float(scale_pos_weight_default), help="Balance of positive/negative weights. Range: [0, ∞)")
                
                # --- DART Booster Specific Parameters ---
                if model_params.get('booster') == 'dart':
                    st.subheader("DART Booster Parameters")
                    model_params['sample_type'] = st.selectbox("sample_type", ['uniform', 'weighted'], help="Type of sampling algorithm.")
                    model_params['normalize_type'] = st.selectbox("normalize_type", ['tree', 'forest'], help="Type of normalization algorithm.")
                    model_params['rate_drop'] = st.slider("rate_drop", 0.0, 1.0, 0.5, 0.05, help="Dropout rate. Range: [0, 1]")
                    model_params['skip_drop'] = st.slider("skip_drop", 0.0, 1.0, 0.5, 0.05, help="Probability of skipping dropout. Range: [0, 1]")

            # --- Parameters for Linear Booster (gblinear) ---
            elif model_params.get('booster') == 'gblinear':
                st.subheader("Linear Booster Parameters")
                model_params['updater'] = st.selectbox("updater", ['coord_descent', 'shotgun', 'shotgun_col', 'distcol'], help="Updater for linear booster.")
                model_params['feature_selector'] = st.selectbox("feature_selector", ['cyclic', 'shuffle', 'greedy', 'thrifty'], help="Feature selection and ordering method.")
                st.subheader("Regularization")
                model_params['lambda'] = st.number_input("lambda (L2 regularization)", min_value=0.0, value=1.0, step=0.1)
                model_params['alpha'] = st.number_input("alpha (L1 regularization)", min_value=0.0, value=0.0, step=0.1)


        with st.sidebar.expander("4. Learning Task Parameters", expanded=True):
            st.write("`objective` is automatically set based on your task type selection.")
            if task_type == "Classification":
                num_class = pd.Series(st.session_state['y_train']).nunique()
                if num_class > 2:
                    model_params['objective'] = 'multi:softmax'
                    model_params['num_class'] = num_class
                    st.info(f"Objective: `multi:softmax` (for multi-class)")
                    model_params['eval_metric'] = st.selectbox("eval_metric", ['mlogloss', 'merror'])
                else:
                    model_params['objective'] = 'binary:logistic'
                    st.info(f"Objective: `binary:logistic` (for binary class)")
                    model_params['eval_metric'] = st.selectbox("eval_metric", ['logloss', 'error', 'auc'])
            else:  # Regression
                model_params['objective'] = 'reg:squarederror'
                st.info(f"Objective: `reg:squarederror` (for regression)")
                model_params['eval_metric'] = st.selectbox("eval_metric", ['rmse', 'mae', 'rmsle'])

            model_params['base_score'] = st.slider("base_score", 0.0, 1.0, 0.5, 0.01, help="Initial prediction score of all instances. Range: [0, 1]")
            fit_params['early_stopping_rounds'] = st.number_input("early_stopping_rounds", 0, 100, 10, 5, help="Set to 0 to disable.")

    # --- Parameters for Other Models ---
    if selected_model_name == "Random Forest":
        model_params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 1000, 100, 10)
        max_depth_val = st.sidebar.number_input("Max Depth (0 for none)", 0, 100, 10)
        model_params['max_depth'] = None if max_depth_val == 0 else int(max_depth_val)
        model_params['criterion'] = st.sidebar.selectbox("Criterion", ['gini', 'entropy', 'log_loss'])

    elif selected_model_name == "LightGBM":
        model_params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 2000, 100, 10)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, 0.005)
        model_params['num_leaves'] = st.sidebar.slider("Number of Leaves", 10, 200, 31)

    elif selected_model_name == "CatBoost":
        model_params['iterations'] = st.sidebar.slider("Iterations", 100, 5000, 1000, 100)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.001, 0.3, 0.05, 0.005)
        model_params['depth'] = st.sidebar.slider("Depth", 4, 12, 6)

    elif selected_model_name == "Logistic Regression":
        model_params['C'] = st.sidebar.number_input("C (Regularization)", 0.01, 100.0, 1.0, 0.01)
        model_params['max_iter'] = st.sidebar.number_input("Max Iterations", 100, 5000, 1000, 100)
        model_params['solver'] = st.sidebar.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])

    elif selected_model_name == "K-Nearest Neighbors":
        model_params['n_neighbors'] = st.sidebar.slider("Number of Neighbors (k)", 1, 50, 5, 1)
        model_params['weights'] = st.sidebar.selectbox("Weights", ['uniform', 'distance'])
        model_params['algorithm'] = st.sidebar.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])

    elif selected_model_name == "Decision Tree":
        model_params['criterion'] = st.sidebar.selectbox("Criterion", ['gini', 'entropy', 'log_loss'])
        model_params['splitter'] = st.sidebar.selectbox("Splitter", ['best', 'random'])
        max_depth_val = st.sidebar.number_input("Max Depth (0 for none)", 0, 100, 10)
        model_params['max_depth'] = None if max_depth_val == 0 else int(max_depth_val)

    elif selected_model_name == "Support Vector Machine":
        model_params['C'] = st.sidebar.number_input("C (Regularization)", 0.01, 100.0, 1.0, 0.01)
        model_params['kernel'] = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
        model_params['gamma'] = st.sidebar.selectbox("Gamma", ['scale', 'auto'])

    elif selected_model_name == "Naive Bayes":
        model_params['var_smoothing'] = st.sidebar.number_input("Variance Smoothing", 1e-10, 1e-5, 1e-9, 1e-10, format="%.e")

    # --- Train Model ---
    st.header(f"Train: {selected_model_name}")
    model = None
    if st.button(f"Start Training", type="primary"):
        with st.spinner(f"Training {selected_model_name}..."):
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Label encoding block
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            st.session_state['label_mapping'] = dict(zip(le.transform(le.classes_), le.classes_))

            # Instantiate model
            if selected_model_name == "XGBoost":
                model = xgb.XGBClassifier(**model_params, use_label_encoder=False)
            elif selected_model_name == "Random Forest":
                model = RandomForestClassifier(**model_params, random_state=42)
            elif selected_model_name == "LightGBM":
                model = lgb.LGBMClassifier(**model_params, random_state=42)
            elif selected_model_name == "CatBoost":
                model = cb.CatBoostClassifier(**model_params, random_state=42, verbose=0)
            elif selected_model_name == "Logistic Regression":
                model = LogisticRegression(**model_params, random_state=42)
            elif selected_model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(**model_params)
            elif selected_model_name == "Decision Tree":
                model = DecisionTreeClassifier(**model_params, random_state=42)
            elif selected_model_name == "Support Vector Machine":
                model = SVC(**model_params, probability=True, random_state=42)
            elif selected_model_name == "Naive Bayes":
                model = GaussianNB(**model_params)
            else:
                st.error(f"Unknown model selected: {selected_model_name}")
                model = None

    # Fit the model
    if model is not None:
        if selected_model_name == "XGBoost" and fit_params.get("early_stopping_rounds", 0) > 0:
            eval_set = [(X_test, y_test)]
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=fit_params['early_stopping_rounds'],
                    verbose=False
                )
            except TypeError:
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
        else:
            model.fit(X_train, y_train)

        st.session_state['trained_model_object'] = model
        st.session_state['trained_model_name'] = selected_model_name
        st.success(f"{selected_model_name} trained successfully!")

    # --- ROBUST LOGGING LOGIC ---
    # Calculate metrics and log them immediately after training
    if 'trained_model_object' in st.session_state and st.session_state['trained_model_object'] is not None:
        model = st.session_state['trained_model_object']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        if model is not None:
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

            new_record = metrics.copy()
            new_record['Model'] = selected_model_name
            if 'comparison_results' not in st.session_state:
                st.session_state['comparison_results'] = []
            st.session_state['comparison_results'].append(new_record)
            st.success(f"Results for **{selected_model_name}** logged. View them on the 'Model Comparison' page.")
            
    # --- END OF ROBUST LOGGING ---
    # A rerun will happen automatically, and the results will be displayed below
    # --- Display Results ---
    if ('trained_model_object' in st.session_state and st.session_state['trained_model_object'] is not None):
        st.divider()

        trained_model_name = st.session_state['trained_model_name']
        st.header(f"Results for: {trained_model_name}")

        model = st.session_state['trained_model_object']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]

        # --- Metrics ---
        metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        for i, (metric, value) in enumerate(metrics.items()):
            col = col1 if i % 2 == 0 else col2
            # show numeric metrics with 4 decimals
            try:
                col.metric(metric, f"{value:.4f}")
            except Exception:
                col.metric(metric, str(value))

        # --- Visualizations ---
        st.subheader("Evaluation Visualizations")
        y_unique = np.unique(y_test)

        # --- Binary Classification ---
        if y_prob is not None and len(y_unique) == 2:
            st.markdown("### Threshold Optimization (Binary Classification)")
            thresholds = np.arange(0.0, 1.01, 0.01)
            metrics_list = []
            for t in thresholds:
                y_pred_t = (y_prob >= t).astype(int)
                precision = precision_score(y_test, y_pred_t, zero_division=0)
                recall = recall_score(y_test, y_pred_t, zero_division=0)
                f1 = f1_score(y_test, y_pred_t, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_t)
                metrics_list.append([t, precision, recall, f1, accuracy])

            metrics_df = pd.DataFrame(metrics_list, columns=["Threshold", "Precision", "Recall", "F1-Score", "Accuracy"])
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in ["Precision", "Recall", "F1-Score", "Accuracy"]:
                ax.plot(metrics_df["Threshold"], metrics_df[col], label=col)
            ax.set_title("Threshold Optimization Curve", fontsize=16)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, linestyle="--")
            st.pyplot(fig)

            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            st.pyplot(fig)

        # --- Multi-class Classification ---
        elif y_prob is not None and len(y_unique) > 2:
            st.markdown("### Threshold Optimization (One-vs-Rest, Multi-class)")
            thresholds = np.arange(0.0, 1.01, 0.05)
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, cls in enumerate(y_unique):
                metrics_list = []
                for t in thresholds:
                    y_pred_t = (y_prob[:, i] >= t).astype(int)
                    precision = precision_score((y_test == cls).astype(int), y_pred_t, zero_division=0)
                    recall = recall_score((y_test == cls).astype(int), y_pred_t, zero_division=0)
                    f1 = f1_score((y_test == cls).astype(int), y_pred_t, zero_division=0)
                    metrics_list.append([t, f1])
                metrics_df = pd.DataFrame(metrics_list, columns=["Threshold", "F1-Score"])
                ax.plot(metrics_df["Threshold"], metrics_df["F1-Score"], label=f"Class {cls}")
            ax.set_title("F1-Score Curves (One-vs-Rest)", fontsize=16)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("F1-Score")
            ax.legend()
            ax.grid(True, linestyle="--")
            st.pyplot(fig)

            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
            st.pyplot(fig)

        # --- Regression ---
        else:
            st.markdown("### Regression Diagnostics")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("True vs Predicted")
            st.pyplot(fig)

            # Residual Plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(residuals, kde=True, bins=30, ax=ax, color="purple")
            ax.set_title("Residuals Distribution")
            ax.set_xlabel("Residual")
            st.pyplot(fig)

        # --- Save Model ---
        st.divider()
        st.header(f"💾 Save {trained_model_name} Model")
        model_save_name = st.text_input("Enter a file name for the saved model", value=f"{trained_model_name.replace(' ', '_')}_v1")
        if st.button(f"Save '{model_save_name}'"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{model_save_name}_{ts}"
            model_path = os.path.join(MODELS_DIR, f"{base_filename}.joblib")
            joblib.dump(model, model_path)

            # Save metadata
            metadata = {
                "model_name": trained_model_name,
                "saved_name": model_save_name,
                "features": st.session_state.get('features'),
                "original_features": st.session_state.get('original_features'),
                "original_dtypes": st.session_state.get('original_dtypes'),
                "target": st.session_state.get('target'),
                "task_type": "Classification",
                "label_mapping": st.session_state.get('label_mapping')
            }
            with open(os.path.join(MODELS_DIR, f"{base_filename}_meta.json"), 'w') as f:
                json.dump(metadata, f, indent=4)

            st.success(f"Model `{trained_model_name}` saved successfully as `{base_filename}.joblib`!")
    else:
        st.info("⚠️ No trained model found yet. Please train a model first.")
