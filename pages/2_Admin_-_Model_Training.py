import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import calculate_classification_metrics, calculate_regression_metrics
import auth

# --- Page Setup and Authentication ---
auth.login()
st.set_page_config(page_title="XGBoost Model Training", layout="wide")

if st.session_state.get('authenticated'):
    if st.session_state.get('role') != 'admin':
        st.error("You do not have permission to view this page.")
        st.stop()

    st.title("Step 2: Train XGBoost Model (Admin)")

    MODELS_DIR = "./project_data/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    if 'X_train' not in st.session_state:
        st.error("Data not prepared. Please go to the 'Data Upload' page first.")
        st.stop()

    # --- Manual Task Type Selection ---
    st.header("1. Select Task Type")
    detected_task = st.session_state.get('task_type', 'Classification')
    task_options = ['Classification', 'Regression']
    try:
        default_index = task_options.index(detected_task)
    except ValueError:
        default_index = 0
    
    task_type = st.radio(
        "Select the modeling task (auto-detected shown as default)",
        options=task_options,
        index=default_index,
        horizontal=True,
        key="task_type_selector"
    )

    # --- Manual Hyperparameter Tuning ---
    st.sidebar.header("XGBoost Hyperparameters")
    st.sidebar.info("Adjust the parameters below. Un-set parameters will use XGBoost defaults.")

    params = {}

    with st.sidebar.expander("1. General Parameters", expanded=True):
        params['booster'] = st.selectbox("Booster", ['gbtree', 'gblinear', 'dart'], index=0)
        params['nthread'] = st.number_input("n_threads / n_jobs", min_value=1, value=os.cpu_count(), help="Number of parallel threads.")
        params['verbosity'] = st.select_slider("Verbosity", options=[0, 1, 2, 3], value=1, help="0 (silent), 1 (warning), 2 (info), 3 (debug).")
        params['seed'] = st.number_input("random_state (seed)", value=42)
        params['disable_default_eval_metric'] = st.checkbox("disable_default_eval_metric", value=False)
        params['process_type'] = st.selectbox("process_type", ['default', 'update'], help="Boosting process to run.")

    with st.sidebar.expander("2. Core Training Parameters", expanded=True):
        params['n_estimators'] = st.number_input("n_estimators", min_value=1, max_value=5000, value=100, step=10)
        params['learning_rate'] = st.slider("learning_rate (eta)", 0.0, 1.0, 0.3, 0.01, help="Step size shrinkage. Range: [0, 1]")

    with st.sidebar.expander("3. Booster Parameters", expanded=True):
        # --- Parameters for Tree Boosters (gbtree, dart) ---
        if params['booster'] in ['gbtree', 'dart']:
            st.subheader("Tree Growth")
            params['max_depth'] = st.slider("max_depth", 3, 15, 6, 1, help="Max tree depth. Range: [3, 10] recommended.")
            params['max_leaves'] = st.number_input("max_leaves", min_value=0, value=0, help="Max leaves. 0 means no limit. Range: [0, ∞)")
            params['min_child_weight'] = st.number_input("min_child_weight", min_value=0.0, value=1.0, step=0.1, help="Min sum of instance weight in a child. Range: [0, ∞)")
            params['gamma'] = st.number_input("gamma (min_split_loss)", min_value=0.0, value=0.0, step=0.1, help="Min loss reduction to partition. Range: [0, ∞)")
            params['grow_policy'] = st.selectbox("grow_policy", ['depthwise', 'lossguide'], help="How new nodes are added.")
            params['max_delta_step'] = st.number_input("max_delta_step", min_value=0.0, value=0.0, step=0.1, help="Max delta step for tree weight estimation. Range: [0, ∞)")
            params['max_bin'] = st.number_input("max_bin", min_value=2, value=256, help="Max discrete bins for continuous features.")
            params['tree_method'] = st.selectbox("tree_method", ['auto', 'exact', 'approx', 'hist', 'gpu_hist'])
            updaters = st.multiselect("updater (tree plugins)",
                                      options=['grow_colmaker', 'grow_histmaker', 'grow_gpu_hist', 'refresh', 'prune'],
                                      help="List of tree updaters to run.")
            if updaters:
                params['updater'] = ','.join(updaters)
            
            st.subheader("Sampling")
            params['subsample'] = st.slider("subsample", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of training instances. Range: (0, 1]")
            params['colsample_bytree'] = st.slider("colsample_bytree", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per tree. Range: (0, 1]")
            params['colsample_bylevel'] = st.slider("colsample_bylevel", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per level. Range: (0, 1]")
            params['colsample_bynode'] = st.slider("colsample_bynode", 0.01, 1.0, 1.0, 0.05, help="Subsample ratio of columns per node. Range: (0, 1]")

            st.subheader("Regularization")
            params['lambda'] = st.number_input("lambda (L2 regularization)", min_value=0.0, value=1.0, step=0.1, help="Range: [0, ∞)")
            params['alpha'] = st.number_input("alpha (L1 regularization)", min_value=0.0, value=0.0, step=0.1, help="Range: [0, ∞)")
            if task_type == 'Classification':
                 y_train_series = pd.Series(st.session_state['y_train'])
                 if y_train_series.nunique() == 2:
                    scale_pos_weight_default = (y_train_series == 0).sum() / max(1, (y_train_series == 1).sum())
                    params['scale_pos_weight'] = st.number_input("scale_pos_weight", min_value=0.0, value=float(scale_pos_weight_default), help="Balance of positive/negative weights. Range: [0, ∞)")
            
            # --- DART Booster Specific Parameters ---
            if params['booster'] == 'dart':
                st.subheader("DART Booster Parameters")
                params['sample_type'] = st.selectbox("sample_type", ['uniform', 'weighted'], help="Type of sampling algorithm.")
                params['normalize_type'] = st.selectbox("normalize_type", ['tree', 'forest'], help="Type of normalization algorithm.")
                params['rate_drop'] = st.slider("rate_drop", 0.0, 1.0, 0.5, 0.05, help="Dropout rate. Range: [0, 1]")
                params['skip_drop'] = st.slider("skip_drop", 0.0, 1.0, 0.5, 0.05, help="Probability of skipping dropout. Range: [0, 1]")

        # --- Parameters for Linear Booster (gblinear) ---
        elif params['booster'] == 'gblinear':
            st.subheader("Linear Booster Parameters")
            params['updater'] = st.selectbox("updater", ['coord_descent', 'shotgun', 'shotgun_col', 'distcol'], help="Updater for linear booster.")
            params['feature_selector'] = st.selectbox("feature_selector", ['cyclic', 'shuffle', 'greedy', 'thrifty'], help="Feature selection and ordering method.")
            st.subheader("Regularization")
            params['lambda'] = st.number_input("lambda (L2 regularization)", min_value=0.0, value=1.0, step=0.1)
            params['alpha'] = st.number_input("alpha (L1 regularization)", min_value=0.0, value=0.0, step=0.1)


    with st.sidebar.expander("4. Learning Task Parameters", expanded=True):
        st.write("`objective` is automatically set based on your task type selection.")
        if task_type == "Classification":
            num_class = pd.Series(st.session_state['y_train']).nunique()
            if num_class > 2:
                params['objective'] = 'multi:softmax'
                params['num_class'] = num_class
                st.info(f"Objective: `multi:softmax` (for multi-class)")
                params['eval_metric'] = st.selectbox("eval_metric", ['mlogloss', 'merror'])
            else:
                params['objective'] = 'binary:logistic'
                st.info(f"Objective: `binary:logistic` (for binary class)")
                params['eval_metric'] = st.selectbox("eval_metric", ['logloss', 'error', 'auc'])
        else: # Regression
            params['objective'] = 'reg:squarederror'
            st.info(f"Objective: `reg:squarederror` (for regression)")
            params['eval_metric'] = st.selectbox("eval_metric", ['rmse', 'mae', 'rmsle'])
        
        params['base_score'] = st.slider("base_score", 0.0, 1.0, 0.5, 0.01, help="Initial prediction score of all instances. Range: [0, 1]")
        params['early_stopping_rounds'] = st.number_input("early_stopping_rounds", 0, 100, 10, 5, help="Set to 0 to disable.")
        st.warning("`sample_weight`: This is a per-instance weight. To use it, please provide a column with weights in your dataset and load it as the 'weight' parameter in a DMatrix. It cannot be set as a single global value here.")

    # --- NEW: Display Selected Parameters ---
    st.subheader("Final Parameters for Training")
    with st.expander("Click to view the complete parameter dictionary"):
        st.json(params)
    # --- END OF NEW SECTION ---

    # --- Train Model ---
    st.header("2. Train the Model")
    if st.button("Start Training with Manual Parameters", type="primary"):
        with st.spinner("Training in progress..."):
            dtrain = xgb.DMatrix(st.session_state['X_train'], label=st.session_state['y_train'])
            dtest = xgb.DMatrix(st.session_state['X_test'], label=st.session_state['y_test'])
            
            evals = [(dtrain, 'train'), (dtest, 'eval')]
            num_boost_round = params.pop('n_estimators', 100)
            early_stopping_rounds = params.pop('early_stopping_rounds', 10)
            if early_stopping_rounds == 0:
                early_stopping_rounds = None

            # Create a copy of params for training, as xgb.train modifies the dict
            train_params = params.copy()

            model = xgb.train(
                train_params, dtrain, num_boost_round=num_boost_round,
                evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False
            )
            st.session_state['booster_model'] = model
            st.session_state['training_params'] = params # Store the original, unmodified params
            st.success("Model training completed!")

    if 'booster_model' in st.session_state:
        st.header("3. Model Evaluation")
        model = st.session_state['booster_model']
        training_params = st.session_state['training_params'] # Use stored params for evaluation logic
        dtest = xgb.DMatrix(st.session_state['X_test'], label=st.session_state['y_test'])
        y_test = st.session_state['y_test']
        y_pred_raw = model.predict(dtest)

        metrics = {}
        if task_type == "Classification":
            if training_params.get('objective') == 'binary:logistic':
                y_pred = (y_pred_raw > 0.5).astype(int)
                y_prob = y_pred_raw
            else:
                y_pred = y_pred_raw.astype(int)
                y_prob = None
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            y_pred = y_pred_raw
            metrics = calculate_regression_metrics(y_test, y_pred)

        st.subheader("Performance Metrics")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            metrics_df = pd.Series(metrics, name="Score").reset_index().rename(columns={'index': 'Metric'})
            st.dataframe(metrics_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=plt.cm.viridis(np.linspace(0.4, 0.9, len(metrics))))
            ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold')
            ax.set_xlabel('Score', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.grid(True, color='#eeeeee', linestyle='--')
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center')
            ax.set_xlim(0, max(1, max(metrics.values(), default=1)) * 1.15)
            st.pyplot(fig)
        
        if task_type == "Classification":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
            st.pyplot(fig)

        st.header("4. Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        if training_params.get('booster') != 'gblinear':
            xgb.plot_importance(model, ax=ax, max_num_features=20)
            st.pyplot(fig)
        else:
            st.info("Feature importance plot is not available for the linear booster.")

        st.header("5. Save the Trained Model")
        model_name = st.text_input("Enter a name for your model (e.g., 'manual_churn_v1')")
        if st.button("Save Model"):
            if model_name:
                ts = datetime.now().strftime("%Y%d%m_%H%M%S")
                base_filename = f"{model_name}_{ts}"
                model_path = os.path.join(MODELS_DIR, f"{base_filename}.json")
                model.save_model(model_path)

                metadata = {
                    "features": st.session_state['features'],
                    "original_features": st.session_state['original_features'],
                    "original_dtypes": st.session_state['original_dtypes'],
                    "target": st.session_state['target'],
                    "task_type": task_type,
                    "model_params": training_params,
                    "label_mapping": st.session_state.get('label_mapping')
                }
                with open(os.path.join(MODELS_DIR, f"{base_filename}_meta.json"), 'w') as f:
                    json.dump(metadata, f, indent=4)
                st.success(f"Model and metadata saved successfully as `{base_filename}.json`!")
            else:
                st.warning("Please enter a name for the model.")

