import streamlit as st
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils import load_data, get_next_version
import auth

# --- Page Setup and Authentication ---
auth.login()
st.set_page_config(page_title="Data Upload & Preparation", layout="wide")


if st.session_state.get('authenticated'):
    if st.session_state.get('role') != 'admin':
        st.error("You do not have permission to view this page.")
        st.stop()

    st.title("Step 1: Data Upload & Preparation (Admin)")

    # Define paths
    UPLOADS_DIR = "./project_data/uploads/"
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # --- 1. Dataset Import Module ---
    st.header("1. Import Your Dataset")

    data_source = st.radio(
        "Choose data source:",
        ("Upload a new file", "Use a previously uploaded file"),
        horizontal=True,
        label_visibility="collapsed"
    )

    df = None

    if data_source == "Upload a new file":
        uploaded_file = st.file_uploader(
            "Upload your data (CSV, Excel, JSON, Parquet)",
            type=["csv", "xlsx", "xls", "json", "parquet"]
        )
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                version = get_next_version(UPLOADS_DIR, "dataset_v")
                save_path = os.path.join(UPLOADS_DIR, f"dataset_v{version}.csv")
                df.to_csv(save_path, index=False)
                st.success(f"File uploaded and saved as `dataset_v{version}.csv`")
                st.session_state['df'] = df

    elif data_source == "Use a previously uploaded file":
        try:
            existing_files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith('.csv')]
            if not existing_files:
                st.warning("No previously uploaded datasets found.")
            else:
                selected_file = st.selectbox("Select a dataset", existing_files)
                if selected_file:
                    df = pd.read_csv(os.path.join(UPLOADS_DIR, selected_file))
                    st.session_state['df'] = df
        except FileNotFoundError:
            st.error(f"The directory {UPLOADS_DIR} was not found. Please upload a file first.")

    if 'df' in st.session_state and st.session_state['df'] is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state['df'].head())
        st.divider()

        # --- 2. Data Preparation and DMatrix Conversion ---
        st.header("2. Prepare Data for XGBoost")

        target_column = st.selectbox(
            "Select the Target Column (the variable you want to predict)",
            options=st.session_state['df'].columns
        )

        test_size = st.slider("Select Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("Prepare & Convert Data", type="primary"):
            with st.spinner("Processing data..."):
                df_processed = st.session_state['df'].copy()

                label_mapping = None
                target_series = df_processed[target_column]
                if pd.api.types.is_numeric_dtype(target_series):
                    task_type = "Regression" if target_series.nunique() > 20 else "Classification"
                else:
                    task_type = "Classification"
                    df_processed[target_column] = df_processed[target_column].astype('category')
                    y_encoded = df_processed[target_column].cat.codes
                    label_mapping = dict(enumerate(df_processed[target_column].cat.categories))
                    df_processed[target_column] = y_encoded
                    st.success("Detected and encoded categorical target variable.")
                    st.write("Label Mapping created:", label_mapping)

                st.session_state['task_type'] = task_type
                st.session_state['label_mapping'] = label_mapping
                st.success(f"Automatically detected task type: **{task_type}**")

                y = df_processed[target_column]
                X = df_processed.drop(columns=[target_column])

                # --- NEW: Save original features and dtypes before encoding ---
                st.session_state['original_features'] = X.columns.tolist()
                st.session_state['original_dtypes'] = X.dtypes.apply(lambda x: x.name).to_dict()
                # --- END NEW ---

                X_encoded = pd.get_dummies(X, drop_first=True)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=test_size, random_state=42, stratify=(y if y.nunique() < 20 else None)
                )

                st.session_state.update({
                    'X_train': X_train, 'y_train': y_train,
                    'X_test': X_test, 'y_test': y_test,
                    'features': X_encoded.columns.tolist(), # Encoded features
                    'target': target_column
                })

                st.success("Data successfully split and prepared!")
                st.info(f"Training set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")
                st.info("You can now proceed to the 'Model Training' page.")
