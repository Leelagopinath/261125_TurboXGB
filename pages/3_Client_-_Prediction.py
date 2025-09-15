import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import json
import auth
from utils import load_data

# --- Page Setup and Authentication ---
auth.login()
st.set_page_config(page_title="Predict with Model", layout="wide")

if st.session_state.get('authenticated'):
    st.title("Step 3: Get Predictions (Client)")

    MODELS_DIR = "./project_data/models/"

    # --- Load Trained Model ---
    st.header("1. Load a Trained XGBoost Model")

    try:
        # Models are now saved in .json format from the manual training page
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.json') and '_meta.json' not in f]
        if not model_files:
            st.warning("No trained models found. Please train a model first on the 'Model Training' page.")
            st.stop()
        
        selected_model_file = st.selectbox("Select a trained model", model_files)

        if selected_model_file:
            model_path = os.path.join(MODELS_DIR, selected_model_file)
            meta_path = os.path.join(MODELS_DIR, selected_model_file.replace('.json', '_meta.json'))
            
            # Load the native XGBoost model
            model = xgb.Booster()
            model.load_model(model_path)
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            st.session_state['loaded_model'] = model
            st.session_state['model_metadata'] = metadata
            st.success(f"Model `{selected_model_file}` loaded successfully.")

    except FileNotFoundError:
        st.error(f"The model directory was not found. Please ensure it exists.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

    # --- Prediction Section ---
    if 'loaded_model' in st.session_state:
        st.divider()
        st.header("2. Choose Prediction Method")
        
        input_method = st.radio(
            "Select how you want to provide data:",
            ("Upload a File (Bulk Prediction)", "Enter Features Manually (Single Prediction)"),
            horizontal=True
        )
        
        model = st.session_state['loaded_model']
        metadata = st.session_state['model_metadata']
        
        # --- Preprocessing Function ---
        def preprocess_for_prediction(df, original_features, original_dtypes, final_feature_list):
            """Applies the same preprocessing as the training step."""
            # Ensure correct data types from original dataset
            for col, dtype in original_dtypes.items():
                if col in df.columns:
                    if 'cat' in dtype or 'obj' in dtype:
                        df[col] = df[col].astype('category')
                    elif 'int' in dtype:
                         df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif 'float' in dtype:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

            # Select only the original features the model was trained on
            df_original_feats = df[original_features]
            
            # One-hot encode
            df_processed = pd.get_dummies(df_original_feats, drop_first=True)
            
            # Align columns with the final feature list from training
            current_cols = df_processed.columns
            for col in final_feature_list:
                if col not in current_cols:
                    df_processed[col] = 0 # Add missing columns
            
            return df_processed[final_feature_list] # Return in correct order

        # --- METHOD 1: BULK UPLOAD ---
        if input_method == "Upload a File (Bulk Prediction)":
            new_data_file = st.file_uploader("Upload data for bulk prediction", type=["csv", "xlsx"])
            if new_data_file:
                new_df = load_data(new_data_file)
                if new_df is not None:
                    st.subheader("Preview of Uploaded Data")
                    st.dataframe(new_df.head())
                    
                    if st.button("Get Bulk Predictions", type="primary"):
                        with st.spinner("Processing and predicting..."):
                            df_aligned = preprocess_for_prediction(new_df.copy(), metadata['original_features'], metadata['original_dtypes'], metadata['features'])
                            dnew = xgb.DMatrix(df_aligned)
                            
                            predictions_raw = model.predict(dnew)
                            
                            new_df['prediction_numeric'] = predictions_raw.astype(int)
                            
                            if metadata.get('label_mapping'):
                                mapping_dict = {int(k): v for k, v in metadata['label_mapping'].items()}
                                new_df['prediction_label'] = new_df['prediction_numeric'].map(mapping_dict)

                            st.subheader("Predictions")
                            st.dataframe(new_df)
                            
                            csv = new_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Predictions as CSV", csv, "bulk_predictions.csv", "text/csv")

        # --- METHOD 2: MANUAL ENTRY ---
        elif input_method == "Enter Features Manually (Single Prediction)":
            st.subheader("Enter Feature Values")
            
            input_data = {}
            original_features = metadata['original_features']
            original_dtypes = metadata['original_dtypes']
            
            with st.form(key='manual_input_form'):
                for feature in original_features:
                    dtype = original_dtypes.get(feature, 'object')
                    if 'cat' in dtype or 'obj' in dtype:
                        # Find unique categories from training data to create a dropdown
                        unique_vals = st.session_state.get('X_train_original', pd.DataFrame(columns=original_features))[feature].unique().tolist()
                        input_data[feature] = st.selectbox(f"Select value for '{feature}'", options=unique_vals)
                    else: # Numeric
                        input_data[feature] = st.number_input(f"Enter value for '{feature}'", value=0.0)
                
                submit_button = st.form_submit_button(label='Get Single Prediction')

            if submit_button:
                input_df = pd.DataFrame([input_data])
                
                df_aligned = preprocess_for_prediction(input_df.copy(), original_features, original_dtypes, metadata['features'])
                dnew = xgb.DMatrix(df_aligned)
                
                prediction_numeric = model.predict(dnew)
                
                st.subheader("Prediction Result")
                if metadata.get('label_mapping'):
                    mapping_dict = {int(k): v for k, v in metadata['label_mapping'].items()}
                    predicted_label = mapping_dict.get(int(prediction_numeric[0]), "Unknown")
                    st.metric(label="Predicted Label", value=predicted_label)
                else:
                    st.metric(label="Predicted Value", value=f"{prediction_numeric[0]:.4f}")

