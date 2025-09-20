import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import auth

# --- Page Setup and Authentication ---
auth.login()
st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")

if st.session_state.get('authenticated'):
    if st.session_state.get('role') != 'admin':
        st.error("You do not have permission to view this page.")
        st.stop()

    # Initialize comparison_results if not present
    if 'comparison_results' not in st.session_state:
        st.session_state['comparison_results'] = []

    st.title("Step 2: Universal Model Trainer (Admin)")

    # Convert the list of dictionaries from session state into a DataFrame
    results_df = pd.DataFrame(st.session_state['comparison_results'])
    # Ensure the report only shows the latest result for any given model name
    results_df = results_df.drop_duplicates(subset=['Model'], keep='last').reset_index(drop=True)

    st.header("Comparative Performance Report")

    # --- Best Model Recommendation ---
    st.subheader("🚀 Best Model Recommendation")
    
    # Define metric columns for selection
    metric_cols = [col for col in results_df.columns if col not in ['Model', 'Timestamp']]
    # Default to F1-Score if available, otherwise the first metric
    default_metric_index = metric_cols.index('F1-Score') if 'F1-Score' in metric_cols else 0
    
    primary_metric = st.selectbox(
        "Select primary metric to determine the best model:",
        options=metric_cols,
        index=default_metric_index
    )

    if not results_df.empty:
        best_model_row = results_df.loc[results_df[primary_metric].idxmax()]
        st.success(f"**Best Model:** `{best_model_row['Model']}` (with a {primary_metric} of **{best_model_row[primary_metric]:.4f}**)")
    
    # --- Performance Metrics Table ---
    st.subheader("📊 Performance Metrics Table")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#90EE90', subset=metric_cols))

    # --- Comparative Plot ---
    st.subheader("📈 Metrics Comparison Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort for better visualization
    plot_data = results_df.set_index('Model')[primary_metric].sort_values(ascending=True)
    
    bars = ax.barh(plot_data.index, plot_data.values, color='skyblue')
    ax.set_title(f'Comparison of Models by {primary_metric}', fontsize=16)
    ax.set_xlabel(primary_metric, fontsize=12)
    ax.bar_label(bars, fmt='%.4f') # Add data labels to bars
    plt.tight_layout()
    st.pyplot(fig)

    # --- Add a button to clear the comparison log ---
    st.sidebar.divider()
    if st.sidebar.button("Clear Comparison Log"):
        st.session_state['comparison_results'] = []
        st.rerun()
