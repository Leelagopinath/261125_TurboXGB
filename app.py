#To Freeze the complete libraries

#      "" pip freeze > requirements.txt ""

#To recreate the same environment elsewhere:

#      "" pip install -r requirements.txt ""


import streamlit as st
import auth

# Set the page configuration
st.set_page_config(
    page_title="Universal XGBoost Modeler",
    page_icon="🚀",
    layout="wide"
)

# Run the login check
auth.login()

# Main title that will appear on all pages
st.title("🚀 Universal XGBoost Modeler")

st.sidebar.success(f"Logged in as: {st.session_state.get('role')}")