import streamlit as st

def login():
    """Displays a login form and handles authentication."""
    
    # Define hardcoded credentials
    ADMIN_USERNAME = "a"
    ADMIN_PASSWORD = "p"
    CLIENT_USERNAME = "c"
    CLIENT_PASSWORD = "p"

    # If already authenticated, do nothing
    if st.session_state.get('authenticated'):
        return

    # Create the login form in the sidebar
    with st.sidebar:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Log In")

    if login_button:
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state['authenticated'] = True
            st.session_state['role'] = 'admin'
            st.rerun()
        elif username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
            st.session_state['authenticated'] = True
            st.session_state['role'] = 'client'
            st.rerun()
        else:
            st.sidebar.error("Incorrect username or password")

    # If not authenticated, stop the rest of the app from running
    if not st.session_state.get('authenticated'):
        st.info("Please log in using the sidebar to continue.")
        st.stop()