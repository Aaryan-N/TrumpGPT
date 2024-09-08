import streamlit as st

# URL of the Flask app
flask_app_url = "http://localhost:8080"

st.set_page_config(layout="wide")
st.title("TrumpGPT")
st.markdown("""
    <style>
        .css-18e3th9 {padding-top: 0rem; padding-bottom: 0rem;}
        .css-1d391kg {padding: 0rem;}
    </style>
    """, unsafe_allow_html=True)

# Embed the Flask app in an iframe
st.components.v1.html(f"""
    <iframe src="{flask_app_url}" width="100%" height="100%" style="position: fixed; top: 0; left: 0; bottom: 0; right: 0; border: none;"></iframe>
""", height=0)