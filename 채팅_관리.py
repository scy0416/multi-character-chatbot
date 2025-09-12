import streamlit as st

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")