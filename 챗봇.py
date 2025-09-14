import streamlit as st

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

if not "chat_id" in st.session_state or st.session_state.chat_id is None:
    st.switch_page("타이틀.py")

st.write(st.session_state.chat_id)