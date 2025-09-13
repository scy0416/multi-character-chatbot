import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

@st.cache_resource
def get_firestore():
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

    return firestore.client()

pg = st.navigation(["타이틀.py", "캐릭터_관리.py", "채팅_관리.py", "챗봇.py"], position="top")
pg.run()