import streamlit as st

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col1:
    if st.button("타이틀 페이지로 가기"):
        st.switch_page("타이틀.py")
with col3:
    if st.button("캐릭터 관리 페이지로 가기"):
        st.switch_page("캐릭터_관리.py")

st.title("채팅 관리")

@st.dialog("새 채팅 시작")
def start_new_chat():
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col1:
        st.text("검색")
    with col2:
        q = st.text_input("", label_visibility="collapsed")
    with col3:
        st.button("", icon=":material/search:")

    st.subheader("캐릭터 선택")

    with st.container(border=True):
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.checkbox("")
        with col2:
            with st.expander("캐릭터 이름"):
                st.write("캐릭터 특성")

    st.button("채팅 시작", use_container_width=True)

st.button("새 채팅 시작", on_click=start_new_chat, use_container_width=True)

st.subheader("진행 중인 채팅")
with st.container(border=True):
    col1, col2, col3 = st.columns([0.65, 0.2, 0.15])
    with col1:
        st.write("캐릭터0, 캐릭터1, 캐릭터2, ... + n")
    with col2:
        st.button("채팅 시작", use_container_width=True)
    with col3:
        st.button("삭제", use_container_width=True, type="primary")