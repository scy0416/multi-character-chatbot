import uuid

import streamlit as st
from app import get_firestore

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col1:
    if st.button("타이틀 페이지로 가기"):
        st.switch_page("타이틀.py")
with col3:
    if st.button("캐릭터 관리 페이지로 가기"):
        st.switch_page("캐릭터_관리.py")

db = get_firestore()

selected = []

st.session_state.chat_id = None

st.title("채팅 관리")

@st.dialog("새 채팅 시작")
def new_chat_character_select():
    # 검색은 일단 보류
    # col1, col2 = st.columns([0.8, 0.2])
    # with col1:
    #     q = st.text_input("검색", label_visibility="collapsed", icon=":material/search:")
    # with col2:
    #     st.button("검색", use_container_width=True)

    st.subheader("캐릭터 선택")

    with st.spinner("불러오는 중..."):
        snaps = db.collection("characters").get()

    with st.container(border=True):
        for snap in snaps:
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.checkbox("선택", key=f"newchat:{snap.id}", label_visibility="collapsed")
            with col2:
                with st.expander(snap.id):
                    st.write(f"성별: {snap.get('gender')}")
                    st.write(f"MBTI: {snap.get('mbti')}")
                    st.write(f"대화스타일: {snap.get('conversation_style')}")
                    st.write(f"기타: {snap.get('etc')}")

    global selected
    selected = [snap.id for snap in snaps if st.session_state.get(f"newchat:{snap.id}")]
    st.button("채팅 시작", use_container_width=True, on_click=new_chat_start, disabled=len(selected)<=0)
    if len(selected) <= 0:
        st.error("캐릭터를 선택하세요!")

def new_chat_start():
    db.collection("chats").document(st.user.sub).collection(str(uuid.uuid4())).document("info").set({"characters": selected})

st.button("새 채팅 시작", on_click=new_chat_character_select, use_container_width=True)

with st.spinner("불러오는 중..."):
    cols = db.collection("chats").document(st.user.sub).collections()

def delete_collection_recursive(coll_ref, batch_size: int = 500):
    while True:
        docs = list(coll_ref.limit(batch_size).stream())
        if not docs:
            break

        for snap in docs:
            for sub in snap.reference.collections():
                delete_collection_recursive(sub, batch_size)

        batch = db.batch()
        for snap in docs:
            batch.delete(snap.reference)
        batch.commit()

def del_chat(chat_id):
    col_ref = db.collection("chats").document(st.user.sub).collection(chat_id)
    with st.spinner(f"채팅({chat_id}) 삭제 중..."):
        delete_collection_recursive(col_ref, batch_size=300)
    st.toast("삭제 완료 ✅")

with st.container(border=True):
    for col in cols:
        characters = col.document("info").get().get("characters")
        col1, col2, col3 = st.columns([0.65, 0.2, 0.15])
        with col1:
            if len(characters) > 3:
                st.write(f"{characters[0]}, {characters[1]}, {characters[2]}, ... +{len(characters)-3}")
            else:
                st.write(", ".join(characters))
        with col2:
            if st.button("채팅 시작", use_container_width=True, key=f"{col.id}1"):
                st.session_state.chat_id = col.id
                st.switch_page("챗봇.py")
        with col3:
            st.button("삭제", use_container_width=True, type="primary", key=f"{col.id}2", on_click=del_chat, kwargs={"chat_id": col.id})

        st.divider()