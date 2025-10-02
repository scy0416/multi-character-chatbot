import uuid

import streamlit as st
from app import get_firestore
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

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

client = Redis(
    host=st.secrets["redis"]["host"],
    port=st.secrets["redis"]["port"],
    password=st.secrets["redis"]["password"],
    ssl=False,
    ssl_cert_reqs="required",
    decode_responses=False
)
saver = RedisSaver(redis_client=client)

st.session_state.chat_id = None
st.session_state.chat_start = False

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

    selected = [snap.id for snap in snaps if st.session_state.get(f"newchat:{snap.id}")]
    if st.button("채팅 시작", use_container_width=True, disabled=len(selected)<=0):
        new_chat_id = str(uuid.uuid4())

        characters_info = {}

        for character in selected:
            characters_info[character] = db.collection("characters").document(character).get().to_dict()
            characters_info[character]["lore"] = ""

        db.collection("chats").document(st.user.sub).collection(new_chat_id).document("info").set({"characters": selected})
        db.collection("chats").document(st.user.sub).collection(new_chat_id).document("participants").set(characters_info)

        st.session_state.chat_id = new_chat_id
        st.switch_page("챗봇.py")
    if len(selected) <= 0:
        st.error("캐릭터를 선택하세요!")

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
    saver.delete_thread(chat_id)
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