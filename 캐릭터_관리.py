import streamlit as st
from app import get_firestore

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col1:
    if st.button("타이틀 페이지로 가기"):
        st.switch_page("타이틀.py")
with col3:
    if st.button("채팅 관리 페이지로 가기"):
        st.switch_page("채팅_관리.py")

db = get_firestore()

st.title("캐릭터 관리")

def name_unique_check(name):
    names = [ref.id for ref in db.collection("characters").list_documents()]
    return not name in names

@st.dialog("캐릭터 추가")
def add_character():
    def reset_input():
        st.session_state["name"] = ""
        st.session_state["gender"] = "남성"
        st.session_state["주의초점"] = "E"
        st.session_state["인식기능"] = "S"
        st.session_state["판단기능"] = "T"
        st.session_state["생활양식"] = "J"
        st.session_state["conversation_style"] = "적음"
        st.session_state["etc"] = ""

    def save_character():
#         print(f"""이름: {st.session_state.get("name")}
# 성별: {st.session_state.get("gender")}
# MBTI: {st.session_state.get("주의초점")}{st.session_state.get("인식기능")}{st.session_state.get("판단기능")}{st.session_state.get("생활양식")}
# 대화스타일: {st.session_state.get("conversation_style")}
# 기타 정보: {st.session_state.get("etc")}""")

        db.collection("characters").document(st.session_state.get("name")).set({
            "gender": st.session_state.get("gender"),
            "mbti": f'{st.session_state.get("주의초점")}{st.session_state.get("인식기능")}{st.session_state.get("판단기능")}{st.session_state.get("생활양식")}',
            "conversation_style": st.session_state.get("conversation_style"),
            "etc": st.session_state.get("etc")
        })

    st.subheader("캐릭터의 기본 정보")
    name = st.text_input("이름:red[*]: ", key="name")
    gender = st.selectbox("성별을 고르세요", ("남성", "여성"), key="gender")

    st.subheader("MBTI")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        주의초점 = st.radio("주의초점", ["E", "I"], captions=["외향", "내향"], key="주의초점")
    with col2:
        인식기능 = st.radio("인식기능", ["S", "N"], captions=["감각", "직관"], key="인식기능")
    with col3:
        판단기능 = st.radio("판단기능", ["T", "F"], captions=["사고", "감정"], key="판단기능")
    with col4:
        생활양식 = st.radio("생활양식", ["J", "P"], captions=["판단", "인식"], key="생활양식")

    st.info(f"{주의초점}{인식기능}{판단기능}{생활양식}")

    st.subheader("대화 스타일")
    conversation_style = st.radio(
        "캐릭터의 말투를 고르세요.",
        ["적음", "보통", "많음"],
        captions=[
            "캐릭터의 말수가 적어집니다.",
            "캐릭터의 말수가 보통이됩니다.",
            "캐릭터의 말수가 많아집니다."
        ],
        key="conversation_style"
    )

    st.subheader("기타")
    st.text_area("기타 성격", placeholder="기타 정보를 입력하세요.", key="etc")

    col1, col2 = st.columns(2)
    with col1:
        st.button("리셋", use_container_width=True, on_click=reset_input)
    with col2:
        st.button("저장", use_container_width=True, on_click=save_character, disabled=st.session_state.get("name")=="" or not name_unique_check(st.session_state.get("name")))
    if st.session_state.get("name") == "":
        st.error("이름을 입력하세요!")
    else:
        if not name_unique_check(st.session_state.get("name")):
            st.error("이미 존재하는 이름입니다!")
    st.caption("추가된 캐릭터는 삭제가 불가능하며, 모든 사용자가 확인 및 대화할 수 있습니다.")

st.button("➕ 캐릭터 추가", on_click=add_character, use_container_width=True)

with st.spinner("불러오는 중..."):
    snaps = db.collection("characters").get()

with st.container(border=True):
    for snap in snaps:
            with st.expander(snap.id):
                st.write(f"성별: {snap.get('gender')}")
                st.write(f"MBTI: {snap.get('mbti')}")
                st.write(f"대화스타일: {snap.get('conversation_style')}")
                st.write(f"기타: {snap.get('etc')}")