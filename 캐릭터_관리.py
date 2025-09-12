import streamlit as st

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

st.title("캐릭터 관리")

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

    def save_character():
        print(f"""이름: {st.session_state.get("name")}
성별: {st.session_state.get("gender")}
MBTI: {st.session_state.get("주의초점")}{st.session_state.get("인식기능")}{st.session_state.get("판단기능")}{st.session_state.get("생활양식")}
대화스타일: {st.session_state.get("conversation_style")}""")

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

    col1, col2 = st.columns(2)
    with col1:
        st.button("리셋", use_container_width=True, on_click=reset_input)
    with col2:
        st.button("저장", use_container_width=True, on_click=save_character, disabled=st.session_state.get("name")=="")
    st.caption("추가된 캐릭터는 삭제가 불가능하며, 모든 사용자가 확인 및 대화할 수 있습니다.")

st.button("➕ 캐릭터 추가", on_click=add_character, use_container_width=True)