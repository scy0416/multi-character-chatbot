import streamlit as st

with st.container(border=True):
    st.title("멀티 캐릭터 챗봇")

    with st.container(border=True):
        st.write("제작자: 송찬영")
        st.write("Github링크: https://github.com/scy0416")
        st.write("메일 주소: scy0416@gmail.com")

    st.write("캐릭터를 직접 정의하고, 정의한 캐릭터들과 함께 채팅을 즐길 수 있는 서비스입니다.")
    st.caption("이 프로젝트는 경기갭이어프로그램의 지원금으로 만들어졌습니다.")
    st.divider()

    if not st.user.is_logged_in:
        if st.button("구글로 로그인"):
            st.login()
    else:
        if st.button("로그아웃"):
            st.logout()