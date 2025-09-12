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
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.subheader("캐릭터 관리")
            st.write("이미 만들어진 캐릭터들을 확인하고, 새로운 캐릭터를 만들어보세요!")
        with col2:
            st.subheader("채팅 관리")
            st.write("캐릭터들과의 채팅을 시작하거나 관리해보세요!")

        if st.button("구글로 로그인"):
            st.login()
    else:
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.subheader("캐릭터 관리")
            st.write("이미 만들어진 캐릭터들을 확인하고, 새로운 캐릭터를 만들어보세요!")
            if st.button("캐릭터 관리 페이지로 이동", use_container_width=True):
                st.switch_page("캐릭터_관리.py")
        with col2:
            st.subheader("채팅 관리")
            st.write("캐릭터들과의 채팅을 시작하거나 관리해보세요!")
            if st.button("채팅 관리 페이지로 이동", use_container_width=True):
                st.switch_page("채팅_관리.py")

        if st.button("로그아웃"):
            st.logout()