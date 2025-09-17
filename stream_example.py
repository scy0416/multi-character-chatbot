import streamlit as st
from langchain_openai import ChatOpenAI
import re
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)

st.set_page_config(page_title="🎮 마커 기반 채팅 스트리밍")
st.title("마커 감지 + st.chat_message 출력")

user_action = st.text_input("플레이어 행동", "사찰 입구로 들어간다")

if st.button("진행"):
    current_role = None
    buffer = ""
    container = None
    text_area = None
    speaker = None

    prompt = f"""
    너는 텍스트 어드벤처 게임 마스터다.
    사용자의 행동: {user_action}

    아래 형식으로 출력하라. 단, 나레이션과 대사는 랜덤하게 발생할 수 있느며, 3~5번 반복하라.:

    <NARRATION>
    (나레이션 텍스트)
    </NARRATION>

    <DIALOGUE speaker="이름">
    (대사 텍스트)
    </DIALOGUE>
    """

    for chunk in llm.stream(prompt):
        if not chunk.content:
            continue

        token = chunk.content
        buffer += token

        # --- 1. 마커 시작 감지 ---
        if "<NARRATION>" in buffer:
            current_role = "narration"
            buffer = buffer.split("<NARRATION>", 1)[-1]  # 태그 제거
            container = st.chat_message("assistant")  # 나레이션은 assistant 역할
            text_area = container.empty()

        elif "<DIALOGUE" in buffer and ">" in buffer and current_role is None:
            # 시작 태그 전체 감지: <DIALOGUE ...>
            tag_match = re.search(r"<DIALOGUE([^>]*)>", buffer)
            if tag_match:
                attrs = tag_match.group(1)
                match = re.search(r'speaker="([^"]+)"', attrs)
                speaker = match.group(1) if match else "캐릭터"
                current_role = f"dialogue:{speaker}"
                # 태그 제거 후 나머지 본문만 남김
                buffer = buffer.split(">", 1)[-1]
                container = st.chat_message(speaker)
                text_area = container.empty()
                #text_area.markdown(f'{speaker}: ')

        # --- 2. 출력 중 ---
        if current_role and not any(tag in buffer for tag in ["</NARRATION>", "</DIALOGUE>"]):
            if current_role != "narration":
                text_area.markdown(f'{speaker}: {buffer}')
            else:
                text_area.markdown(buffer)

        # --- 3. 마커 종료 감지 ---
        if current_role == "narration" and "</NARRATION>" in buffer:
            buffer = buffer.replace("</NARRATION>", "")
            text_area.markdown(buffer)
            buffer = ""
            current_role = None

        elif current_role and current_role.startswith("dialogue") and "</DIALOGUE>" in buffer:
            buffer = buffer.replace("</DIALOGUE>", "")
            text_area.markdown(f'{speaker}: {buffer}')
            buffer = ""
            current_role = None
            speaker = None
