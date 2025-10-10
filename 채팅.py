import uuid
from typing import List, TypedDict, Dict
from pydantic import BaseModel, Field
import re

import streamlit as st
from app import get_firestore
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

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
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)

client = MongoClient(st.secrets["mongodb"]["MONGODB_URI"], tls=True)
saver = MongoDBSaver(
    client,
    db_name=st.secrets["mongodb"]["DB_NAME"],
    checkpoint_collection_name="checkpoints",
    writes_collection_name="checkpoints_writes"
)

st.session_state.chat_id = st.query_params.get("cid", None)

@st.fragment
def session_manage():
    @st.dialog("새 채팅 시작")
    def new_chat_character_select():
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
                        st.write(f"대화 스타일: {snap.get('conversation_style')}")
                        st.write(f"기타: {snap.get('etc')}")

        selected = [snap.id for snap in snaps if st.session_state.get(f"newchat:{snap.id}")]
        if st.button("새 채팅 시작", use_container_width=True, disabled=len(selected)<=0):
            new_chat_id = str(uuid.uuid4())

            characters_info = {}

            for character in selected:
                characters_info[character] = db.collection("characters").document(character).get().to_dict()
                characters_info[character]["lore"] = ""

            db.collection("chats").document(st.user.sub).collection(new_chat_id).document("info").set({"characters": selected})
            db.collection("chats").document(st.user.sub).collection(new_chat_id).document("participants").set(characters_info)

            st.session_state.chat_id = new_chat_id
            st.query_params.cid = new_chat_id
            st.rerun(scope="app")
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
                for sub in snap.reference.collections():delete_collection_recursive(sub, batch_size)

            batch = db.batch()
            for snap in docs:
                batch.delete(snap.reference)
            batch.commit()

    def del_chat(chat_id):
        col_ref = db.collection("chats").document(st.user.sub).collection(chat_id)
        with st.spinner(f"채팅({chat_id}) 삭제 중..."):
            delete_collection_recursive(col_ref, batch_size=300)
        saver.delete_thread(chat_id)
        st.query_params.clear()
        st.toast("삭제 완료 ✅")
        st.rerun(scope="app")

    with st.container(border=True):
        for col in cols:
            characters = col.document("info").get().get("characters")

            if len(characters) > 3:
                st.write(f"{characters[0]}, {characters[1]}, {characters[2]}, ... +{len(characters) - 3}")
            else:
                st.write(", ".join(characters))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("시작", use_container_width=True, key=f"{col.id}1"):
                    st.session_state.chat_id = col.id
                    st.query_params.cid = col.id
                    st.rerun(scope="app")
            with col2:
                st.button("삭제", use_container_width=True, type="primary", key=f"{col.id}2", on_click=del_chat, kwargs={"chat_id": col.id})

class State(TypedDict):
    history: List
    characters: Dict
    lead_speaker: str
    aside_speaker: str
    lead_turn: int
    aside_turn: int

class utterance_character(BaseModel):
    utterance_character: str = Field(description="가장 발화를 하고자 하는 캐릭터의 이름")

class Handoff(BaseModel):
    handoff: bool = Field(description="어사이드를 진행한 캐릭터의 핸드오프를 하고자 하는 의지")

def 사용자_발화(state: State):
    payload = interrupt({})
    user_text = payload["text"]
    state["history"].append(HumanMessage(content=user_text))
    state["lead_turn"] = 0
    return state

def 리드_욕구_확인(state: State):
    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 멀티 캐릭터 챗봇의 캐릭터들의 리드 대화 욕구를 측정해서 발화를 할 캐릭터를 정하는 조정자입니다.
당신에게는 지금까지의 대화내역과 캐릭터들에 대한 정보가 전달될 것이며, 그를 토대로 가장 발화를 일으킬만한 캐릭터를 딱 한 명 정하세요.
너무 한 캐릭터만 대화하지 않도록 골고루 말할 수 있게 선택하세요.
### 캐릭터의 정보
{characters}
### 대화 내역"""),
        MessagesPlaceholder("history")
    ])

    system_prompt = system_prompt_template.invoke({"characters": str(state["characters"]), "history": state["history"]})
    state["lead_speaker"] = llm.with_structured_output(utterance_character).invoke(system_prompt).utterance_character
    state["aside_speaker"] = ""
    state["aside_turn"] = 0
    return state

def 리드_생성(state: State):
    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 멀티 캐릭터 챗봇의 캐릭터 대화 생성자입니다.
당신에게는 지금까지의 대화 내역과 발화를 하고자 하는 캐릭터의 정보가 전달됩니다.

아래의 형식으로 출력하세요. 단, 캐릭터의 대화 스타일에 따라서 최대 5회까지 나눠서 발언할 수 있습니다.

<DIALOGUE speaker="{current_speaker}">
(대사 텍스트)
</DIALOGUE>

### 대화 내역"""),
        MessagesPlaceholder("history"),
        SystemMessagePromptTemplate.from_template("""### 발화자 캐릭터
이름: {current_speaker}
{character}""")
    ])

    system_prompt = system_prompt_template.invoke(
        {"current_speaker": state["lead_speaker"], "history": state["history"],
         "character": str(state["characters"][state["lead_speaker"]])})

    buffer = ""
    current_role = None
    speaker = None
    for chunk in llm.stream(system_prompt):
        if not chunk.content:
            continue

        token = chunk.content
        buffer += token

        # 마커 시작 감지
        if "<DIALOGUE" in buffer and ">" in buffer and current_role is None:
            tag_match = re.search(f"<DIALOGUE([^>]*)>", buffer)
            if tag_match:
                attrs = tag_match.group(1)
                match = re.search(f'speaker="([^"]+)"', attrs)
                speaker = match.group(1) if match else "캐릭터"
                current_role = f"dialogue:{speaker}"
                buffer = buffer.split(">", 1)[-1]
        # 마커 종료 감지
        if current_role and current_role.startswith("dialogue") and "</DIALOGUE>" in buffer:
            buffer = buffer.replace("</DIALOGUE>", "")
            final_dialogue = f"{speaker}: {buffer}"
            buffer = ""
            current_role = None
            speaker = None
            state["history"].append(AIMessage(content=final_dialogue))
    state["lead_turn"] += 1
    return state

def 어사이드_욕구_확인(state: State):
    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 멀티 캐릭터 챗봇의 캐릭터들의 어사이드 대화욕구를 측정해서 리드에 대한 어사이드 발화를 할 캐릭터를 정하는 조정자입니다.
어사이드는 리드 대화에 대한 끼어들기 대화로 끼어드는 캐릭터가 존재할 수도, 존재하지 않을 수도 있습니다.
당신에게는 지금까지의 대화내역과 리드 발화를 진행한 캐릭터를 제외한 캐릭터들에 대한 정보가 전달될 것이며, 그를 토대로 가장 발화를 일으킬만한 캐릭터를 정하세요.
끼어들 캐릭터가 없다면 빈 문자열을 반환하세요.
### 캐릭터 정보
{characters}
### 대화 내역"""),
        MessagesPlaceholder("history")
    ])

    aside_able_characters = ""
    for c in state["characters"]:
        if not (state["aside_speaker"] == "" and c == state["lead_speaker"]) and c != state["aside_speaker"]:
            aside_able_characters += str({c: state["characters"][c]})
    system_prompt = system_prompt_template.invoke({"characters": aside_able_characters, "history": state["history"]})
    state["aside_speaker"] = llm.with_structured_output(utterance_character).invoke(system_prompt).utterance_character
    return state

def 어사이드_생성(state: State):
    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 멀티 캐릭터 챗봇의 캐릭터 대화 생성자입니다.
        당신에게는 지금까지의 대화 내역과 발화를 하고자 하는 캐릭터의 정보가 전달됩니다.

        아래의 형식으로 출력하세요. 단, 캐릭터의 대화 스타일에 따라서 최대 5회까지 나눠서 발언할 수 있습니다.

        어사이드는 진행되고 있는 대화 주제에 대답하는 답변을 만들어내세요.

        <DIALOGUE speaker="{current_speaker}">
        (대사 텍스트)
        </DIALOGUE>

        ### 대화 내역"""),
        MessagesPlaceholder("history"),
        SystemMessagePromptTemplate.from_template("""### 발화자 캐릭터
        이름: {current_speaker}
        {character}""")
    ])

    system_prompt = system_prompt_template.invoke(
        {
            "current_speaker": state["aside_speaker"],
            "history": state["history"],
            "character": str(state["characters"][state["aside_speaker"]])
        }
    )

    buffer = ""
    current_role = None
    speaker = None
    for chunk in llm.stream(system_prompt):
        if not chunk.content:
            continue

        token = chunk.content
        buffer += token

        # 마커 시작 감지
        if "<DIALOGUE" in buffer and ">" in buffer and current_role is None:
            tag_match = re.search(f"<DIALOGUE([^>]*)>", buffer)
            if tag_match:
                attrs = tag_match.group(1)
                match = re.search(f'speaker="([^"]+)"', attrs)
                speaker = match.group(1) if match else "캐릭터"
                current_role = f"dialogue:{speaker}"
                buffer = buffer.split(">", 1)[-1]
        # 마커 종료 감지
        if current_role and current_role.startswith("dialogue") and "</DIALOGUE>" in buffer:
            buffer = buffer.replace("</DIALOGUE>", "")
            final_dialogue = f"{speaker}: {buffer}"
            buffer = ""
            current_role = None
            speaker = None
            state["history"].append(AIMessage(content=final_dialogue))
    state["aside_turn"] += 1
    return state

def 핸드오프_욕구_확인(state: State):
    if state["aside_speaker"] == "":
        return "사용자 발화"

    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 어사이드를 진행할 캐릭터가 핸드오프를 통해서 대화를 리드할지에 대해서 결정하는 조정자입니다.
핸드오프는 대화의 진행을 가로채서 주제를 바꾸는 행위입니다.
당신에게는 지금까지의 대화내역과 어사이드를 진행한 캐릭터에 대한 정보가 전달될 것이며, 이를 토대로 어사이드를 진행한 캐릭터가 핸드오프를 진행할지에 대해서 정하세요.
### 어사이드를 진행한 캐릭터 정보
{character}
### 대화내역"""),
        MessagesPlaceholder("history")
    ])

    system_prompt = system_prompt_template.invoke({"character": state["aside_speaker"], "history": state["history"]})
    if state["lead_turn"] < 2 and llm.with_structured_output(Handoff).invoke(system_prompt).handoff:
        return "핸드오프 중간처리"
    else:
        return "어사이드 생성"

def 핸드오프_처리(state:State):
    state["lead_speaker"] = state["aside_speaker"]
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("사용자 발화", 사용자_발화)
graph_builder.add_node("리드 욕구 확인", 리드_욕구_확인)
graph_builder.add_node("리드 생성", 리드_생성)
graph_builder.add_node("어사이드 욕구 확인", 어사이드_욕구_확인)
graph_builder.add_node("어사이드 생성", 어사이드_생성)
graph_builder.add_node("핸드오프 중간처리", 핸드오프_처리)

graph_builder.add_edge(START, "사용자 발화")
graph_builder.add_edge("사용자 발화", "리드 욕구 확인")
graph_builder.add_edge("리드 욕구 확인", "리드 생성")
graph_builder.add_conditional_edges("리드 생성", lambda state: "어사이드 욕구 확인" if len(state["characters"]) > 1 else "사용자 발화",
                                    ["어사이드 욕구 확인", "사용자 발화"])
graph_builder.add_conditional_edges("어사이드 욕구 확인", 핸드오프_욕구_확인, ["핸드오프 중간처리", "어사이드 생성", "사용자 발화"])
graph_builder.add_conditional_edges("어사이드 생성", lambda state: "어사이드 욕구 확인" if state["aside_turn"] < 5 else "사용자 발화",
                                    ["어사이드 욕구 확인", "사용자 발화"])
graph_builder.add_edge("핸드오프 중간처리", "리드 생성")

graph = graph_builder.compile(checkpointer=saver)

with st.sidebar:
    session_manage()

if st.query_params.get("cid", None) is not None:
    cid = st.query_params.cid
    #chat(st.session_state.chat_id)
    with st.spinner("불러오는 중..."):
        chat_info = db.collection("chats").document(st.user.sub).collection(cid).document("info").get().to_dict()
        chat_participants = db.collection("chats").document(st.user.sub).collection(cid).document(
            "participants").get().to_dict()

    config = {"configurable": {"thread_id": cid}}

    tup = saver.get_tuple(config)
    exists_latest = (tup is not None)

    if not exists_latest:
        chat_info = db.collection("chats").document(st.user.sub).collection(cid).document("info").get().to_dict()
        chat_participants = db.collection("chats").document(st.user.sub).collection(cid).document(
            "participants").get().to_dict()

        graph.invoke(State(
            history=[],
            characters=chat_participants,
            lead_speaker="",
            aside_speaker="",
            lead_turn=0,
            aside_turn=0
        ), config)

    config, checkpoint, metadata, parent_config, pending_writes = saver.get_tuple(config)

    message_box = st.container(border=True)

    for message in checkpoint["channel_values"]["history"]:
        if isinstance(message, AIMessage):
            speaker = message.content.split(":")[0]
            with message_box.chat_message(speaker):
                st.write(message.content)
        else:
            with message_box.chat_message("human"):
                st.write(message.content)

    if pending_writes and any("__interrupt__" in w for w in pending_writes):
        prompt = st.chat_input("대화를 입력하세요")
        if prompt:
            with message_box:
                with st.chat_message("user"):
                    st.write(prompt)
            cmd = Command(resume={"text": prompt})

            buffer = ""
            current_role = None
            speaker = None
            container = None
            placeholder = None

            for event in graph.stream(cmd, config, stream_mode="messages"):
                node_name = event[1].get("langgraph_node")

                if node_name in ["리드 생성", "어사이드 생성"]:
                    if not event[0].content:
                        continue

                    buffer += event[0].content

                    if "<DIALOGUE" in buffer and ">" in buffer and current_role is None:
                        tag_match = re.search(r"<DIALOGUE([^>]*)>", buffer)
                        if tag_match:
                            attrs = tag_match.group(1)
                            match = re.search(f'speaker="([^"]+)"', attrs)
                            speaker = match.group(1) if match else "캐릭터"
                            current_role = f"dialogue:{speaker}"
                            buffer = buffer.split(">", 1)[-1]
                            container = message_box.chat_message(speaker)
                            placeholder = container.empty()
                    if current_role and not any(tag in buffer for tag in ["</DIALOGUE>"]):
                        placeholder.write(f"{speaker}: {buffer}")
                    if current_role and current_role.startswith("dialogue") and "</DIALOGUE>" in buffer:
                        buffer = buffer.replace("</DIALOGUE>", "")
                        placeholder.write(f"{speaker}: {buffer}")
                        buffer = ""
                        current_role = None
                        speaker = None
    else:
        st.chat_input("대화를 입력하세요.", disabled=True)
        graph.invoke(None, config)
        st.rerun()