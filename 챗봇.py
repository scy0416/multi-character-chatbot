from typing import List, TypedDict, Dict
from pydantic import BaseModel, Field
import re

import streamlit as st
from app import get_firestore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph_checkpoint_firestore import FirestoreSaver
from utils_auth import activate_adc_from_secrets
from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.redis import RedisSaver
#from redis import Redis
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

if not st.user.is_logged_in:
    st.switch_page("타이틀.py")

if not "chat_id" in st.session_state or st.session_state.chat_id is None:
    st.switch_page("타이틀.py")

db = get_firestore()
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)

class State(TypedDict):
    history: List
    characters: Dict
    speaker: str
    #await_human: bool

class Ctx(TypedDict):
    speak_desire: Dict

def 사용자_발화(state: State, runtime: Runtime[Ctx]):
    payload = interrupt({})
    user_text = payload["text"]
    state["history"].append(HumanMessage(content=user_text))
    return state

def 테스트_노드(state: State, runtime: Runtime[Ctx]):
    print(state["history"])

class utterance_character(BaseModel):
    utterance_character: str = Field(description="가장 발화를 하고자 하는 캐릭터입니다.")
def 리드_욕구_확인(state: State, runtime: Runtime[Ctx]):
    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 멀티 캐릭터 챗봇의 캐릭터들의 리드 대화욕구를 측정해서 발화를 할 캐릭터를 정하는 조정자입니다.
당신에게는 지금까지의 대화내역과 캐릭터들에 대한 정보가 전달될 것이며, 그를 토대로 가장 발화를 일으킬만한 캐릭터를 딱 한 명 정하세요.
너무 한 캐릭터만 대화하지 않도록 골고루 말할 수 있게 선택하세요.
### 캐릭터 정보
{characters}
### 대화 내역"""),
        MessagesPlaceholder("history")
    ])

    system_prompt = system_prompt_template.invoke({"characters": str(state["characters"]), "history": state["history"]})
    state["speaker"] = llm.with_structured_output(utterance_character).invoke(system_prompt).utterance_character
    return state
    pass

def 리드_생성(state: State, runtime: Runtime[Ctx]):
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

    system_prompt = system_prompt_template.invoke({"current_speaker": state["speaker"], "history": state["history"], "character": str(state["characters"][state["speaker"]])})

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
    return state
    pass

def 어사이드_욕구_확인(state: State, runtime: Runtime[Ctx]):
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
        if c == state["speaker"]:
            continue
        #aside_able_characters += str(state["characters"][c])
        aside_able_characters += str({c:state["characters"][c]})
    system_prompt = system_prompt_template.invoke({"characters": aside_able_characters, "history": state["history"]})
    state["speaker"] = llm.with_structured_output(utterance_character).invoke(system_prompt).utterance_character
    return state
    pass

def 어사이드_생성(state: State, runtime: Runtime[Ctx]):
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

    system_prompt = system_prompt_template.invoke({"current_speaker": state["speaker"], "history": state["history"],
                                                   "character": str(state["characters"][state["speaker"]])})

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
    return state
    pass

class handoff(BaseModel):
    handoff: bool = Field(description="어사이드를 진행한 캐릭터의 핸드오프를 하고자 하는 의지")
def 핸드오프_욕구_확인(state: State, runtime: Runtime[Ctx]):
    if state["speaker"] == '':
        return "사용자 발화"

    system_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""당신은 어사이드를 진행한 캐릭터가 핸드오프를 통해서 대화를 리드할지에 대해서 결정하는 조정자입니다.
핸드오프는 어사이드를 진행한 후에 캐릭터가 대화를 가로채서 진행하는 행위입니다.
당신에게는 지금까지의 대화내역과 어사이드를 진행한 캐릭터에 대한 정보가 전달될 것이며, 이를 토대로 어사이드를 진행한 캐릭터가 핸드오프를 진행할지에 대해서 정하세요.
### 어사이드를 진행한 캐릭터 정보
{character}
### 대화 내역"""),
        MessagesPlaceholder("history")
    ])

    system_prompt = system_prompt_template.invoke({"character": state["speaker"], "history": state["history"]})
    #state["speaker"] = llm.with_structured_output(utterance_character).invoke(system_prompt).utterance_character
    if llm.with_structured_output(handoff).invoke(system_prompt).handoff:
        return "리드 생성"
    else:
        return "어사이드 생성"

graph_builder = StateGraph(State, context_schema=Ctx)

graph_builder.add_node("사용자 발화", 사용자_발화)
#graph_builder.add_node("테스트 노드", 테스트_노드)
graph_builder.add_node("리드 욕구 확인", 리드_욕구_확인)
graph_builder.add_node("리드 생성", 리드_생성)
graph_builder.add_node("어사이드 욕구 확인", 어사이드_욕구_확인)
graph_builder.add_node("어사이드 생성", 어사이드_생성)

graph_builder.add_edge(START, "사용자 발화")
#graph_builder.add_edge("사용자 발화", "테스트 노드")
graph_builder.add_edge("사용자 발화", "리드 욕구 확인")
graph_builder.add_edge("리드 욕구 확인", "리드 생성")
#graph_builder.add_edge("리드 생성", "사용자 발화")
graph_builder.add_conditional_edges("리드 생성", lambda state, runtime: "어사이드 욕구 확인" if len(state["characters"]) > 1 else "사용자 발화", ["어사이드 욕구 확인", "사용자 발화"])
#graph_builder.add_conditional_edges("어사이드 욕구 확인", lambda state, runtime: "어사이드 생성" if state["speaker"] else "사용자 발화", ["어사이드 생성", "사용자 발화"])
#graph_builder.add_edge("어사이드 생성", "사용자 발화")
#graph_builder.add_conditional_edges("어사이드 생성", 핸드오프_욕구_확인, ["리드 생성", "사용자 발화"])
graph_builder.add_conditional_edges("어사이드 욕구 확인", 핸드오프_욕구_확인, ["리드 생성", "어사이드 생성", "사용자 발화"])
graph_builder.add_edge("어사이드 생성", "사용자 발화")

# graph = graph_builder.compile(checkpointer=InMemorySaver())
# if "adc_key_path" not in st.session_state:
#     st.session_state["adc_key_path"] = activate_adc_from_secrets()
# memory = FirestoreSaver(project_id="multi-character-chat", checkpoints_collection="checkpoints", writes_collection="checkpoints_writes")
# graph = graph_builder.compile(checkpointer=memory)
#saver = RedisSaver.from_conn_string(st.secrets['redis']['REDIS_ENDPOINT'])

# client = Redis(
#     host=st.secrets["redis"]["host"],
#     port=st.secrets["redis"]["port"],
#     password=st.secrets["redis"]["password"],
#     ssl=False,
#     ssl_cert_reqs="required",
#     decode_responses=False
# )
# saver=RedisSaver(redis_client=client)
client = MongoClient(st.secrets["mongodb"]["MONGODB_URI"], tls=True)
saver = MongoDBSaver(
    client,
    db_name=st.secrets["mongodb"]["DB_NAME"],
    checkpoint_collection_name="checkpoints",
    writes_collection_name="checkpoints_writes"
)
#saver = MongoDBSaver.from_conn_string(st.secrets["mongodb"]["MONGODB_URI"], st.secrets["mongodb"]["DB_NAME"])
#saver.setup()
graph = graph_builder.compile(checkpointer=saver)
#graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
print(graph.get_graph().draw_mermaid())

#st.write(st.session_state.chat_id)

# st.write(db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("info").get().to_dict())
# st.write(db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("participants").get().to_dict())

with st.spinner("불러오는 중..."):
    chat_info = db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("info").get().to_dict()
    chat_participants = db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("participants").get().to_dict()

# st.write(chat_info)
# st.write(chat_participants)

config = {"configurable": {"thread_id": st.session_state.chat_id}}

#tup = memory.get_tuple(config)
tup = saver.get_tuple(config)
exists_latest = (tup is not None)

if exists_latest:
    #print("체크 포인트가 존재함")
    pass
else:
    #print("체크 포인트가 존재하지 않음")
    chat_info = db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("info").get().to_dict()
    chat_participants = db.collection("chats").document(st.user.sub).collection(st.session_state.chat_id).document("participants").get().to_dict()

    graph.invoke(State(history=[], characters=chat_participants, speaker=""), config)

config, checkpoint, metadata, parent_config, pending_writes = saver.get_tuple(config)
#print(checkpoint)

#print(config)
#print(checkpoint['channel_values']['history'])
#print(pending_writes)

#print(checkpoint["channel_values"])
#print("__interrupt__" in pending_writes[0])

message_box = st.container(border=True)

for message in checkpoint['channel_values']['history']:
    if isinstance(message, AIMessage):
        with message_box.chat_message("ai"):
            st.write(message.content)
    else:
        with message_box.chat_message("human"):
            st.write(message.content)

if pending_writes and any("__interrupt__" in w for w in pending_writes):
    prompt = st.chat_input("대화를 입력하세요.")
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

            # match node_name:
            #     case "리드 생성":
            #         if not event[0].content:
            #             continue
            #
            #         buffer += event[0].content
            #
            #         if "<DIALOGUE" in buffer and ">" in buffer and current_role is None:
            #             tag_match = re.search(r"<DIALOGUE([^>]*)>", buffer)
            #             if tag_match:
            #                 attrs = tag_match.group(1)
            #                 match = re.search(f'speaker="([^"]+)"', attrs)
            #                 speaker = match.group(1) if match else "캐릭터"
            #                 current_role = f"dialogue:{speaker}"
            #                 buffer = buffer.split(">", 1)[-1]
            #                 container = message_box.chat_message(speaker)
            #                 placeholder = container.empty()
            #         if current_role and not any(tag in buffer for tag in ["</DIALOGUE>"]):
            #             placeholder.write(f"{speaker}: {buffer}")
            #         if current_role and current_role.startswith("dialogue") and "</DIALOGUE>" in buffer:
            #             buffer = buffer.replace("</DIALOGUE>", "")
            #             placeholder.write(f"{speaker}: {buffer}")
            #             buffer = ""
            #             current_role = None
            #             speaker = None
else:
    st.chat_input("대화를 입력하세요.", disabled=True)
    graph.invoke(None, config)
    st.rerun()


# for v in graph.stream(State(history=[], characters=chat_participants, speaker=""), config, stream_mode="values"):
#     pass
#
# if prompt := st.chat_input("대화를 입력하세요."):
#     cmd = Command(resume={"text": prompt})
#
#     for v in graph.stream(cmd, config, stream_mode="values"):
#         pass