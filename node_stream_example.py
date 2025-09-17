import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    query: str

llm = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)

def node_a(state: State):
    llm.invoke(state["query"])

def node_b(state: State):
    llm.invoke(state["query"])

graph = StateGraph(State)

graph.add_node("노드A", node_a)
graph.add_node("노드B", node_b)

graph.add_edge(START, "노드A")
graph.add_edge("노드A", "노드B")
graph.add_edge("노드B", END)

app = graph.compile()

st.set_page_config("노드마다의 스트리밍 테스트", layout="wide")
st.title("노드마다 따로 스트리밍")

col1, col2 = st.columns(2)
with col1.container(border=True):
    placeholder_a = st.empty()
    partial_a = ""
with col2.container(border=True):
    placeholder_b = st.empty()
    partial_b = ""

if st.button("테스트 시작"):
    partial_a = ""
    partial_b = ""
    placeholder_a.write(partial_a)
    placeholder_b.write(partial_b)

    events = app.stream({"query": "인공지능에 대한 5문장의 시를 만들어줘"}, stream_mode="messages")

    for event in events:
        node_name = event[1].get("langgraph_node")

        match node_name:
            case "노드A":
                partial_a += event[0].content
                placeholder_a.write(partial_a)
            case "노드B":
                partial_b += event[0].content
                placeholder_b.write(partial_b)