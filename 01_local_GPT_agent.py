import os
from dotenv import load_dotenv
from langchain_teddynote import logging

import streamlit as st

from langchain_core.documents import Document
from messages_util import AgentStreamParser, AgentCallbacks

from agent_util import create_agent_with_chat_history

from typing import List, Union

import time  # 파일 상단에 추가

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Local GPT")

st.title("Local GPT 💬")

# 사이드바 생성
with st.sidebar:
    # # 파일 업로드
    # uploaded_files = st.file_uploader(
    #     "Upload a file",
    #     type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
    #     accept_multiple_files=True,
    # )

    # 모델 선택 메뉴
    llm_mode = st.radio(
        "Select Mode",
        ["***ChatGPT***", "***Private GPT***"],
        captions=["***GPT-4o-mini***", "***Llama-3.1-8B***"],
    )
    # 초기화 버튼 생성
    clear_btn = st.button("New Chat")


if "localGPT_messages" not in st.session_state:
    st.session_state["localGPT_messages"] = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "observation" not in st.session_state:
    st.session_state["observation"] = {}

if "selected_mode" not in st.session_state:
    st.session_state["selected_mode"] = llm_mode

if "localGPT_agent" not in st.session_state:
    st.session_state["localGPT_agent"] = create_agent_with_chat_history(llm_mode)


# 상수 정의
class MessageRole:
    USER = "user"  # 사용자 메시지 타입
    ASSISTANT = "assistant"  # 어시스턴트 메시지 타입


class MessageType:
    """
    메시지 타입을 정의하는 클래스입니다.
    """

    SOURCE = "source"  # 소스 메시지
    TEXT = "text"  # 텍스트 메시지
    RELATED_INFO = "related_info"  # 관련 정보 메시지


def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict): 실행된 도구 정보
    """
    if tool_name := tool.get("tool"):
        tool_input = tool.get("tool_input", {})
        query = tool_input.get("query")
        if query:
            status_container = st.empty()
            with status_container.status(f"🔍 Searching for **{query}**...") as status:
                status.update(state="complete", expanded=False)
                time.sleep(1)
            status_container.empty()


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        action = observation["action"]
        tool_name = action.tool
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["localGPT_messages"][-1][
                1
            ].clear()  # 에러 발생 시 마지막 메시지 삭제

        st.session_state["observation"] = {"tool": tool_name, "observation": obs}
        if tool_name == "search_tavily":
            add_message(MessageRole.ASSISTANT, [MessageType.SOURCE, obs])
            print_source_message(obs)
        # elif tool_name == "create_related_info":
        #     add_message(MessageRole.ASSISTANT, [MessageType.RELATED_INFO, obs])
        #     print_related_info(obs)


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    global ai_answer  # 전역 변수로 선언

    if not hasattr(result_callback, "ai_answer"):
        result_callback.ai_answer = ""  # 초기화

    result_callback.ai_answer += result
    st.markdown(result_callback.ai_answer)
    add_message(MessageRole.ASSISTANT, [MessageType.TEXT, result_callback.ai_answer])


# 이전 대화를 출력
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["localGPT_messages"]:
        if role == MessageRole.USER:
            user = st.chat_message("user")
        elif role == MessageRole.ASSISTANT:
            assistant = st.chat_message("assistant")
        for content in content_list:
            if isinstance(content, list):
                message_type, message_content = content
                if role == MessageRole.USER:
                    user.write(message_content)
                elif role == MessageRole.ASSISTANT:
                    with assistant:
                        if message_type == MessageType.TEXT:
                            st.markdown(message_content)
                        elif message_type == MessageType.SOURCE:
                            print_source_message(message_content)
                        elif message_type == MessageType.RELATED_INFO:
                            print_related_info(message_content)


# 새로운 메시지를 추가
def add_message(role: MessageRole, content: List[Union[MessageType, any]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["localGPT_messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


def print_source_message(observation: List[Document]):
    """
    소스 메시지를 출력하는 함수입니다.
    """

    cols = st.columns(4)

    for i in range(3):
        if i < len(observation):
            with cols[i]:
                link_container = st.empty()
                link_container.link_button(
                    label=observation[i].metadata["title"],
                    url=observation[i].metadata["source"],
                    use_container_width=True,
                )

    # 마지막 컬럼은 popover로 나머지 문서들 표시
    with cols[3]:
        if len(observation) > 3:
            more_container = st.empty()
            with more_container.popover("More Sources"):
                for doc in observation[3:]:
                    doc_container = st.empty()
                    doc_container.link_button(
                        label=doc.metadata["title"],
                        url=doc.metadata["source"],
                        use_container_width=True,
                    )


def print_related_info(related_info: List[str]):
    """
    관련 정보를 출력하는 함수

    Args:
        related_info (List[str]): 출력할 관련 정보 리스트
    """
    with st.expander("Related Information"):
        for i, question in enumerate(related_info):
            key = f"related_info_{i}_{hash(question)}"
            if st.button(question, key=key):
                # 직접 chat_message를 생성하지 않고 session state만 업데이트
                st.session_state["user_input"] = question
                st.rerun()


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["localGPT_messages"] = []


def execute_agent(user_input: str):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """

    localGPT_agent = st.session_state["localGPT_agent"]
    response = localGPT_agent.stream(
        {"input": user_input}, config={"configurable": {"session_id": "localGPT"}}
    )

    parser_callback = AgentCallbacks(
        tool_callback=tool_callback,
        observation_callback=observation_callback,
        result_callback=result_callback,
    )
    agent_stream_parser = AgentStreamParser(parser_callback)

    with st.chat_message("assistant"):

        for step in response:
            agent_stream_parser.process_agent_steps(step)

    if observation := st.session_state["observation"]:
        if observation["tool"] == "create_related_info":
            add_message(
                MessageRole.ASSISTANT,
                [MessageType.RELATED_INFO, observation["observation"]],
            )
            print_related_info(observation["observation"])
    st.session_state["observation"] = {}


# 이전 대화 기록 출력
print_messages()

# 새로운 입력 처리 (chat input 또는 related info 버튼 클릭)
if temp_input := st.chat_input("Ask me anything!"):
    st.session_state["user_input"] = temp_input

if st.session_state["user_input"]:
    user_input = st.session_state["user_input"]
    st.chat_message("user").write(user_input)
    st.session_state["user_input"] = ""  # 입력 초기화

    add_message(MessageRole.USER, [MessageType.TEXT, user_input])

    # 에이전트 실행
    execute_agent(user_input)
