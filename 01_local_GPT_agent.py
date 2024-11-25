import os
from dotenv import load_dotenv
from langchain_teddynote import logging

import streamlit as st

from langchain_core.documents import Document
from messages_util import AgentStreamParser, AgentCallbacks

from agent_util import create_agent_with_chat_history
from retriever import create_retriever_from_PDF
from typing import List, Union

import time

# Load API KEY
load_dotenv()

# Set project name
logging.langsmith("Local GPT")

st.title("Local GPT 💬")

# Create sidebar
with st.sidebar:
    # Upload file
    uploaded_files = st.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
    )

    # Select model
    llm_mode = st.radio(
        "Select Mode",
        ["***ChatGPT***", "***Private GPT***"],
        captions=["***GPT-4o-mini***", "***Llama-3.1-8B***"],
    )
    # Create clear button
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


if uploaded_files:
    # Cache file
    retriever = create_retriever_from_PDF(uploaded_files)
    uploaded_files = None


# Define constants
class MessageRole:
    USER = "user"  # Type of user message
    ASSISTANT = "assistant"  # Type of assistant message


class MessageType:
    """
    Define message types
    """

    SOURCE = "source"  # Source message
    TEXT = "text"  # Text message
    RELATED_INFO = "related_info"  # Related information message


def tool_callback(tool) -> None:
    """
    Callback function to process tool execution results

    Args:
        tool (dict): Tool execution information
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
    Callback function to process observation results

    Args:
        observation (dict): Observation results
    """
    if "observation" in observation:
        action = observation["action"]
        tool_name = action.tool
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["localGPT_messages"][-1][
                1
            ].clear()  # Delete last message if error occurs

        st.session_state["observation"] = {"tool": tool_name, "observation": obs}
        if tool_name == "search_tavily":
            add_message(MessageRole.ASSISTANT, [MessageType.SOURCE, obs])
            print_source_message(obs)
        # elif tool_name == "create_related_info":
        #     add_message(MessageRole.ASSISTANT, [MessageType.RELATED_INFO, obs])
        #     print_related_info(obs)


def result_callback(result: str) -> None:
    """
    Callback function to process final results

    Args:
        result (str): Final results
    """
    global ai_answer  # Declare as global variable

    if not hasattr(result_callback, "ai_answer"):
        result_callback.ai_answer = ""  # Initialize

    result_callback.ai_answer += result
    st.markdown(result_callback.ai_answer)
    add_message(MessageRole.ASSISTANT, [MessageType.TEXT, result_callback.ai_answer])


# Print previous conversation
def print_messages():
    """
    Print saved messages
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
    Add new message

    Args:
        role (MessageRole): Message role (user or assistant)
        content (List[Union[MessageType, str]]): Message content
    """
    messages = st.session_state["localGPT_messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend(
            [content]
        )  # Consecutive messages with the same role are combined
    else:
        messages.append([role, [content]])  # New messages with a new role are added


def print_source_message(observation: List[Document]):
    """
    Print source messages

    Args:
        observation (List[Document]): Observation results
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

    # last column is displayed as a popover for the remaining documents
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
    Print related information

    Args:
        related_info (List[str]): Related information list
    """
    with st.expander("Related Information"):
        for i, question in enumerate(related_info):
            key = f"related_info_{i}_{hash(question)}"
            if st.button(question, key=key):
                # Do not create chat_message directly, only update session state
                st.session_state["user_input"] = question
                st.rerun()


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["localGPT_messages"] = []

if llm_mode != st.session_state["selected_mode"]:
    st.session_state["localGPT_agent"] = create_agent_with_chat_history(llm_mode)
    st.session_state["selected_mode"] = llm_mode


def execute_agent(user_input: str):
    """
    Process user's question and generate response

    Args:
        query (str): User's question
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


# Print previous conversation
print_messages()

# Process new input (chat input or related info button click)
if temp_input := st.chat_input("Ask me anything!"):
    st.session_state["user_input"] = temp_input

if st.session_state["user_input"]:
    user_input = st.session_state["user_input"]
    st.chat_message("user").write(user_input)
    st.session_state["user_input"] = ""  # Reset input

    add_message(MessageRole.USER, [MessageType.TEXT, user_input])

    # Run agent
    execute_agent(user_input)
