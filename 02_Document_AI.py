import os
import streamlit as st

from langchain_core.messages.chat import ChatMessage

from dotenv import load_dotenv
from langchain_teddynote import logging
from layout_parser import graph_document_ai, GraphState


from constants import PROGRESS_MESSAGE_GRAPH_NODES
from output import clean_cache_files
from document_utils import download_files, check_file_type
from retriever import create_ensemble_retriever
from chat_utils import create_chain

# Load API KEY information
load_dotenv()

# Enter project name
logging.langsmith("Document AI")

# Create cache directory
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# Create files directory for file upload
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# Create embeddings directory for embedding files
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("Document AI 💬")

# Code to run only once
if "document_messages" not in st.session_state:
    # Session state to store conversation history
    st.session_state["document_messages"] = []

if "retriever" not in st.session_state:
    # Session state for document retriever
    st.session_state["retriever"] = None

if "chain" not in st.session_state:
    # Session state for chain
    st.session_state["chain"] = None

if "filepath" not in st.session_state:
    # Session state for file path
    st.session_state["filepath"] = None


# Create sidebar
with st.sidebar:
    # File upload
    uploaded_files = st.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
    )
    # Select model
    translate_lang = st.selectbox("Translate", ["Korean", "English", "German"], index=0)
    # Translate toggle
    translate_toggle = st.checkbox("Enable Translation", value=False)
    # Create AI Translate & Summary button
    start_btn = st.button("Document AI")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["document_messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시리를 추가
def add_message(role, message):
    st.session_state["document_messages"].append(
        ChatMessage(role=role, content=message)
    )


# Save file to cache (processing time-consuming tasks)
def cache_file(files):
    # Check file extension
    file_extension = check_file_type(files[0].name)

    # Save uploaded files to cache directory
    file_paths = []

    if file_extension == "pdf":
        if len(files) > 1:  # If there is more than one file
            st.warning("PDF file only support one file")
            st.session_state["filepath"] = None
        else:
            file_content = files[0].read()  # Read the first file
            file_path = f"./.cache/files/{files[0].name}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            # Additional PDF file processing code
            file_paths.append(file_path)
            st.session_state["filepath"] = file_paths
    elif file_extension == "image":
        for file in files:
            file_content = file.read()  # Read the file
            file_path = f"./.cache/files/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            file_paths.append(file_path)
        st.session_state["filepath"] = file_paths
    else:
        st.warning("Not supported file type")
        st.session_state["filepath"] = None


if uploaded_files:
    # Save file to cache
    cache_file(uploaded_files)
    uploaded_files = None


def process_graph(file_paths):
    # Create container for progress display
    progress_bar = st.progress(0)
    status_container = st.empty()

    state = GraphState()

    filetype = check_file_type(file_paths[0])
    if filetype == "pdf":
        file_paths = file_paths[0]
    elif filetype == "image":
        file_paths = file_paths
    else:
        st.error("Not supported file type")
        return

    # Create graph
    message_dict = PROGRESS_MESSAGE_GRAPH_NODES
    graph = graph_document_ai(translate_toggle)
    inputs = {
        "filepath": file_paths,
        "filetype": filetype,
        "batch_size": 10,
        "translate_lang": translate_lang,
        "translate_toggle": translate_toggle,
    }

    total_nodes = len(graph.nodes)

    for i, output in enumerate(graph.stream(inputs)):
        progress = (i + 1) / total_nodes
        progress_bar.progress(progress)

        # Calculate progress percentage
        progress_percentage = int(progress * 100)

        for key, value in output.items():
            # Display message for the next step
            status_container.text(f"{progress_percentage}% - {message_dict[key]}")
            state.update(value)

    # Set progress bar to 100% and display "Finished!" message in green
    progress_bar.progress(1.0)
    status_container.markdown(":green[100% - Finished!]")

    return state


if start_btn:
    file_paths = st.session_state["filepath"]
    if file_paths is None:
        st.error("Please upload a file first.")
    else:
        # Create graph
        state = process_graph(file_paths)
        # Download file
        download_files(file_paths[0], state, translate_toggle)
        st.session_state["filepath"] = None

        # Create document retriever
        retriever = create_ensemble_retriever(state["documents"])
        # Save document retriever
        st.session_state["retriever"] = retriever
        # Create chain
        chain = create_chain()
        # Save chain
        st.session_state["chain"] = chain
        # Clean cache files
        # clean_cache_files()


# Print previous conversation history   
print_messages()

# User input
user_input = st.chat_input("Ask your question!")

# Create empty container for warning message
warning_msg = st.empty()

if user_input:
    retriever = st.session_state["retriever"]
    # Create chain
    chain = st.session_state["chain"]

    if chain is not None:
        # User input
        st.chat_message("user").write(user_input)
        # Call document retriever
        contexts = retriever.invoke(user_input)
        # Call streaming
        response = chain.stream({"context": contexts, "question": user_input})

        with st.chat_message("assistant"):
            # Create empty container to stream tokens
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # Save conversation history
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("Please upload a file first.")
