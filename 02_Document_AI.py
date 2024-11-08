import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
from langchain_teddynote import logging
from layout_parser import graph_document_ai, GraphState
import pprint

import os
from constants import PROGRESS_MESSAGE_GRAPH_NODES
from output import create_md, create_and_download_zip, clean_cache_files


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Document AI")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 임베딩 파일 저장 폴더
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("Document AI 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["document_messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

if "filepath" not in st.session_state:
    st.session_state["filepath"] = None


# 사이드바 생성
with st.sidebar:
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
    )
    # 모델 선택 메뉴
    translate_lang = st.selectbox("Translate", ["Korean", "English", "German"], index=0)
    # Translate 토글 추가
    translate_toggle = st.checkbox("Enable Translation", value=False)
    # AI Translate & Summary 버튼 생성
    start_btn = st.button("Document AI")


def download_files(file_path, state):
    zip_filepath = os.path.splitext(file_path)[0]  # 폴더 경로d

    if translate_toggle:
        create_md(file_path, state, "translate")

    create_md(file_path, state, "text_summary")
    create_md(file_path, state, "image_summary")
    create_md(file_path, state, "table_summary")

    zip_filename = create_and_download_zip(zip_filepath)  # 전체 경로를 받음

    # 압축 파일 다운로드
    if zip_filename and os.path.exists(
        zip_filename
    ):  # zip_filename이 None이 아닐 때 확인
        with open(zip_filename, "rb") as f:
            st.download_button(
                label="Download Results",
                data=f,
                file_name=os.path.basename(zip_filename),  # 파일 이름만 사용
                mime="application/zip",
            )

        # 임시 압축 파일 삭제
        os.remove(zip_filename)  # zip_filename을 삭제
    else:
        st.error("Problem in creating zip file")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["document_messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시리를 추가
def add_message(role, message):
    st.session_state["document_messages"].append(
        ChatMessage(role=role, content=message)
    )


def create_chain(retriever):
    # 단계 6: 프롬프트 생성(Create Prompt)
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatOllama(model="gemma2-27B:latest", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
def cache_file(files):
    file_extension = os.path.splitext(files[0].name)[
        1
    ].lower()  # 첫 번째 파일의 확장자 확인
    # 업로드한 파일을 캐시 디랙토리에 저장합니다.
    file_paths = []
    if files:  # 파일이 존재하는 경우
        if file_extension == ".pdf":
            if len(files) > 1:  # 파일이 1개 이상인 경우
                st.warning("PDF file only support one file")
                st.session_state["filepath"] = None
            else:
                file_content = files[0].read()  # 첫 번째 파일 읽기
                file_path = f"./.cache/files/{files[0].name}"
                with open(file_path, "wb") as f:
                    f.write(file_content)
                # 추가적인 PDF 파일 처리 코드 작성 가능
                file_paths.append(file_path)
                st.session_state["filepath"] = file_paths
        # elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        #     for file in files:
        #         file_content = file.read()  # 파일 읽기
        #         file_path = f"./.cache/files/{file.name}"
        #         with open(file_path, "wb") as f:
        #             f.write(file_content)
        #         file_paths.append(file_path)
        #     st.session_state["filepath"] = file_paths
        else:
            st.warning("Not supported file type")
            st.session_state["filepath"] = None


if uploaded_files:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정)
    cache_file(uploaded_files)
    uploaded_files = None


def process_graph(file_paths):
    print("process_graph")
    # 진행도 표시를 위한 컨테이너 생성
    progress_bar = st.progress(0)
    status_container = st.empty()

    state = GraphState()

    # 그래프 생성
    message_dict = PROGRESS_MESSAGE_GRAPH_NODES
    graph = graph_document_ai(translate_toggle)
    inputs = {
        "filepath": file_paths[0],
        "batch_size": 10,
        "translate_lang": translate_lang,
        "translate_toggle": translate_toggle,
    }

    total_nodes = len(graph.nodes)

    for i, output in enumerate(graph.stream(inputs)):
        print(f"output: {output}")
        progress = (i + 1) / total_nodes
        progress_bar.progress(progress)

        # 진행도를 백분율로 계산
        progress_percentage = int(progress * 100)

        for key, value in output.items():
            # 다음 단계의 메시지를 표시
            pprint.pprint(f"output from node '{key}':")
            status_container.text(f"{progress_percentage}% - {message_dict[key]}")
            # pprint.pprint("---")
            # pprint.pprint(value, indent=2, width=80, depth=None)
            state.update(value)

    # Progress bar를 100%로 설정하고 "Finished!" 메시지를 녹색으로 표시
    progress_bar.progress(1.0)
    status_container.markdown(":green[100% - Finished!]")

    return state


if start_btn:
    file_paths = st.session_state["filepath"]
    print(f"file_paths: {file_paths}")
    if file_paths is None:
        st.error("Please upload a file first.")
    else:
        state = process_graph(file_paths)
        download_files(file_paths[0], state)
        st.session_state["filepath"] = None
        # retriever = create_retriever_save_local(
        #     file_paths, state, "RBA_index", translate_toggle
        # )
        # if retriever:
        #     chain = create_chain(retriever)
        #     st.session_state["chain"] = chain
        clean_cache_files()


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("Ask your question!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if user_input:
    # chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)를 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("Please upload a file first.")
