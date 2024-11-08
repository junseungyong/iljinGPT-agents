import os
import shutil
from constants import OUTPUT_DIR
import re
from layout_parser import GraphState
import zipfile
import streamlit as st
from datetime import datetime


def add_page_numbers(content):
    pages = []
    for page_num, page_content in content.items():
        # 페이지 번호 추가
        page_with_number = f"# Page {page_num + 1}\n\n{page_content}"

        # 첫 번째 '#' 기호를 '##'로 변경 (이미 '##'인 경우 '###'로 변경)
        page_with_number = re.sub(r"^# ", "## ", page_with_number, flags=re.MULTILINE)
        page_with_number = re.sub(r"^## ", "### ", page_with_number, flags=re.MULTILINE)

        pages.append(page_with_number)

    return "\n".join(pages)


def add_images(output_folder, content):
    pages = []
    for image_num, page_content in content.items():
        image_path = f"{image_num}.png"
        if os.path.exists(os.path.join(output_folder, image_path)):
            pages_with_image = f"![{image_num}]({image_path})\n\n{page_content}"
        else:
            pages_with_image = f"![{image_num}](no_image.png)\n\n{page_content}"  # 이미지가 없을 경우 기본 이미지 설정

        # 첫 번째 '#' 기호를 '##'로 변경 (이미 '##'인 경우 '###'로 변경)
        pages_with_image = re.sub(r"^# ", "## ", pages_with_image, flags=re.MULTILINE)
        pages_with_image = re.sub(r"^## ", "### ", pages_with_image, flags=re.MULTILINE)

        pages.append(pages_with_image)

    return "\n".join(pages)


def create_md(file_path, state: GraphState, type="translate"):
    """
    markdown 파일 생성
    """
    output_folder = os.path.splitext(file_path)[0]
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 파일 생성
    md_output_file = os.path.join(output_folder, f"{filename}_{type}.md")

    # 페이지 번호 추가
    if type == "translate" and state["translated_texts"] is not None:
        conbined_texts = add_page_numbers(state["translated_texts"])
    elif type == "text_summary" and state["texts_summary"] is not None:
        conbined_texts = add_page_numbers(state["texts_summary"])
    elif type == "image_summary" and state["images_summary"] is not None:
        conbined_texts = add_images(output_folder, state["images_summary"])
    elif type == "table_summary" and state["tables_summary"] is not None:
        conbined_texts = add_images(output_folder, state["tables_summary"])
    else:
        st.error("Invalid type")
    # 파일로 저장
    with open(md_output_file, "w", encoding="utf-8") as f:
        f.write(conbined_texts)


def create_and_download_zip(folder_path):
    # 폴더 존재 여부 확인
    if not os.path.isdir(folder_path):
        st.error(f"The folder does not exist: {folder_path}")
        return None  # None 반환

    # 압축 파일 이름 생성 (현재 시간 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"markdown_files_{timestamp}.zip"
    zip_filepath = os.path.join(folder_path, zip_filename)  # 압축 파일의 전체 경로

    # 폴더 내용 압축
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                # 현재 압축 중인 파일은 제외
                if file == zip_filename:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    return zip_filepath  # 전체 경로 반환


def clean_cache_files():
    cache_dir = "./.cache/files"
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
