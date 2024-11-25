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
        # Add page number
        page_with_number = f"# Page {page_num + 1}\n\n{page_content}"

        # Change the first '#' symbol to '##' (if already '##', change to '###')
        page_with_number = re.sub(r"^# ", "## ", page_with_number, flags=re.MULTILINE)
        page_with_number = re.sub(r"^## ", "### ", page_with_number, flags=re.MULTILINE)

        pages.append(page_with_number)

    return "\n".join(pages)


def add_images(output_folder, content):
    pages = []
    for image_num, page_content in content.items():
        image_path = f"{image_num}.png"
        if os.path.exists(os.path.join(output_folder, image_path)):
            pages_with_image = f"\n\n![{image_num}]({image_path})\n\n{page_content}"
        else:
            pages_with_image = f"\n\n![{image_num}](no_image.png)\n\n{page_content}"  # 이미지가 없을 경우 기본 이미지 설정

        # # 첫 번째 '#' 기호를 '##'로 변경 (이미 '##'인 경우 '###'로 변경)
        # pages_with_image = re.sub(r"^# ", "## ", pages_with_image, flags=re.MULTILINE)
        # pages_with_image = re.sub(r"^## ", "### ", pages_with_image, flags=re.MULTILINE)

        pages.append(pages_with_image)

    return "\n".join(pages)


def create_md(file_path, state: GraphState, type="translate"):
    """
    Create a markdown file

    Args:
        file_path (str): The path to the file to be processed
        state (GraphState): The state of the graph
        type (str): The type of the file to be created
    """
    output_folder = os.path.splitext(file_path)[0]
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Create folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Create file
    md_output_file = os.path.join(output_folder, f"{filename}_{type}.md")

    # Get data safely from state
    translated_texts = state.get("translated_texts")
    texts_summary = state.get("texts_summary")
    images_summary = state.get("images_summary")
    tables_summary = state.get("tables_summary")

    # Add page numbers
    if type == "translate" and translated_texts is not None:
        conbined_texts = add_page_numbers(translated_texts)
    elif type == "text_summary" and texts_summary is not None:
        conbined_texts = add_page_numbers(texts_summary)
    elif type == "image_summary" and images_summary is not None:
        conbined_texts = add_images(output_folder, images_summary)
    elif type == "table_summary" and tables_summary is not None:
        conbined_texts = add_images(output_folder, tables_summary)
    else:
        st.error(f"Invalid type or missing data for type: {type}")
        return

    # Save to file
    with open(md_output_file, "w", encoding="utf-8") as f:
        f.write(conbined_texts)


def create_and_download_zip(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        st.error(f"The folder does not exist: {folder_path}")
        return None  # None 반환

    # Create a zip file name (include current time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"markdown_files_{timestamp}.zip"
    zip_filepath = os.path.join(folder_path, zip_filename)  # Full path to the zip file

    # Compress the folder contents
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Exclude the current zip file
                if file == zip_filename:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    return zip_filepath  # Return the full path


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
