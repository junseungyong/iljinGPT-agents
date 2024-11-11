import os
import streamlit as st

from output import create_md, create_and_download_zip


def download_files(file_path, state, translate_toggle):
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


def check_file_type(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        return "pdf"
    elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return "image"
    else:
        return None
