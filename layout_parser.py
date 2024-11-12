from dotenv import load_dotenv
from langchain_teddynote import logging
from typing import TypedDict
import os
import pymupdf
import fitz
import json
import requests
from PIL import Image
import PIL
from langgraph.graph import StateGraph, END, START
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.documents import Document

from langchain_teddynote.models import MultiModal
from langchain_core.runnables import chain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    filepath: str  # path
    filetype: str  # pdf
    translate_lang: str  # translate lang
    translate_toggle: bool  # translate toggle
    page_numbers: list[int]  # page numbers
    batch_size: int  # batch size
    split_filepaths: list[str]  # split files
    index_contents: list[str]  # index contents
    analyzed_files: list[str]  # analyzed files
    page_elements: dict[int, dict[str, list[dict]]]  # page elements
    page_metadata: dict[int, dict]  # page metadata
    doc_metadata: dict
    page_summary: dict[int, str]  # page summary
    images: list[str]  # image paths
    images_summary: list[str]  # image summary
    tables: list[str]  # table
    tables_summary: dict[int, str]  # table summary
    texts: list[str]  # text
    documents: list[Document]  # documents
    translated_texts: list[str]  # translated text
    texts_summary: list[str]  # text summary
    image_summary_data_batches: list[dict]  # 이미지 요약 데이터 배치
    table_summary_data_batches: list[dict]  # 표 요약 데이터 배치


class ExtractMetadata(BaseModel):
    version: str = Field(description="revision number or version")
    date: str = Field(description="date of the document")
    title: str = Field(description="title of the document")


class DocumentParser:
    def __init__(self, api_key):
        """
        DocumentParser 클래스의 생성자

        :param api_key: Upstage API 인증을 위한 API 키
        """
        self.api_key = api_key

    def _upstage_document_parse(self, input_file):
        """
        Upstage API를 사용하여 문서 분석

        :param input_file: 분석할 문서 파일 경로
        :return: 분석 결과
        """
        # Upstage API 엔드포인트 URL
        url = "https://api.upstage.ai/v1/document-ai/document-parse"
        # 헤더 설정
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # 파일 업로드를 위한 파일 객체 생성
        files = {"document": open(input_file, "rb")}

        data = {"output_formats": "['html', 'markdown', 'text']"}

        # API 요청 보내기
        response = requests.post(url, headers=headers, files=files, data=data)

        # 요청이 성공하면 결과를 파일에 저장
        if response.status_code == 200:
            output_file = os.path.splitext(input_file)[0] + ".json"
            # 결과를 파일에 저장
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)

            return output_file
        else:
            # 요청이 실패하면 오류 메시지 반환
            raise ValueError(f"API 요청 실패: {response.status_code}")

    def execute(self, input_file):
        """
        문서 분석 실행

        :param input_file: 분석할 문서 파일 경로
        :return: 분석 결과 파일 경로
        """
        return self._upstage_document_parse(input_file)


class ImageCropper:
    @staticmethod
    def load_image_without_rotation(file_path):
        """
        이미지를 열고 회전을 제거하는 메서드

        :param file_path: 이미지 파일 경로
        :return: 회전이 제거된 이미지 객체
        """
        # EXIF 태그를 무시하도록 설정
        PIL.Image.LOAD_TRUNCATED_IMAGES = True

        # 이미지 열기
        img = Image.open(file_path)

        # EXIF 데이터 가져오기
        exif = img._getexif()

        if exif:
            # EXIF에서 방향 정보 찾기
            orientation_key = 274  # 'Orientation' 태그의 키
            if orientation_key in exif:
                orientation = exif[orientation_key]

                # 방향에 따라 이미지 회전
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)

        return img

    @staticmethod
    def pdf_to_image(pdf_file, page_num, dpi=300):
        """
        PDF 파일의 특정 페이지를 이미지로 변환하는 메서드

        :param page_num: 변환할 페이지 번호 (1부터 시작)
        :param dpi: 이미지 해상도 (기본값: 300)
        :return: 변환된 이미지 객체
        """
        with pymupdf.open(pdf_file) as doc:
            page = doc[page_num].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates):
        """
        좌표를 정규화하는 정적 메서드

        :param coordinates: 원본 좌표 리스트
        :param output_page_size: 출력 페이지 크기 [너비, 높이]
        :return: 정규화된 좌표 (x1, y1, x2, y2)
        """
        x_values = [coord["x"] for coord in coordinates]
        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

        return (
            x1,
            y1,
            x2,
            y2,
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        """
        이미지를 주어진 좌표에 따라 자르고 저장하는 정적 메서드

        :param img: 원본 이미지 객체
        :param coordinates: 정규화된 좌표 (x1, y1, x2, y2)
        :param output_file: 저장할 파일 경로
        """
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)

    # def crop_image(img, bounding_box, output_file):
    #     """
    #     이미지를 주어진 bounding box에 따라 자르고 저장하는 정적 메서드

    #     :param img: 원본 이미지 객체
    #     :param bounding_box: 크롭할 영역의 좌표 리스트 [{"x": x, "y": y}, ...]
    #     :param output_file: 저장할 파일 경로
    #     """
    #     x_values = [coord["x"] for coord in bounding_box]
    #     y_values = [coord["y"] for coord in bounding_box]
    #     x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

    #     cropped_img = img.crop((x1, y1, x2, y2))
    #     cropped_img.save(output_file)


def extract_start_end_page(filename):
    """
    파일 이름에서 시작 페이지와 끝 페이지 번호를 추출하는 함수입니다.

    :param filename: 분석할 파일의 이름
    :return: 시작 페이지 번호와 끝 페이지 번호를 튜플로 반환
    """
    # 파일 경로에서 파일 이름만 추출
    file_name = os.path.basename(filename)
    # 파일 이름을 '_' 기준으로 분리
    file_name_parts = file_name.split("_")

    if len(file_name_parts) >= 3:
        # 파일 이름의 뒤에서 두 번째 부분에서 숫자를 추출하여 시작 페이지로 설정
        start_page = int(re.findall(r"(\d+)", file_name_parts[-2])[0])
        # 파일 이름의 마지막 부분에서 숫자를 추출하여 끝 페이지로 설정
        end_page = int(re.findall(r"(\d+)", file_name_parts[-1])[0])
    else:
        # 파일 이름 형식이 예상과 다를 경우 기본값 설정
        start_page, end_page = 0, 0

    return start_page, end_page


def create_extract_metadata_chain():

    output_parser = PydanticOutputParser(pydantic_object=ExtractMetadata)

    prompt = PromptTemplate.from_template(
        """Please extract the metadata from the following context.

CONTEXT:
{context}

FORMAT:
{format}
"""
    )

    prompt = prompt.partial(format=output_parser.get_format_instructions())
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
    )
    metadata_extract_chain = prompt | llm | output_parser

    return metadata_extract_chain


def create_text_summary_chain():
    prompt = PromptTemplate.from_template(
        """Please summarize the sentence according to the following REQUEST.
    
        REQUEST:
        1. Summarize the main points in bullet points.
        2. Write the summary in {output_language}.
        3. DO NOT translate any technical terms.
        4. DO NOT include any unnecessary information.
        5. Summary must include important entities, numerical values.
        
        CONTEXT:
        {context}

        SUMMARY:"
        """
    )

    # ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
    # llm = ChatOpenAI(
    #     model_name="gpt-4o-mini",
    #     temperature=0,
    # )
    llm = ChatOllama(model="gemma2-27B:latest", temperature=0)

    # 문서 요약을 위한 체인을 생성합니다.
    # 이 체인은 여러 문서를 입력받아 하나의 요약된 텍스트로 결합합니다.
    text_summary_chain = create_stuff_documents_chain(llm, prompt)

    return text_summary_chain


def create_text_translate_chain():
    prompt = PromptTemplate.from_template(
        """You are a translator with vast knowledge of human languages. Please translate the following context to {output_language}.
        if the context language is same as {output_language}, just return the context as is.

        CONTEXT:
        {context}

        TRANSLATED_TEXT:"
        """
    )

    # ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
    # llm = ChatOpenAI(
    #     model_name="gpt-4o-mini",
    #     temperature=0,
    # )
    llm = ChatOllama(model="gemma2-27B:latest", temperature=0)

    # 문서 번역을 위한 체인을 생성합니다.
    text_tranlate_chain = create_stuff_documents_chain(llm, prompt)

    return text_tranlate_chain


@chain
def extract_image_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from IMAGE.
    With a given image, your task is to extract key entities, summarize them, and write useful information.
    Please write the summary in {language}."""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["image"]
        language = data_batch["lang"]
        user_prompt_template = f"""Here is the context related to the image: {context}

LANGUAGE: {language}
###

Output Format:

TITLE:
SUMMARY:
ENTITIES:
"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


@chain
def extract_table_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from TABLE. With a given image, your task is to extract key entities, summarize them, and write useful information.
    Please write the summary in {language}."""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["table"]
        language = data_batch["lang"]
        user_prompt_template = f"""Here is the context related to the image of table: {context}

LANGUAGE: {language}        
###

Output Format:

TITLE:
SUMMARY:
ENTITIES:
DATA_INSIGHTS:
"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


def route_document(state: GraphState):
    filetype = state["filetype"]
    if filetype == "pdf":
        return "split_pdf"
    else:
        return "merge_image"


def split_pdf(state: GraphState):
    """
    입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

    :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
    :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
    """
    # PDF 파일 경로와 배치 크기 추출
    filepath = state["filepath"]
    batch_size = state["batch_size"]

    # PDF 파일 열기
    input_pdf = fitz.open(filepath)
    num_pages = input_pdf.page_count
    print(f"총 페이지 수: {num_pages}")

    page_metadata = dict()
    for page in range(num_pages):
        rect = input_pdf[page].rect
        metadata = {
            "size": [
                int(rect.width),
                int(rect.height),
            ],
        }
        page_metadata[page] = metadata

    ret = []
    # PDF 분할 작업 시작
    for start_page in range(0, num_pages, batch_size):
        # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
        end_page = min(start_page + batch_size, num_pages) - 1

        # 분할된 PDF 파일명 생성
        input_file_basename = os.path.splitext(filepath)[0]
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        print(f"분할 PDF 생성: {output_file}")

        # 새로운 PDF 파일 생성 및 페이지 삽입
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # 원본 PDF 파일 닫기
    input_pdf.close()

    # 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체 반환
    return GraphState(split_filepaths=ret, page_metadata=page_metadata)


def merge_image(state: GraphState):
    filepaths = state["filepath"]

    page_metadata = dict()
    for page, filepath in enumerate(filepaths):
        print(f"filepath: {filepath}")
        img = Image.open(filepath)
        width, height = img.size
        metadata = {
            "size": [
                int(width),
                int(height),
            ],
        }
        page_metadata[page] = metadata

    return GraphState(split_filepaths=filepaths, page_metadata=page_metadata)


def analyze_layout(state: GraphState):
    # 분할된 PDF 파일 목록을 가져옵니다.
    split_files = state["split_filepaths"]

    # DocumentParser 객체를 생성합니다. API 키는 환경 변수에서 가져옵니다.
    analyzer = DocumentParser(os.environ.get("UPSTAGE_API_KEY"))

    # 분석된 파일들의 경로를 저장할 리스트를 초기화합니다.
    analyzed_files = []

    # 각 분할된 PDF 파일에 대해 레이아웃 분석을 수행합니다.
    for file in split_files:
        # 레이아웃 분석을 실행하고 결과 파일 경로를 리스트에 추가합니다.
        analyzed_files.append(analyzer.execute(file))

    # 분석된 파일 경로들을 정렬하여 새로운 GraphState 객체를 생성하고 반환합니다.
    # 정렬은 파일들의 순서를 유지하기 위해 수행됩니다.
    return GraphState(analyzed_files=sorted(analyzed_files))


def add_analyzed_layout(state: GraphState):
    split_files = state["split_filepaths"]

    analyzed_files = []

    for file in split_files:
        output_file = os.path.splitext(file)[0] + ".json"
        analyzed_files.append(output_file)

    return GraphState(analyzed_files=sorted(analyzed_files))


def extract_page_elements(state: GraphState):
    # 분석된 JSON 파일 목록을 가져옵니다.
    json_files = state["analyzed_files"]
    file_type = state["filetype"]
    # 페이지별 요소를 저장할 딕셔너리를 초기화합니다.
    page_elements = dict()

    # 전체 문서에서 고유한 요소 ID를 부여하기 위한 카운터입니다.
    element_id = 0

    # 각 JSON 파일을 순회하며 처리합니다.
    for i, json_file in enumerate(json_files):
        if file_type == "image":
            pass
        else:
            # 파일명에서 시작 페이지 번호를 추출합니다.
            start_page, _ = extract_start_end_page(json_file)

        # JSON 파일을 열어 데이터를 로드합니다.
        with open(json_file, "r") as f:
            data = json.load(f)

        # JSON 데이터의 각 요소를 처리합니다.
        for element in data["elements"]:
            # 원본 페이지 번호를 정수로 변환합니다.
            if file_type == "image":
                relative_page = i
            else:
                original_page = int(element["page"])
                # 전체 문서 기준의 상대적 페이지 번호를 계산합니다.
                relative_page = start_page + original_page - 1
            # 해당 페이지의 요소 리스트가 없으면 새로 생성합니다.
            if relative_page not in page_elements:
                page_elements[relative_page] = []

            # 요소에 고유 ID를 부여합니다.
            element["id"] = element_id
            element_id += 1

            # 요소의 페이지 번호를 상대적 페이지 번호로 업데이트합니다.
            element["page"] = relative_page
            # 요소를 해당 페이지의 리스트에 추가합니다.
            page_elements[relative_page].append(element)

    # 추출된 페이지별 요소 정보로 새로운 GraphState 객체를 생성하여 반환합니다.
    return GraphState(page_elements=page_elements)


def extract_tag_elements_per_page(state: GraphState):
    # GraphState 객체에서 페이지 요소들을 가져옵니다.
    page_elements = state["page_elements"]

    # 파싱된 페이지 요소들을 저장할 새로운 딕셔너리를 생성합니다.
    parsed_page_elements = dict()

    # 각 페이지와 해당 페이지의 요소들을 순회합니다.
    for key, page_element in page_elements.items():
        # 이미지, 테이블, 텍스트 요소들을 저장할 리스트를 초기화합니다.
        image_elements = []
        table_elements = []
        text_elements = []
        chart_elements = []
        equation_elements = []
        index_elements = []
        # 페이지의 각 요소를 순회하며 카테고리별로 분류합니다.
        for element in page_element:
            if element["category"] == "figure":
                # 이미지 요소인 경우 image_elements 리스트에 추가합니다.
                image_elements.append(element)
            elif element["category"] == "table":
                # 테이블 요소인 경우 table_elements 리스트에 추가합니다.
                table_elements.append(element)
            elif element["category"] == "chart":
                chart_elements.append(element)
            elif element["category"] == "equation":
                equation_elements.append(element)
            elif element["category"] == "index":
                index_elements.append(element)
            else:
                # 그 외의 요소는 모두 텍스트 요소로 간주하여 text_elements 리스트에 추가합니다.
                text_elements.append(element)

        # 분류된 요소들을 페이지 키와 함께 새로운 딕셔너리에 저장합니다.
        parsed_page_elements[key] = {
            "image_elements": image_elements,
            "table_elements": table_elements,
            "text_elements": text_elements,
            "chart_elements": chart_elements,
            "equation_elements": equation_elements,
            "index_elements": index_elements,
            "elements": page_element,  # 원본 페이지 요소도 함께 저장합니다.
        }

    # 파싱된 페이지 요소들을 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(page_elements=parsed_page_elements)


def extract_page_numbers(state: GraphState):
    return GraphState(page_numbers=list(state["page_elements"].keys()))


def crop_image(state: GraphState):
    """
    PDF 파일에서 이미지를 추출하고 크롭하는 함수

    :param state: GraphState 객체
    :return: 크롭된 이미지 정보가 포함된 GraphState 객체
    """
    files = state["filepath"]  # 파일 경로
    file_type = state["filetype"]
    page_numbers = state["page_numbers"]  # 처리할 페이지 번호 목록
    if file_type == "image":
        output_folder = os.path.splitext(files[0])[0]  # 출력 폴더 경로 설정
    else:
        output_folder = os.path.splitext(files)[0]  # 출력 폴더 경로 설정
    os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

    cropped_images = dict()  # 크롭된 이미지 정보를 저장할 딕셔너리
    for page_num in page_numbers:
        if file_type == "pdf":
            image_file = ImageCropper.pdf_to_image(
                files, page_num
            )  # PDF 페이지를 이미지로 변환
        elif file_type == "image":
            image_file = ImageCropper.load_image_without_rotation(files[page_num])

        for element in state["page_elements"][page_num]["image_elements"]:
            if element["category"] == "figure":
                # 이미지 요소의 좌표를 정규화
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["coordinates"]
                )

                # 크롭된 이미지 저장 경로 설정
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # 이미지 크롭 및 저장
                ImageCropper.crop_image(image_file, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
    return GraphState(
        images=cropped_images
    )  # 크롭된 이미지 정보를 포함한 GraphState 반환


def crop_table(state: GraphState):
    """
    PDF 파일에서 표를 추출하고 크롭하는 함수

    :param state: GraphState 객체
    :return: 크롭된 표 이미지 정보가 포함된 GraphState 객체
    """
    files = state["filepath"]  # PDF 파일 경로
    file_type = state["filetype"]
    page_numbers = state["page_numbers"]  # 처리할 페이지 번호 목록
    if file_type == "image":
        output_folder = os.path.splitext(files[0])[0]  # 출력 폴더 경로 설정
    else:
        output_folder = os.path.splitext(files)[0]  # 출력 폴더 경로 설정
    os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

    cropped_images = dict()  # 크롭된 표 이미지 정보를 저장할 딕셔너리
    for page_num in page_numbers:
        if file_type == "pdf":
            image_file = ImageCropper.pdf_to_image(
                files, page_num
            )  # PDF 페이지를 이미지로 변환
        elif file_type == "image":
            image_file = ImageCropper.load_image_without_rotation(files[page_num])
        for element in state["page_elements"][page_num]["table_elements"]:
            if element["category"] == "table":
                # 표 요소의 좌표를 정규화
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["coordinates"],
                )

                # 크롭된 표 이미지 저장 경로 설정
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # 표 이미지 크롭 및 저장
                ImageCropper.crop_image(image_file, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
    return GraphState(
        tables=cropped_images
    )  # 크롭된 표 이미지 정보를 포함한 GraphState 반환


def extract_page_text(state: GraphState):
    files = state["filepath"]
    page_numbers = state["page_numbers"]
    extracted_texts = dict()

    for page_num in page_numbers:
        extracted_texts[page_num] = ""
        page_element = state["page_elements"][page_num]

        print(f"Page {page_num} structure: {page_element.keys()}")

        if "text_elements" in page_element:
            for text_element in page_element["text_elements"]:
                if (
                    isinstance(text_element, dict)
                    and "content" in text_element
                    and "markdown" in text_element["content"]
                ):
                    extracted_texts[page_num] += text_element["content"]["markdown"]
                else:
                    print(
                        f"Unexpected text_element structure on page {page_num}: {text_element}"
                    )

        if "table_elements" in page_element:
            for table_element in page_element["table_elements"]:
                if (
                    isinstance(table_element, dict)
                    and "content" in table_element
                    and "markdown" in table_element["content"]
                ):
                    extracted_texts[page_num] += table_element["content"]["markdown"]
                else:
                    print(
                        f"Unexpected table_element structure on page {page_num}: {table_element}"
                    )

        documents = []
        source = os.path.splitext(files)[0]
        for key, value in extracted_texts.items():
            page_number = key
            text = value
            metadata = {"page": page_number, "source": source}
            documents.append(Document(page_content=text, metadata=metadata))
    return GraphState(texts=extracted_texts, documents=documents)


def extract_doc_metadata(state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    texts = state["texts"]

    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    inputs = [{"context": Document(page_content=sorted_texts[0][1])}]

    doc_metadata_chain = create_extract_metadata_chain()
    metadata = doc_metadata_chain.invoke(inputs)

    return GraphState(doc_metadata=metadata)


def translate_text(state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    page_numbers = state["page_numbers"]
    texts = state["texts"]
    translate_lang = state["translate_lang"]

    # 번역된 텍스트를 저장할 딕셔너리를 초기화합니다.
    translated_texts = dict()

    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
    inputs = [
        {"context": [Document(page_content=text)], "output_language": translate_lang}
        for page_num, text in sorted_texts
    ]

    # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    text_tranlate_chain = create_text_translate_chain()
    translation_results = text_tranlate_chain.batch(inputs)

    # translation_results를 순서대로 페이지 번호와 매핑
    for page_num, translation in zip(page_numbers, translation_results):
        translated_texts[page_num] = translation

    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(translated_texts=translated_texts)


def create_text_summary(state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    page_numbers = state["page_numbers"]
    translate_toggle = state["translate_toggle"]
    translate_lang = state["translate_lang"]
    if translate_toggle:
        texts = state["translated_texts"]
    else:
        texts = state["texts"]

    # 요약된 텍스트를 저장할 딕셔너리를 초기화합니다.
    text_summary = dict()

    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
    inputs = [
        {"context": [Document(page_content=text)], "output_language": translate_lang}
        for page_num, text in sorted_texts
    ]

    # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    text_summary_chain = create_text_summary_chain()
    summaries = text_summary_chain.batch(inputs)

    # translation_results를 순서대로 페이지 번호와 매핑
    for page_num, translation in zip(page_numbers, summaries):
        text_summary[page_num] = translation

    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(texts_summary=text_summary)


def create_image_summary_data_batches(state: GraphState):
    # 이미지 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["texts_summary"][page_num]
        # 해당 페이지의 모든 이미지 요소에 대해 반복
        for image_element in state["page_elements"][page_num]["image_elements"]:
            # 이미지 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 이미지 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "image": state["images"][image_id],  # 이미지 파일 경로
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 이미지 ID
                    "lang": state["translate_lang"],  # 언어
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(image_summary_data_batches=data_batches)


def create_table_summary_data_batches(state: GraphState):
    # 테이블 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["texts_summary"][page_num]
        # 해당 페이지의 모든 테이블 요소에 대해 반복
        for image_element in state["page_elements"][page_num]["table_elements"]:
            # 테이블 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 테이블 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "table": state["tables"][image_id],  # 테이블 데이터
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 테이블 ID
                    "lang": state["translate_lang"],  # 언어
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(table_summary_data_batches=data_batches)


def create_image_summary(state: GraphState):
    # 이미지 요약 추출
    # extract_image_summary 함수를 호출하여 이미지 요약 생성
    image_summaries = extract_image_summary.invoke(state["image_summary_data_batches"])

    # 이미지 요약 결과를 저장할 딕셔너리 초기화
    image_summary_output = dict()

    # 각 데이터 배치와 이미지 요약을 순회하며 처리
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 이미지 요약 저장
        image_summary_output[data_batch["id"]] = image_summary

    # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(images_summary=image_summary_output)


def create_table_summary(state: GraphState):
    # 테이블 요약 추출
    table_summaries = extract_table_summary.invoke(state["table_summary_data_batches"])

    # 테이블 요약 결과를 저장할 딕셔너리 초기화
    table_summary_output = dict()

    # 각 데이터 배치와 테이블 요약을 순회하며 처리
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 테이블 요약 저장
        table_summary_output[data_batch["id"]] = table_summary

    # 테이블 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(tables_summary=table_summary_output)


def clean_up(state: GraphState):
    for file in state["split_filepaths"] + state["analyzed_files"]:
        os.remove(file)


def graph_document_ai(translate_toggle: bool):
    workflow = StateGraph(GraphState)

    workflow.add_node("split_pdf", split_pdf)
    workflow.add_node("merge_image", merge_image)

    workflow.add_node("analyze_layout", analyze_layout)

    workflow.add_node("extract_page_elements", extract_page_elements)
    workflow.add_node("extract_tag_elements_per_page", extract_tag_elements_per_page)
    workflow.add_node("extract_page_numbers", extract_page_numbers)

    workflow.add_node("crop_image", crop_image)
    workflow.add_node("crop_table", crop_table)
    workflow.add_node("extract_page_text", extract_page_text)

    if translate_toggle:
        workflow.add_node("translate_text", translate_text)

    workflow.add_node("create_text_summary", create_text_summary)
    workflow.add_node(
        "create_image_summary_data_batches", create_image_summary_data_batches
    )
    workflow.add_node(
        "create_table_summary_data_batches", create_table_summary_data_batches
    )
    workflow.add_node("create_image_summary", create_image_summary)
    workflow.add_node("create_table_summary", create_table_summary)
    # workflow.add_node("clean_up", clean_up)

    workflow.add_conditional_edges(
        START,
        route_document,
        {
            "split_pdf": "split_pdf",
            "merge_image": "merge_image",
        },
    )
    workflow.add_edge("split_pdf", "analyze_layout")
    workflow.add_edge("merge_image", "analyze_layout")
    workflow.add_edge("analyze_layout", "extract_page_elements")
    workflow.add_edge("extract_page_elements", "extract_tag_elements_per_page")
    workflow.add_edge("extract_tag_elements_per_page", "extract_page_numbers")
    workflow.add_edge("extract_page_numbers", "crop_image")
    workflow.add_edge("crop_image", "crop_table")
    workflow.add_edge("crop_table", "extract_page_text")

    if translate_toggle:
        workflow.add_edge("extract_page_text", "translate_text")
        workflow.add_edge("translate_text", "create_text_summary")
    else:
        workflow.add_edge("extract_page_text", "create_text_summary")

    workflow.add_edge("create_text_summary", "create_image_summary_data_batches")
    workflow.add_edge(
        "create_image_summary_data_batches", "create_table_summary_data_batches"
    )
    workflow.add_edge("create_table_summary_data_batches", "create_image_summary")
    workflow.add_edge("create_image_summary", "create_table_summary")
    workflow.add_edge("create_table_summary", END)
    # workflow.add_edge("clean_up", END)

    graph = workflow.compile()

    return graph
