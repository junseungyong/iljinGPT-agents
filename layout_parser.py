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

import ollama


# Class to store GraphState
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
    image_summary_data_batches: list[dict]  # image summary data batches
    table_summary_data_batches: list[dict]  # table summary data batches


class ExtractMetadata(BaseModel):
    version: str = Field(description="revision number or version")
    date: str = Field(description="date of the document")
    title: str = Field(description="title of the document")


class DocumentParser:
    def __init__(self, api_key):
        """
        Constructor for DocumentParser class

        :param api_key: API key for Upstage API authentication
        """
        self.api_key = api_key

    def _upstage_document_parse(self, input_file):
        """
        Analyze document using Upstage API

        :param input_file: Path to the document file to analyze
        :return: Path to the analysis result
        """
        # Upstage API endpoint URL
        url = "https://api.upstage.ai/v1/document-ai/document-parse"
        # Set header
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Create file object for file upload
        files = {"document": open(input_file, "rb")}

        data = {"output_formats": "['html', 'markdown', 'text']"}

        # Send API request
        response = requests.post(url, headers=headers, files=files, data=data)

        # If the request is successful, save the result to a file
        if response.status_code == 200:
            output_file = os.path.splitext(input_file)[0] + ".json"
            # Save the result to a file
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)

            return output_file
        else:
            # If the request fails, return an error message
            raise ValueError(f"API request failed: {response.status_code}")

    def execute(self, input_file):
        """
        Execute document analysis

        :param input_file: Path to the document file to analyze
        :return: Path to the analysis result
        """
        return self._upstage_document_parse(input_file)


class ImageCropper:
    @staticmethod
    def load_image_without_rotation(file_path):
        """
        Method to load image and remove rotation

        :param file_path: Path to the image file
        :return: Image object with removed rotation
        """
        # Ignore EXIF tags
        PIL.Image.LOAD_TRUNCATED_IMAGES = True

        # Open image
        img = Image.open(file_path)

        # Get EXIF data
        exif = img._getexif()

        if exif:
            # Find orientation information from EXIF
            orientation_key = 274  # 'Orientation' tag key
            if orientation_key in exif:
                orientation = exif[orientation_key]

                # Rotate image according to orientation
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
        Method to convert a specific page of a PDF file to an image

        :param page_num: Page number to convert (starts from 1)
        :param dpi: Image resolution (default: 300)
        :return: Converted image object
        """
        with pymupdf.open(pdf_file) as doc:
            page = doc[page_num].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates):
        """
        Method to normalize coordinates

        :param coordinates: Original coordinates list
        :param output_page_size: Output page size [width, height]
        :return: Normalized coordinates (x1, y1, x2, y2)
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
        Method to crop image and save it according to given coordinates

        :param img: Original image object
        :param coordinates: Normalized coordinates (x1, y1, x2, y2)
        :param output_file: Path to save the file
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
    #     Method to crop image and save it according to given bounding box

    #     :param img: Original image object
    #     :param bounding_box: Coordinates list of the area to crop [{"x": x, "y": y}, ...]
    #     :param output_file: Path to save the file
    #     """
    #     x_values = [coord["x"] for coord in bounding_box]
    #     y_values = [coord["y"] for coord in bounding_box]
    #     x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

    #     cropped_img = img.crop((x1, y1, x2, y2))
    #     cropped_img.save(output_file)


def extract_start_end_page(filename):
    """
    Method to extract start and end page numbers from filename

    :param filename: Name of the file to analyze
    :return: Start and end page numbers as a tuple
    """
    # Extract file name from file path
    file_name = os.path.basename(filename)
    # Split file name by '_'
    file_name_parts = file_name.split("_")

    if len(file_name_parts) >= 3:
        # Extract number from the second last part of the file name as start page
        start_page = int(re.findall(r"(\d+)", file_name_parts[-2])[0])
        # Extract number from the last part of the file name as end page
        end_page = int(re.findall(r"(\d+)", file_name_parts[-1])[0])
    else:
        # If the file name format is different from expected, set default values
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

    llm = ChatOllama(model="gemma2-27B:latest", temperature=0)

    # Create a chain for document summarization
    # This chain takes multiple documents as input and combines them into one summarized text
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

    llm = ChatOllama(model="gemma2-27B:latest", temperature=0)

    # Create a chain for document translation
    text_tranlate_chain = create_stuff_documents_chain(llm, prompt)

    return text_tranlate_chain


@chain
def extract_image_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
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
DATA_INSIGHTS:
"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # Create a multimodal object
    multimodal_llm = MultiModal(llm)

    # Query from image file
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


@chain
def extract_table_summary(data_batches):
    # Create an object
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
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

    # Create a multimodal object
    multimodal_llm = MultiModal(llm)

    # Query from image file
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
    Split the input PDF into multiple smaller PDF files.

    :param state: GraphState object, containing the PDF file path and batch size information
    :return: GraphState object containing the list of split PDF file paths
    """
    # Extract the PDF file path and batch size
    filepath = state["filepath"]
    batch_size = state["batch_size"]

    # Open the PDF file
    input_pdf = fitz.open(filepath)
    num_pages = input_pdf.page_count
    print(f"Total page count: {num_pages}")

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
    # Start the PDF splitting process
    for start_page in range(0, num_pages, batch_size):
        # Calculate the last page of the batch (to avoid exceeding the total page count)
        end_page = min(start_page + batch_size, num_pages) - 1

        # Create the name of the split PDF file
        input_file_basename = os.path.splitext(filepath)[0]
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        print(f"Split PDF created: {output_file}")

        # Create a new PDF file and insert pages
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # Close the original PDF file
    input_pdf.close()

    # Return the GraphState object containing the list of split PDF file paths
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
    # Get the list of split PDF file paths
    split_files = state["split_filepaths"]

    # Create a DocumentParser object. The API key is retrieved from the environment variable.
    analyzer = DocumentParser(os.environ.get("UPSTAGE_API_KEY"))

    # Initialize a list to store the paths of analyzed files
    analyzed_files = []

    # Analyze the layout of each split PDF file
    for file in split_files:
        # Execute the layout analysis and add the result file path to the list
        analyzed_files.append(analyzer.execute(file))

    # Sort the analyzed file paths and create a new GraphState object to return them
    # Sorting is performed to maintain the order of the files
    return GraphState(analyzed_files=sorted(analyzed_files))


def add_analyzed_layout(state: GraphState):
    split_files = state["split_filepaths"]

    analyzed_files = []

    for file in split_files:
        output_file = os.path.splitext(file)[0] + ".json"
        analyzed_files.append(output_file)

    return GraphState(analyzed_files=sorted(analyzed_files))


def extract_page_elements(state: GraphState):
    # Get the list of analyzed JSON file paths
    json_files = state["analyzed_files"]
    file_type = state["filetype"]
    # Initialize a dictionary to store page-wise elements
    page_elements = dict()

    # Initialize a counter to assign unique element IDs
    element_id = 0

    # Iterate through each JSON file
    for i, json_file in enumerate(json_files):
        if file_type == "image":
            pass
        else:
            # Extract the start page number from the file name
            start_page, _ = extract_start_end_page(json_file)

        # Load the JSON file
        with open(json_file, "r") as f:
            data = json.load(f)

        # JSON 데이터의 각 요소를 처리합니다.
        for element in data["elements"]:
            # Convert the original page number to an integer
            if file_type == "image":
                relative_page = i
            else:
                original_page = int(element["page"])
                # Calculate the relative page number based on the entire document
                relative_page = start_page + original_page - 1
            # If the list of elements for the page does not exist, create a new one
            if relative_page not in page_elements:
                page_elements[relative_page] = []

            # Assign a unique ID to the element
            element["id"] = element_id
            element_id += 1

            # Update the element's page number to the relative page number
            element["page"] = relative_page
            # Add the element to the list of elements for the page
            page_elements[relative_page].append(element)

    # Create a new GraphState object with the extracted page-wise element information
    return GraphState(page_elements=page_elements)


def extract_tag_elements_per_page(state: GraphState):
    # Get the page-wise elements from the GraphState object
    page_elements = state["page_elements"]

    # Create a new dictionary to store the parsed page elements
    parsed_page_elements = dict()

    # Iterate through each page and its elements
    for key, page_element in page_elements.items():
        # Initialize lists to store image, table, text elements
        image_elements = []
        table_elements = []
        text_elements = []
        chart_elements = []
        equation_elements = []
        index_elements = []
        # Iterate through each element in the page
        for element in page_element:
            if element["category"] == "figure":
                # If the element is an image, add it to the image_elements list
                image_elements.append(element)
            elif element["category"] == "table":
                # If the element is a table, add it to the table_elements list
                table_elements.append(element)
            elif element["category"] == "chart":
                chart_elements.append(element)
            elif element["category"] == "equation":
                equation_elements.append(element)
            elif element["category"] == "index":
                index_elements.append(element)
            else:
                # If the element is not an image, table, chart, equation, or index, consider it a text element
                text_elements.append(element)

        # Store the categorized elements with the page key in a new dictionary
        parsed_page_elements[key] = {
            "image_elements": image_elements,
            "table_elements": table_elements,
            "text_elements": text_elements,
            "chart_elements": chart_elements,
            "equation_elements": equation_elements,
            "index_elements": index_elements,
            "elements": page_element,  # Store the original page elements as well
        }

    # Return a new GraphState object containing the parsed page elements
    return GraphState(page_elements=parsed_page_elements)


def extract_page_numbers(state: GraphState):
    return GraphState(page_numbers=list(state["page_elements"].keys()))


def crop_image(state: GraphState):
    """
    Extract and crop images from a PDF file

    :param state: GraphState object
    :return: GraphState object containing the cropped image information
    """
    files = state["filepath"]  # File path
    file_type = state["filetype"]
    page_numbers = state["page_numbers"]  # List of page numbers to process
    if file_type == "image":
        output_folder = os.path.splitext(files[0])[0]  # Set the output folder path
    else:
        output_folder = os.path.splitext(files)[0]  # Set the output folder path
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder

    cropped_images = dict()  # Dictionary to store the cropped image information
    for page_num in page_numbers:
        if file_type == "pdf":
            image_file = ImageCropper.pdf_to_image(
                files, page_num
            )  # Convert the PDF page to an image
        elif file_type == "image":
            image_file = ImageCropper.load_image_without_rotation(files[page_num])

        for element in state["page_elements"][page_num]["image_elements"]:
            if element["category"] == "figure":
                # Normalize the coordinates of the image element
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["coordinates"]
                )

                # Set the path to save the cropped image
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # Crop the image and save it
                ImageCropper.crop_image(image_file, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
    return GraphState(
        images=cropped_images
    )  # Return a new GraphState object containing the cropped image information


def crop_table(state: GraphState):
    """
    Extract and crop tables from a PDF file

    :param state: GraphState object
    :return: GraphState object containing the cropped table image information
    """
    files = state["filepath"]  # PDF file path
    file_type = state["filetype"]
    page_numbers = state["page_numbers"]  # List of page numbers to process
    if file_type == "image":
        output_folder = os.path.splitext(files[0])[0]  # Set the output folder path
    else:
        output_folder = os.path.splitext(files)[0]  # Set the output folder path
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder

    cropped_images = dict()  # Dictionary to store the cropped table image information
    for page_num in page_numbers:
        if file_type == "pdf":
            image_file = ImageCropper.pdf_to_image(
                files, page_num
            )  # Convert the PDF page to an image
        elif file_type == "image":
            image_file = ImageCropper.load_image_without_rotation(files[page_num])
        for element in state["page_elements"][page_num]["table_elements"]:
            if element["category"] == "table":
                # Normalize the coordinates of the table element
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["coordinates"],
                )

                # Set the path to save the cropped table image
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # Crop the table image and save it
                ImageCropper.crop_image(image_file, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
    return GraphState(
        tables=cropped_images
    )  # Return a new GraphState object containing the cropped table image information


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
    # Extract the text data from the GraphState object
    texts = state["texts"]

    # Sort the text data by page number (key) in ascending order
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    inputs = [{"context": Document(page_content=sorted_texts[0][1])}]

    doc_metadata_chain = create_extract_metadata_chain()
    metadata = doc_metadata_chain.invoke(inputs)

    return GraphState(doc_metadata=metadata)


def translate_text(state: GraphState):
    # Extract the text data from the GraphState object
    page_numbers = state["page_numbers"]
    texts = state["texts"]
    translate_lang = state["translate_lang"]

    # Initialize a dictionary to store the translated text
    translated_texts = dict()

    # Sort the text data by page number (key) in ascending order
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # Convert each page's text to a Document object and create an input list
    inputs = [
        {"context": [Document(page_content=text)], "output_language": translate_lang}
        for page_num, text in sorted_texts
    ]

    # Use text_tranlate_chain to generate translations in batch mode
    text_tranlate_chain = create_text_translate_chain()
    translation_results = text_tranlate_chain.batch(inputs)

    # Map the translation results to the page numbers in order
    for page_num, translation in zip(page_numbers, translation_results):
        translated_texts[page_num] = translation

    # Return a new GraphState object containing the translated text
    return GraphState(translated_texts=translated_texts)


def create_text_summary(state: GraphState):
    # Extract the text data from the GraphState object
    page_numbers = state["page_numbers"]
    translate_toggle = state["translate_toggle"]
    translate_lang = state["translate_lang"]
    if translate_toggle:
        texts = state["translated_texts"]
    else:
        texts = state["texts"]

    # Initialize a dictionary to store the summarized text
    text_summary = dict()

    # Sort the text data by page number (key) in ascending order
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # Convert each page's text to a Document object and create an input list
    inputs = [
        {"context": [Document(page_content=text)], "output_language": translate_lang}
        for page_num, text in sorted_texts
    ]

    # Use text_summary_chain to generate summaries in batch mode
    text_summary_chain = create_text_summary_chain()
    summaries = text_summary_chain.batch(inputs)

    # Map the summaries to the page numbers in order
    for page_num, translation in zip(page_numbers, summaries):
        text_summary[page_num] = translation

    # Return a new GraphState object containing the summarized text
    return GraphState(texts_summary=text_summary)


def create_image_summary_data_batches(state: GraphState):
    # Function to create data batches for image summaries
    data_batches = []

    # Sort the page numbers in ascending order
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # Get the summarized text for each page
        text = state["texts_summary"][page_num]
        # Iterate through all image elements for the page
        for image_element in state["page_elements"][page_num]["image_elements"]:
            # Convert the image ID to an integer
            image_id = int(image_element["id"])

            # Add the image information, related text, page number, and ID to the data batch
            data_batches.append(
                {
                    "image": state["images"][image_id],  # Image file path
                    "text": text,  # Related text summary
                    "page": page_num,  # Page number
                    "id": image_id,  # Image ID
                    "lang": state["translate_lang"],  # Language
                }
            )
    # Return a new GraphState object containing the created data batches
    return GraphState(image_summary_data_batches=data_batches)


def create_table_summary_data_batches(state: GraphState):
    # Function to create data batches for table summaries
    data_batches = []

    # Sort the page numbers in ascending order
    page_numbers = sorted(list(state["page_elements"].keys()))

    for page_num in page_numbers:
        # Get the summarized text for each page
        text = state["texts_summary"][page_num]
        # Iterate through all table elements for the page
        for image_element in state["page_elements"][page_num]["table_elements"]:
            # Convert the table ID to an integer
            image_id = int(image_element["id"])

            # Add the table information, related text, page number, and ID to the data batch
            data_batches.append(
                {
                    "table": state["tables"][image_id],  # Table data
                    "text": text,  # Related text summary
                    "page": page_num,  # Page number
                    "id": image_id,  # Table ID
                    "lang": state["translate_lang"],  # Language
                }
            )
    # Return a new GraphState object containing the created data batches
    return GraphState(table_summary_data_batches=data_batches)


def create_image_summary(state: GraphState):
    # Extract image summaries
    # Call the extract_image_summary function to generate image summaries
    image_summaries = extract_image_summary.invoke(state["image_summary_data_batches"])

    # Initialize a dictionary to store the image summaries
    image_summary_output = dict()

    # Iterate through each data batch and image summary
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        # Use the ID of the data batch as the key to store the image summary
        image_summary_output[data_batch["id"]] = image_summary

    # Return a new GraphState object containing the image summaries
    return GraphState(images_summary=image_summary_output)


def create_table_summary(state: GraphState):
    # Extract table summaries
    table_summaries = extract_table_summary.invoke(state["table_summary_data_batches"])

    # Initialize a dictionary to store the table summaries
    table_summary_output = dict()

    # Iterate through each data batch and table summary
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        # Use the ID of the data batch as the key to store the table summary
        table_summary_output[data_batch["id"]] = table_summary

    # Return a new GraphState object containing the table summaries
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

    graph = workflow.compile()

    return graph
