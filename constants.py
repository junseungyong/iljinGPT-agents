import os
from langchain_community.document_loaders import PyMuPDFLoader

# Root directory
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Database path
DB_PATH = f"{ROOT_DIR}/DB"

# Number of ingestion threads
INGEST_THREADS = os.cpu_count() or 8

# Document loader map
DOCUMENT_MAP = {
    ".pdf": PyMuPDFLoader,
}

# Progress message for graph nodes
PROGRESS_MESSAGE_GRAPH_NODES = {
    "split_pdf": "Processing the document..",
    "merge_image": "Processing the document..",
    "add_analyzed_layout": "Analyzing layout..",
    "extract_page_metadata": "Extracting page metadata..",
    "extract_page_elements": "Extracting page elements..",
    "extract_tag_elements_per_page": "Extracting tag elements..",
    "extract_page_numbers": "Extracting page numbers..",
    "crop_image": "Cropping image..",
    "crop_table": "Cropping table..",
    "extract_page_text": "Extracting page text..",
    "translate_text": "Translating text..",
    "create_text_summary": "Creating text summary..",
    "create_image_summary_data_batches": "Creating image summary data batches..",
    "create_table_summary_data_batches": "Creating table summary data batches..",
    "create_image_summary": "Creating image summary..",
    "create_table_summary": "Creating table summary..",
    "clean_up": "Cleaning up..",
}

# Output directory
OUTPUT_DIR = ".cache/output"

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
