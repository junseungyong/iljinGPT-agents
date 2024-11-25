from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from constants import EMBEDDING_MODEL


def get_hf_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda:1"},  # mps, cpu
        encode_kwargs={"normalize_embeddings": True},
    )


def create_ensemble_retriever(documents):

    hf_embeddings = get_hf_embeddings()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=hf_embeddings,
    )

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever


def create_retriever_from_PDF(file):
    # Store the uploaded file in the cache directory.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Step 1: Load Documents
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # Step 2: Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # Step 3: Create Embeddings
    embeddings = OpenAIEmbeddings()

    # Step 4: Create DB and Save
    # Create a vector store.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # Step 5: Create Retriever
    # Search and create information included in the document.
    retriever = vectorstore.as_retriever()
    return retriever
