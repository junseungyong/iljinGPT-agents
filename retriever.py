from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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
