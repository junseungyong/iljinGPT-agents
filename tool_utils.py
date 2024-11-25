from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_teddynote.tools import GoogleNews
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing import List, Dict


class RelatedQuestions(BaseModel):
    related_questions: list[str] = Field(
        description="A list of 5 related questions based on the given context and question."
    )


def format_searched_tavily(search_results):
    # Get the value of the 'results' key from search_results
    docs = search_results.get("results", [])

    return [
        Document(
            page_content=doc["content"],
            metadata={"title": doc["title"], "source": doc["url"]},
        )
        for doc in docs
    ]


def format_searched_news(search_results):
    return [
        Document(
            page_content=result["content"],
            metadata={"source": result["url"]},
        )
        for result in search_results
    ]


@tool
def search_tavily(query: str) -> List[Dict[str, str]]:
    """Search Tavily by input keyword"""
    tavily_tool = TavilyClient()
    results = tavily_tool.search(query=query, max_results=5, search_depth="basic")

    result_docs = format_searched_tavily(results)
    return result_docs


@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    result_news = news_tool.search_by_keyword(query, k=5)

    result_docs = format_searched_news(result_news)

    return result_docs


@tool
def create_related_info(query: str, context: list[Document]) -> list[str]:
    """Create related information by input keyword and context"""

    output_parser = PydanticOutputParser(pydantic_object=RelatedQuestions)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant that generates related questions based on the given context and question."
                "Use the provided information to generate 5 related questions that can help further explore the topic."
                "Output must be in JSON format matching the defined structure.",
            ),
            (
                "human",
                "Based on the following context and question, generate 5 related questions."
                "\n\nContext:\n ------- \n{context}\n ------- \n"
                "\n\nQuestion:\n ------- \n{question}\n ------- \n"
                "\n\nFormat:\n ------- \n{format}\n ------- \n",
            ),
        ]
    )

    # Combine prompt template and output parser
    prompt_with_parser = prompt.partial(format=output_parser.get_format_instructions())

    # Initialize the LLM
    # llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Create the chain
    chain = prompt_with_parser | llm | output_parser

    result = chain.invoke({"context": context, "question": query})

    return result.related_questions


SEARCH_TOOLS = [search_tavily, create_related_info]
SEARCH_TOOLS_OLLAMA = [search_tavily]
