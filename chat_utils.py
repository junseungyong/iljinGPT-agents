from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Create dictionary to store session_id
store = {}


# Get session history based on session_id
def get_session_history(session_ids):
    if session_ids not in store:  # If session_id is not in store
        # Create new ChatMessageHistory object and store it in store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return session history for the corresponding session ID


def create_chain():
    # Define prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the following information, please provide a concise and accurate answer to the question."
                "If the information is not sufficient to answer the question, say so.",
            ),
            # Use chat_history as key for conversation history
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "\n\nHere is the question:\n ------- \n{question}\n ------- \n"
                "\n\nHere is the context:\n ------- \n{context}\n ------- \n",
            ),  # Use user input as variable
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # Function to get session history
        input_messages_key="question",  # Key for user question
        history_messages_key="chat_history",  # Key for history messages
    )

    return chain_with_history.with_config(configurable={"session_id": "abc1234"})
