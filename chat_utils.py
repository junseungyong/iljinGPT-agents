from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# session_id 를 저장할 딕셔너리 생성
store = {}


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


def create_chain():
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the following information, please provide a concise and accurate answer to the question."
                "If the information is not sufficient to answer the question, say so.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "\n\nHere is the question:\n ------- \n{question}\n ------- \n"
                "\n\nHere is the context:\n ------- \n{context}\n ------- \n",
            ),  # 사용자 입력을 변수로 사용
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    return chain_with_history.with_config(configurable={"session_id": "abc1234"})
