from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain.agents import create_tool_calling_agent, AgentExecutor

from tool_utils import SEARCH_TOOLS, SEARCH_TOOLS_OLLAMA
from chat_utils import get_session_history


# Create GPT agent
def create_gpt_agent():
    tools = SEARCH_TOOLS

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                load_prompt("prompts/agentic-rag-prompt.yaml").template,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    gpt_agent = create_tool_calling_agent(gpt, tools, prompt)

    return gpt_agent


# Create Ollama agent
def create_ollama_agent():
    tools = SEARCH_TOOLS_OLLAMA

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                load_prompt("prompts/agentic-ollama-prompt.yaml").template,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    ollama = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)

    ollama_agent = create_tool_calling_agent(ollama, tools, prompt)

    return ollama_agent


# Create agent with chat history
def create_agent_with_chat_history(model):
    if model == "***ChatGPT***":
        agent = create_gpt_agent()
    else:
        agent = create_ollama_agent()

    agent_executor = AgentExecutor(
        agent=agent,
        tools=SEARCH_TOOLS,
        verbose=False,
        handle_parsing_errors=True,
    )

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # Get session history
        get_session_history,
        # Key for input prompt: "input"
        input_messages_key="input",
        # Key for history prompt: "chat_history"
        history_messages_key="chat_history",
    )

    return agent_with_chat_history
