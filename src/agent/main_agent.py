import pandas as pd
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.core import config
from src.core.data_loader import load_and_preprocess_data
from src.core.retrieval import create_retriever
from src.tools.pandas_tool import create_pandas_tool
from src.tools.vector_search_tool import create_vector_search_tool
from src.core.prompts import MAIN_AGENT_PROMPT


def create_main_agent(sample_size: Optional[int] = None) -> AgentExecutor:
    """Constructs and configures the main healthcare agent executor.

    This function serves as the central factory for the agent. It orchestrates
    the key components: the LLM, the data, and the specialized tools. The agent
    is designed as a router to intelligently delegate user queries to the most
    appropriate tool:
        - PandasDataFrameAnalyzer: For structured, analytical queries (e.g., counts, averages).
        - PatientRecordSemanticSearch: For semantic, similarity-based queries (e.g., 'find similar cases').

    Args:
        sample_size: If provided, the agent will operate on a random subset
            of the data. This is primarily for faster testing and evaluation.
            If None, the full dataset is used.

    Returns:
        A runnable LangChain AgentExecutor instance, ready to process
        queries via its `.invoke()` method.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in the environment.")

    llm = ChatOpenAI(model=config.AGENT_LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

    df = load_and_preprocess_data(config.DATASET_PATH, sample_size=sample_size)

    pandas_tool = create_pandas_tool(df, llm)
    retriever = create_retriever()
    vector_tool = create_vector_search_tool(retriever)
    tools = [pandas_tool, vector_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MAIN_AGENT_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
