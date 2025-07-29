"""
Defines and constructs the main Healthcare Q&A AgentExecutor for the application.
Orchestrates the agent routing logic, model selection, and specialized tools for both
analytical (pandas) and semantic (RAG/vector search) queries, with robust guardrails.

Provides the entry point for integrating the LLM, retrieval pipelines, and agent prompt logic.
Used by the app and evaluation scripts.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.core import config
from src.core.data_loader import load_and_preprocess_data
from src.core.retrieval import create_retrieval_components
from src.tools.pandas_tool import create_pandas_tool
from src.tools.vector_search_tool import create_vector_search_tool
from src.tools.refusal_tool import create_refusal_tool
from src.core.prompts import MAIN_AGENT_PROMPT


def create_main_agent(sample_size: Optional[int] = None) -> AgentExecutor:
    """
    Constructs the main healthcare agent executor with production-grade guardrails.
    This agent routes questions to:
        - PandasDataFrameAnalyzer: For structured analytical queries.
        - PatientRecordSemanticSearch: For semantic, similarity-based queries.
        - Refusal: For code/meta/medical-advice/out-of-scope queries.

    Args:
        sample_size: If provided, operates on a random subset of the data for
            faster testing and evaluation. If None, the full dataset is used.

    Returns:
        A runnable LangChain AgentExecutor instance.
    """
    # --- API Key Verification ---
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in the environment.")

    # --- Initialize LLM ---
    llm = ChatOpenAI(
        model=config.AGENT_LLM_MODEL,
        temperature=0,
        api_key=config.OPENAI_API_KEY
    )

    # --- Load DataFrame (optionally sampled) ---
    df = load_and_preprocess_data(config.DATASET_PATH, sample_size=sample_size)

    # --- Create Specialized Tools ---
    pandas_tool = create_pandas_tool(df, llm)
    base_retriever, reranker = create_retrieval_components()
    vector_tool = create_vector_search_tool(
        retriever=base_retriever,
        reranker=reranker
    )
    refusal_tool = create_refusal_tool()

    tools = [pandas_tool, vector_tool, refusal_tool]

    # --- Compose Agent Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", MAIN_AGENT_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # --- Create Tool-Calling Agent ---
    agent = create_tool_calling_agent(llm, tools, prompt)

    # --- Build AgentExecutor ---
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
