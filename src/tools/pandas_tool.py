# src/tools/pandas_tool.py
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool
from src.core.prompts import PANDAS_AGENT_PREFIX, PANDAS_TOOL_DESCRIPTION

def create_pandas_tool(df: pd.DataFrame, llm: ChatOpenAI) -> Tool:
    """
    Creates a tool that can query a pandas DataFrame using a dedicated agent.

    This tool wraps a specialized pandas agent, providing it as a capability
    to the main routing agent.

    Args:
        df: The pandas DataFrame to be queried.
        llm: The language model to power the pandas agent.

    Returns:
        A LangChain Tool instance for the main agent to use.
    """
    pandas_agent: AgentExecutor = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        handle_parsing_errors=True,
        # TODO - SECURITY WARNING: allow_dangerous_code=True is a security risk.
        # In a production environment, this should be replaced with a sandboxed
        # execution environment (e.g., using Docker or a service like e2b)
        # to prevent arbitrary code execution
        allow_dangerous_code=True,
        prefix=PANDAS_AGENT_PREFIX
    )

    def run_pandas_agent(query: str) -> str:
        """Invokes the pandas agent and ensures the output is a clean string."""
        try:
            response = pandas_agent.invoke({"input": query})
            output = response.get("output", "The pandas agent did not return a valid output.")
            return str(output)
        except Exception as e:
            # Catching errors from the sub-agent is crucial for main agent stability
            return f"Error in PandasDataFrameAnalyzer: The query failed with the following error: {e}"

    return Tool(
        name="PandasDataFrameAnalyzer",
        func=run_pandas_agent,
        description=PANDAS_TOOL_DESCRIPTION
    )
