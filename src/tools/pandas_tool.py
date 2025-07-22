import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool
from src.core.prompts import PANDAS_AGENT_PREFIX, PANDAS_TOOL_DESCRIPTION


def create_pandas_tool(df: pd.DataFrame, llm: ChatOpenAI) -> Tool:
    """
    Creates a tool that can query a pandas DataFrame using a dedicated agent.
    ...
    """
    pandas_agent: AgentExecutor = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,  # TODO: Just for development
        prefix=PANDAS_AGENT_PREFIX
    )

    def run_pandas_agent(query: str) -> str:
        """
        Invokes the pandas agent and ensures the final output is a string.
        """
        try:
            response = pandas_agent.invoke({"input": query})
            # Sanitize and convert output to string to prevent agent crashes
            output = response.get("output", "The pandas agent did not return a valid output.")
            return str(output)
        except Exception as e:
            # Catching errors from the sub-agent is crucial
            return f"Error in PandasDataFrameAnalyzer: The query failed with the following error: {e}"

    return Tool(
        name="PandasDataFrameAnalyzer",
        func=run_pandas_agent,
        description=PANDAS_TOOL_DESCRIPTION
    )
