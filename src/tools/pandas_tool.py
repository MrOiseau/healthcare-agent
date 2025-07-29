"""
Defines the PandasDataFrameAnalyzer tool for the agent. Wraps a specialized pandas agent
for handling analytical queries, calculations, and DataFrame operations within the
healthcare dataset. Returns structured output for downstream analysis.

Used as one of the main agent tools.
"""

import pandas as pd
import json
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
        # TODO - SECURITY WARNING: allow_dangerous_code=True is a security risk.
        # In a production environment, this should be replaced with a sandboxed
        # execution environment (e.g., using Docker or a service like e2b)
        # to prevent arbitrary code execution
        allow_dangerous_code=True,
        prefix=PANDAS_AGENT_PREFIX,
        return_intermediate_steps=True # Ensure the sub-agent returns steps
    )

    def run_pandas_agent(query: str) -> str:
        """Invokes the pandas agent and returns a structured JSON string with details."""
        try:
            response = pandas_agent.invoke({"input": query})

            # Extract the executed code from the sub-agent's intermediate steps
            executed_code = "Code not found."
            if "intermediate_steps" in response and response["intermediate_steps"]:
                # The pandas agent's first step is usually the python_repl_ast tool call
                action = response["intermediate_steps"][0][0]
                if hasattr(action, 'tool') and action.tool == "python_repl_ast":
                    executed_code = action.tool_input

            structured_output = {
                "tool_input": query,
                "executed_code": executed_code,
                "output": str(response.get("output", "The pandas agent did not return a valid output.")),
            }
            return json.dumps(structured_output)

        except Exception as e:
            return json.dumps({
                "error": f"Error in PandasDataFrameAnalyzer: The query failed with the following error: {e}",
                "tool_input": query,
                "executed_code": "N/A",
                "output": "Error"
            })

    return Tool(
        name="PandasDataFrameAnalyzer",
        func=run_pandas_agent,
        description=PANDAS_TOOL_DESCRIPTION
    )
