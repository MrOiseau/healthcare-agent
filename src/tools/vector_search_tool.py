from langchain.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from src.core.prompts import VECTOR_SEARCH_TOOL_DESCRIPTION
from typing import List


def create_vector_search_tool(retriever: BaseRetriever) -> Tool:
    """
    Creates a tool for performing semantic searches over patient records.

    This uses a wrapper function (`run_and_format_retriever`) to invoke the
    retriever and format its output (a list of Document objects) into a single,
    readable string. This prevents complex objects from breaking the main agent loop.

    Args:
        retriever: The configured retriever from the vector store.

    Returns:
        A robust LangChain tool for semantic retrieval.
    """
    def run_and_format_retriever(query: str) -> str:
        """
        Invokes the retriever and formats the output documents into a single string.
        """
        try:
            docs: List[Document] = retriever.invoke(query)
            
            if not docs:
                return "No relevant patient records were found for this query."
            
            # Formatting the context clearly with metadata helps the LLM trace the source of its information
            formatted_results = [
                f"Record (from row index {doc.metadata.get('row_index', 'N/A')}):\n{doc.page_content}"
                for doc in docs
            ]
            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            return f"Error during semantic search: {e}"

    return Tool(
        name="PatientRecordSemanticSearch",
        func=run_and_format_retriever,
        description=VECTOR_SEARCH_TOOL_DESCRIPTION
    )
