"""
Main Streamlit application for the Healthcare Q&A Agent. Handles all user interactions,
chat history, session management, and seamless integration with the agent pipeline.
Renders answers, intermediate reasoning, insights, and system configuration for
transparent, reproducible data exploration.

Intended as the primary user interface for real-time healthcare dataset analysis.

Usage:
    PYTHONPATH=$PYTHONPATH:. streamlit run src/app/app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import copy
from typing import Any, Dict, List, Optional
from datetime import datetime
from src.agent.main_agent import create_main_agent
from src.core import config
from src.core.data_loader import load_and_preprocess_data
from langchain_core.messages import AIMessage, HumanMessage
from src.app.app_helpers import (
    make_arrow_compatible,
    generate_session_id,
    make_json_safe,
    make_response_serializable,
    get_session_file,
)
from src.app.insights import render_insights_tab
from src.core.guardrails import guardrail_response, output_guardrail


# --- Handle new chat creation (must be at the top for rerun logic) ---
if 'start_new_chat' in st.session_state and st.session_state.start_new_chat:
    new_session_id = generate_session_id()
    st.query_params.session_id = new_session_id
    st.session_state.clear()
    st.rerun()

# --- Streamlit Page Config ---
st.set_page_config(page_title="Healthcare Q&A Agent", layout="wide")

# --- Caching expensive resources ---
@st.cache_resource
def get_agent_executor() -> Any:
    """Create and cache the main LangChain agent executor."""
    return create_main_agent()

@st.cache_data
def get_full_dataframe(path: str) -> pd.DataFrame:
    """Load and cache the preprocessed full DataFrame."""
    return load_and_preprocess_data(path)

# --- Persistent Session Management ---
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

def get_or_create_session_id() -> str:
    """
    Get current session ID from query params or create a new one.
    """
    query_params = st.query_params
    if "session_id" in query_params:
        return query_params.session_id
    session_id = generate_session_id()
    st.query_params.session_id = session_id
    return session_id

def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Load the chat history for a given session.
    If file is missing or corrupt, starts fresh.
    """
    file_path = get_session_file(SESSION_DIR, session_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Patch legacy/corrupt response field if needed
            for msg in data:
                if "response" in msg and isinstance(msg["response"], str):
                    try:
                        resp_candidate = json.loads(msg["response"])
                        if isinstance(resp_candidate, dict):
                            msg["response"] = resp_candidate
                    except Exception:
                        pass
            return data
        except Exception as e:
            corrupt_path = file_path + ".corrupt"
            try:
                os.rename(file_path, corrupt_path)
            except Exception:
                pass
            st.warning(f"Could not load previous session (moved to {corrupt_path}). Reason: {e}. Starting fresh.")
    # Default: single AI greeting
    return [
        {"role": "ai", "content": "Hello! I am your healthcare data assistant. How can I help you today?", "response": None},
    ]

def save_chat_history(session_id: str, history: List[Dict[str, Any]]) -> None:
    """
    Save the chat history to disk in a robust, JSON-safe way.
    """
    file_path = get_session_file(SESSION_DIR, session_id)
    try:
        json_safe_history = []
        for item in history:
            json_item = copy.deepcopy(item)
            if "response" in json_item and json_item["response"] is not None:
                json_item["response"] = make_json_safe(json_item["response"])
            json_safe_history.append(json_item)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save session: {str(e)}")

# --- Session Initialization ---
session_id = get_or_create_session_id()
initial_history = load_chat_history(session_id)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = initial_history
if "selected_response_index" not in st.session_state:
    st.session_state.selected_response_index = 0 if initial_history else 0

# --- Load main resources and UI header ---
with st.spinner("Loading resources..."):
    agent_executor = get_agent_executor()
    full_df = get_full_dataframe(config.DATASET_PATH)

st.title("⚕️ Advanced Healthcare Q&A Agent")
st.caption("Powered by LangChain & OpenAI. Ask analytical or semantic questions about the patient dataset.")

# --- Minimal chat bubble styling ---
st.markdown(
    """
    <style>
    .chat-bubble { display: flex; align-items: flex-start; margin-bottom: 1em; }
    .chat-user { font-size:1.5em; margin-right: 0.45em; flex-shrink:0; }
    .chat-agent { font-size:1.5em; margin-right: 0.45em; flex-shrink:0; }
    .bubble-user { background: #f7f7fa; border-radius: 9px; padding: 0.75em 1.1em; font-size: 1.07em; min-width: 0; }
    .bubble-agent { background: #eef3f8; border-radius: 9px; padding: 0.75em 1.1em; font-size: 1.07em; min-width: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: chat history & session control ---
with st.sidebar:
    st.markdown("### Chat History")
    for idx, item in enumerate(st.session_state.chat_history):
        content = item["content"]
        if item["role"] == "human":
            st.markdown(f"**👤 You:** {content}")
        else:
            button_label = f"**⧉ AI:** {content[:40]}{'...' if len(content) > 40 else ''}"
            if st.button(button_label, key=f"select_ai_{idx}"):
                st.session_state.selected_response_index = idx
                save_chat_history(session_id, st.session_state.chat_history)
    st.markdown("---")
    st.caption("Click any AI answer above to view details below the chat.")

    if st.button("🔄 New Chat"):
        st.session_state.start_new_chat = True
        st.rerun()

# --- Main conversation UI ---
st.markdown("### Conversation")
for item in st.session_state.chat_history:
    role = item["role"]
    content = item["content"]
    bubble_class, icon = ("bubble-user", "👤") if role == "human" else ("bubble-agent", "⧉")
    st.markdown(
        f'<div class="chat-bubble"><span class="chat-user">{icon}</span><div class="{bubble_class}">{content}</div></div>',
        unsafe_allow_html=True
    )

# --- Handle user input and invoke agent ---
if user_query := st.chat_input("Ask a question about the healthcare data..."):
    st.session_state.chat_history.append({
        "role": "human", 
        "content": user_query, 
        "response": None
    })
    
    # Guardrail layer first
    refusal = guardrail_response(user_query)
    if refusal:
        st.session_state.chat_history.append({
            "role": "ai",
            "content": refusal,
            "response": None
        })
        st.session_state.selected_response_index = len(st.session_state.chat_history) - 1
        save_chat_history(session_id, st.session_state.chat_history)
        st.rerun()
        
    # Only process if in-scope
    with st.spinner("Thinking..."):
        try:
            langchain_messages = []
            for item in st.session_state.chat_history:
                if item["role"] == "human":
                    langchain_messages.append(HumanMessage(content=item["content"]))
                else:
                    langchain_messages.append(AIMessage(content=item["content"]))

            raw_response = agent_executor.invoke(
                {"input": user_query, "chat_history": langchain_messages}
            )
            serializable_response = make_response_serializable(raw_response)
            safe_response = make_json_safe(serializable_response)

            # Final output scan - guardrail output filter
            ai_response_content = serializable_response.get("output", "Sorry, I couldn't get a response.")
            ai_response_content = output_guardrail(ai_response_content)

            st.session_state.chat_history.append(
                {"role": "ai", "content": ai_response_content, "response": safe_response}
            )
            st.session_state.selected_response_index = len(st.session_state.chat_history) - 1

            save_chat_history(session_id, st.session_state.chat_history)
        except Exception as e:
            st.error("An error occurred while processing your request.")
            st.exception(e)
            ai_response_content = "I'm sorry, I encountered a technical issue. Please try rephrasing your question."
            st.session_state.chat_history.append({"role": "ai", "content": ai_response_content, "response": None})
            st.session_state.selected_response_index = len(st.session_state.chat_history) - 1

            save_chat_history(session_id, st.session_state.chat_history)
    st.rerun()

# --- Answer details (tabs) ---
st.markdown("---")
st.markdown("### Answer Details")

history_len = len(st.session_state.chat_history)
if history_len == 0:
    st.info("No messages yet. Start a conversation!")
else:
    # Handle out-of-bounds or missing selected index
    if ("selected_response_index" not in st.session_state or
        st.session_state.selected_response_index >= history_len or
        st.session_state.selected_response_index < 0):
        ai_indices = [i for i, m in enumerate(st.session_state.chat_history) if m['role'] == 'ai']
        st.session_state.selected_response_index = ai_indices[-1] if ai_indices else history_len - 1

    selected_idx = st.session_state.selected_response_index
    selected_item = st.session_state.chat_history[selected_idx]

    if selected_item["response"] is None:
        st.info("No details for this answer.")

    if selected_item["role"] == "ai" and selected_item.get("response"):
        response = selected_item["response"]
        answer_tab, trajectory_tab, insights_tab, config_tab = st.tabs(
            ["📝 Answer", "🔬 Reasoning", "📈 Insights", "⚙️ Config"]
        )

        with answer_tab:
            st.write(response.get("output", ""))

        with trajectory_tab:
            st.subheader("Agent's Thought Process")
            if not response.get("intermediate_steps"):
                st.info("No tools were used. The agent answered directly.")
            else:
                for i, step in enumerate(response["intermediate_steps"]):
                    agent_action_dict, tool_output_str = step
                    tool_name = agent_action_dict['tool']
                    tool_input = agent_action_dict['tool_input']

                    with st.expander(f"Step {i+1}: Tool **{tool_name}**", expanded=True):
                        try:
                            tool_output_data = json.loads(tool_output_str)
                            if "error" in tool_output_data:
                                st.error(f"Tool Error: {tool_output_data['error']}")
                        except (json.JSONDecodeError, TypeError):
                            tool_output_data = None

                        if tool_name == "PandasDataFrameAnalyzer" and tool_output_data:
                            st.markdown("##### Tool Input (Query for Pandas Agent)")
                            st.code(tool_output_data.get("tool_input", "N/A"), language="text")
                            st.markdown("##### Code Executed")
                            st.code(tool_output_data.get("executed_code", "N/A"), language="python")
                            st.markdown("##### Tool Output (Result)")
                            st.text(tool_output_data.get("output", "N/A"))
                        elif tool_name == "PatientRecordSemanticSearch" and tool_output_data:
                            st.markdown("##### Tool Input (Search Query)")
                            st.code(tool_output_data.get("tool_input", "N/A"), language="text")

                            tab1, tab2, tab3 = st.tabs(["Retrieved (pre-rerank)", "Reranked (post-rerank)", "Final Output"])

                            with tab1:
                                st.markdown("Raw text chunks + metadata from vector store.")
                                for doc in tool_output_data.get("retrieved_docs", []):
                                    st.text_area(
                                        f"Row {doc.get('metadata', {}).get('row_index', 'N/A')}",
                                        doc.get('page_content', ''),
                                        height=100,
                                        key=f"retrieved_{i}_{doc.get('metadata', {}).get('row_index', 'N/A')}_pre"
                                    )
                            with tab2:
                                st.markdown("Top-N chunks delivered to the LLM after reranking.")
                                for doc in tool_output_data.get("reranked_docs", []):
                                    st.text_area(
                                        f"Row {doc.get('metadata', {}).get('row_index', 'N/A')}",
                                        doc.get('page_content', ''),
                                        height=100,
                                        key=f"reranked_{i}_{doc.get('metadata', {}).get('row_index', 'N/A')}_post"
                                    )
                            with tab3:
                                st.markdown("Final string returned by the tool.")
                                st.text(tool_output_data.get("final_output", "N/A"))

                        else:
                            st.markdown("##### Tool Input")
                            st.code(tool_input, language="text")
                            st.markdown("##### Tool Output (Raw)")
                            st.text(tool_output_str)

        with insights_tab:
            render_insights_tab(response, full_df)

        with config_tab:
            st.subheader("System Configuration")
            config_data = {k: getattr(config, k) for k in [
                "AGENT_LLM_MODEL", "EMBEDDING_MODEL", "RERANKER_MODEL",
                "INITIAL_K_RETRIEVED_DOCS", "TOP_K_RERANKED_DOCS"
            ]}
            st.table(make_arrow_compatible(pd.DataFrame(config_data.items(), columns=["Parameter", "Value"])))

    elif selected_item["role"] == "ai":
        st.info("No details available for this message (e.g., initial greeting or error).")
    else:
        st.info("Select an AI answer from the chat history to see details.")
