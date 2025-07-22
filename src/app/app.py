import streamlit as st
from src.agent.main_agent import create_main_agent
from src.core import config
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd


# Helper to fix ArrowTypeError for st.table/st.dataframe
def make_arrow_compatible(df):
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str)
    return df

st.set_page_config(page_title="Healthcare Q&A Agent", layout="wide")

st.markdown(
    """
    <style>
    .chat-bubble {
        display: flex; align-items: flex-start; margin-bottom: 1em;
    }
    .chat-user {
        font-size:1.5em; margin-right: 0.45em; flex-shrink:0;
    }
    .chat-agent {
        font-size:1.5em; margin-right: 0.45em; flex-shrink:0;
    }
    .bubble-user {
        background: #f7f7fa; border-radius: 9px;
        padding: 0.75em 1.1em; font-size: 1.07em; min-width: 0;
    }
    .bubble-agent {
        background: #eef3f8; border-radius: 9px;
        padding: 0.75em 1.1em; font-size: 1.07em; min-width: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚕️ Advanced Healthcare Q&A Agent")
st.caption("Powered by LangChain & OpenAI. Ask analytical or semantic questions about the patient dataset.")

if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Agent... This may take a moment."):
        st.session_state.agent_executor = create_main_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"type": "ai", "message": AIMessage(content="Hello! I am your healthcare data assistant. How can I help you today?"), "response": None},
    ]
if "selected_response_index" not in st.session_state:
    st.session_state.selected_response_index = 0

# ---- SIDEBAR: Persistent chat history (clickable) ----
with st.sidebar:
    st.markdown("### Chat History")
    for idx, item in enumerate(st.session_state.chat_history):
        msg = item["message"]
        if isinstance(msg, HumanMessage):
            st.markdown(f"**👤 You:** {msg.content}")
        else:
            button_label = f"**⧉ AI:** {msg.content[:40]}{'...' if len(msg.content) > 40 else ''}"
            if st.button(button_label, key=f"select_ai_{idx}"):
                st.session_state.selected_response_index = idx
    st.markdown("---")
    st.caption("Click any AI answer above to view details below the chat.")

# --- MAIN AREA: Render custom chat bubbles ---
st.markdown("### Conversation")
for idx, item in enumerate(st.session_state.chat_history):
    msg = item["message"]
    if isinstance(msg, HumanMessage):
        st.markdown(
            f"""
            <div class="chat-bubble">
                <span class="chat-user">👤</span>
                <div class="bubble-user">{msg.content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-bubble">
                <span class="chat-agent">⧉</span>
                <div class="bubble-agent">{msg.content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- INPUT HANDLING ---
user_query = st.chat_input("Ask a question about the healthcare data...")

if user_query:
    st.session_state.chat_history.append({"type": "human", "message": HumanMessage(content=user_query), "response": None})

    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent_executor.invoke(
                {
                    "input": user_query,
                    "chat_history": [item["message"] for item in st.session_state.chat_history],
                }
            )
            ai_response_content = response.get("output", "Sorry, I encountered an issue and couldn't get a response.")
            st.session_state.chat_history.append({"type": "ai", "message": AIMessage(content=ai_response_content), "response": response})
            st.session_state.selected_response_index = len(st.session_state.chat_history) - 1
        except Exception as e:
            st.error("An error occurred while processing your request.")
            st.exception(e)
            ai_response_content = "I'm sorry, I encountered a technical issue. Please try rephrasing your question."
            st.session_state.chat_history.append({"type": "ai", "message": AIMessage(content=ai_response_content), "response": None})
            st.session_state.selected_response_index = len(st.session_state.chat_history) - 1

# --- SEPARATE: Show tabs only for selected AI answer, below chat history ---
st.markdown("---")
st.markdown("### Answer Details")

selected = st.session_state.selected_response_index
if 0 <= selected < len(st.session_state.chat_history):
    selected_item = st.session_state.chat_history[selected]
    if selected_item["type"] == "ai" and selected_item["response"] is not None:
        response = selected_item["response"]
        answer_tab, retrieval_tab, trajectory_tab, config_tab = st.tabs(
            ["📝 Answer", "📑 Retrieval", "🔬 Reasoning", "⚙️ Config"]
        )

        with answer_tab:
            st.write(response.get("output", ""))

        with retrieval_tab:
            st.subheader("Semantic Search Details")
            tool_call = next(
                (step for step in response.get("intermediate_steps", []) if step[0].tool == "PatientRecordSemanticSearch"),
                None
            )
            if tool_call:
                tool_output = tool_call[1]
                parts = tool_output.split("[2. Reranking]", 1)
                if len(parts) == 2:
                    retrieved_section, reranked_section = parts
                    st.markdown("#### 1. Initial Retrieval (from Vector Store)")
                    st.text(retrieved_section)
                    st.markdown("#### 2. Reranked Results (sent to LLM)")
                    st.text(reranked_section)
                else:
                    st.warning("Could not parse retrieval logs. Displaying raw tool output.")
                    st.text(tool_output)
            else:
                st.info("The semantic search tool was not used for this query.")

        with trajectory_tab:
            st.subheader("Agent's Thought Process")
            if response.get("intermediate_steps"):
                for i, step in enumerate(response["intermediate_steps"]):
                    agent_action = step[0]
                    tool_output = step[1]
                    with st.expander(f"Step {i+1}: Tool **{agent_action.tool}**", expanded=True):
                        st.markdown("##### Tool Input:")
                        st.code(agent_action.tool_input, language="text")
                        st.markdown("##### Tool Output (raw):")
                        st.text(tool_output)
            else:
                st.info("No tools were used. The agent answered directly.")

        with config_tab:
            st.subheader("System Configuration")
            config_data = {
                "Agent LLM Model": config.AGENT_LLM_MODEL,
                "Embedding Model": config.EMBEDDING_MODEL,
                "Reranker Model": config.RERANKER_MODEL,
                "Initial Documents Retrieved (k)": config.INITIAL_K_RETRIEVED_DOCS,
                "Final Documents after Reranking (top_n)": config.TOP_K_RERANKED_DOCS,
            }
            config_df = pd.DataFrame(config_data.items(), columns=["Parameter", "Value"])
            st.table(make_arrow_compatible(config_df))

    elif selected_item["type"] == "ai":
        st.info("No details available for this message.")

# Test it:
# PYTHONPATH=$PYTHONPATH:. streamlit run src/app/app.py
