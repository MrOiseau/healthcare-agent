import streamlit as st
from src.agent.main_agent import create_main_agent
from langchain_core.messages import AIMessage, HumanMessage

# --- App Configuration ---
st.set_page_config(page_title="Healthcare Q&A Agent", layout="wide")
st.title("⚕️ Advanced Healthcare Q&A Agent")
st.caption("Powered by LangChain & OpenAI. Ask analytical or semantic questions about the patient dataset.")

# --- Agent Initialization ---
# Use session state to initialize the agent only once per session.
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Agent... This may take a moment."):
        st.session_state.agent_executor = create_main_agent()

# --- Chat History Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am your healthcare data assistant. How can I help you today?"),
    ]

# --- UI Rendering ---
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)

# --- User Input Handling ---
if user_query := st.chat_input("Ask a question about the healthcare data..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history}
                )
                ai_response_content = response["output"]
                st.write(ai_response_content)
                st.session_state.chat_history.append(AIMessage(content=ai_response_content))
            except Exception as e:
                # A robust error message for the user, while logging the real error for the developer
                print(f"ERROR: Agent execution failed: {e}")
                error_message = "I'm sorry, I encountered a technical issue. Please try rephrasing your question."
                st.error(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))

# Test it:
# PYTHONPATH=$PYTHONPATH:. streamlit run src/app/app.py
