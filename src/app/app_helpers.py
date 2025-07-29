"""
Helper utilities for the Streamlit Healthcare Q&A Agent app ("src/app/app.py"). 
Provides functions for session ID generation, DataFrame compatibility with Streamlit, 
JSON-safe serialization, and robust response handling. 
Used throughout the UI for state and data management.
"""

import pandas as pd
import json
import uuid
import os
from typing import Any


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Converts object columns to string to prevent Streamlit ArrowTypeError."""
    for col in df.columns:
        if df[col].dtype == 'O':  # 'O' for object
            df[col] = df[col].astype(str)
    return df

def generate_session_id() -> str:
    """
    Generate a unique, human-readable session ID with datetime and random suffix.
    """
    from datetime import datetime
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rand = uuid.uuid4().hex[:8]
    return f"{dt}_{rand}"

def make_json_safe(obj: Any) -> Any:
    """
    Recursively ensure an object is JSON-safe for serialization.
    """
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return json.loads(json.dumps(obj, default=str))

def make_response_serializable(response: dict) -> dict:
    """
    Converts the agent's response to a JSON-serializable dict.
    """
    if not response or "intermediate_steps" not in response:
        return response
    serializable_steps = []
    for step in response["intermediate_steps"]:
        action, observation = step
        serializable_action = {
            "tool": getattr(action, "tool", None) if hasattr(action, "tool") else action.get("tool"),
            "tool_input": json.dumps(getattr(action, "tool_input", None) if hasattr(action, "tool_input") else action.get("tool_input"))
                        if isinstance(
                            getattr(action, "tool_input", None)
                            if hasattr(action, "tool_input")
                            else action.get("tool_input"),
                            (dict, list, tuple, set),
                        )
                        else str(getattr(action, "tool_input", None) if hasattr(action, "tool_input") else action.get("tool_input")),
            "log": getattr(action, "log", "") if hasattr(action, "log") else action.get("log", "")
        }
        serializable_steps.append((serializable_action, observation))
    serializable_response = {
        "input": response.get("input"),
        "chat_history": response.get("chat_history"),
        "output": response.get("output"),
        "intermediate_steps": serializable_steps
    }
    return serializable_response

def get_session_file(session_dir: str, session_id: str) -> str:
    """Return the file path for a session given its ID."""
    return os.path.join(session_dir, f"{session_id}.json")
