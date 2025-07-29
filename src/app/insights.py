"""
Streamlit module for generating visual and tabular insights based on agent tool output.
Includes logic for dynamic visualization of pandas queries and vector search comparisons,
enabling interactive, data-driven user feedback within the app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from typing import Optional, Tuple, List, Any
from src.app.app_helpers import make_arrow_compatible


def _get_last_tool_step(response: dict) -> Optional[Any]:
    """
    Returns the last tool step (action, observation) tuple from the agent response,
    or None if no steps are present.

    Args:
        response: Agent response dictionary.

    Returns:
        Last tool step tuple or None.
    """
    if response and response.get("intermediate_steps"):
        return response["intermediate_steps"][-1]
    return None

def _parse_pandas_code(code: str) -> List[str]:
    """
    Extracts DataFrame column names referenced in pandas code.

    Args:
        code: Source code as a string.

    Returns:
        List of unique column names referenced in the code.
    """
    if not isinstance(code, str):
        return []
    single_cols = re.findall(r"df\[['\"]([^'\"]+)['\"]\]", code)
    multi_cols_match = re.search(r"df\[\[(.*?)\]\]", code)
    if multi_cols_match:
        multi_cols_str = multi_cols_match.group(1)
        found_multi = [c.strip().strip("'\"") for c in multi_cols_str.split(',')]
        single_cols.extend(found_multi)
    return list(set(single_cols))

def _parse_pandas_filter_condition(code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a simple filter condition like df['col'] == 'value' from pandas code.

    Args:
        code: Source code as a string.

    Returns:
        Tuple (column_name, value) or (None, None) if not found.
    """
    match = re.search(r"df\[['\"]([^'\"]+)['\"]\]\s*==\s*['\"]([^'\"]+)['\"]", code)
    if match:
        return match.group(1), match.group(2)
    return None, None

def _generate_pandas_insight(executed_code: str, full_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Given pandas code and the full DataFrame, attempts to infer user intent
    and generate an appropriate visualization (bar, histogram, etc).

    Args:
        executed_code: Code executed to answer the user query.
        full_df: Full DataFrame context.

    Returns:
        Plotly figure, or None if no insight can be generated.
    """
    columns = _parse_pandas_code(executed_code)
    if not columns:
        return None

    filter_col, filter_val = _parse_pandas_filter_condition(executed_code)
    aggregation_match = re.search(r"\.(max|min|mean|sum)\(\)", executed_code)

    # Insight: Aggregation by group, highlight user category
    if aggregation_match and filter_col and len(columns) >= 1:
        agg_func = aggregation_match.group(1)
        numeric_col = next((c for c in columns if c != filter_col and pd.api.types.is_numeric_dtype(full_df[c])), None)
        if not numeric_col:
            return None
        st.subheader(f"Insight: {agg_func.title()} of '{numeric_col.replace('_', ' ').title()}' by '{filter_col.replace('_', ' ').title()}'")
        st.caption(
            f"This chart broadens your query about '{filter_val}' to show the {agg_func} {numeric_col.replace('_', ' ')} across the top 20 categories in {filter_col.replace('_', ' ')}, with yours highlighted."
        )
        insight_df = full_df.groupby(filter_col)[numeric_col].agg(agg_func).nlargest(20).sort_values(ascending=False)
        colors = ['#636EFA'] * len(insight_df)
        if filter_val in insight_df.index:
            idx = insight_df.index.get_loc(filter_val)
            colors[idx] = '#FFA15A'
        fig = px.bar(
            insight_df,
            x=insight_df.index,
            y=insight_df.values,
            title=f"{agg_func.title()} {numeric_col.replace('_', ' ').title()} by {filter_col.replace('_', ' ').title()} (Top 20)",
            labels={'x': filter_col.replace('_', ' ').title(), 'y': f"{agg_func.title()} {numeric_col.replace('_', ' ').title()}"}
        )
        fig.update_traces(marker_color=colors)
        return fig

    # Insight: Distribution/counts by a categorical variable, highlight user's choice
    if ('len(df' in executed_code or '.count()' in executed_code) and filter_col:
        st.subheader(f"Insight: Distribution of '{filter_col.replace('_', ' ').title()}'")
        st.caption(
            f"Your query focused on '{filter_val}'. This chart shows the distribution across the top 20 categories, with yours highlighted."
        )
        counts = full_df[filter_col].value_counts().nlargest(20)
        colors = ['#636EFA'] * len(counts)
        if filter_val in counts.index:
            idx = counts.index.get_loc(filter_val)
            colors[idx] = '#FFA15A'
        fig = px.bar(
            counts,
            x=counts.index,
            y=counts.values,
            title=f"Top 20 Counts for {filter_col.replace('_', ' ').title()}",
            labels={'x': filter_col.replace('_', ' ').title(), 'y': 'Count'}
        )
        fig.update_traces(marker_color=colors)
        return fig

    # Fallback: Histogram for a single numeric column
    if len(columns) == 1 and pd.api.types.is_numeric_dtype(full_df[columns[0]]):
        col = columns[0]
        st.subheader(f"Insight: Distribution of '{col.replace('_', ' ').title()}'")
        st.caption(
            f"This histogram shows the overall distribution for the '{col}' column you queried."
        )
        fig = px.histogram(full_df.head(5000), x=col, title=f"Distribution of {col.replace('_', ' ').title()} (Sample of 5000)")
        return fig

    # Fallback: Bar chart for top categories in a single categorical column
    if len(columns) == 1 and not pd.api.types.is_numeric_dtype(full_df[columns[0]]):
        col = columns[0]
        st.subheader(f"Insight: Top 20 Categories in '{col.replace('_', ' ').title()}'")
        st.caption(
            f"This bar chart shows the most frequent categories for the '{col}' column you queried."
        )
        counts = full_df[col].value_counts().nlargest(20)
        fig = px.bar(
            counts,
            x=counts.index,
            y=counts.values,
            title=f"Top 20 Counts for {col.replace('_', ' ').title()}",
            labels={'x': col.replace('_', ' ').title(), 'y': 'Count'}
        )
        return fig

    return None

def _generate_vector_search_insight(reranked_docs: List[dict]) -> Optional[pd.DataFrame]:
    """
    Builds a comparison table for the top retrieved documents (e.g., patients) from vector search.

    Args:
        reranked_docs: List of dicts, each representing a document/chunk.

    Returns:
        DataFrame of extracted attributes for side-by-side comparison, or None.
    """
    if not reranked_docs or len(reranked_docs) < 2:
        return None

    st.subheader("Insight: Comparison of Top Retrieved Patient Records")
    st.caption("This table provides a side-by-side comparison of the key attributes from the most relevant records found for your query.")

    comparison_data = []
    for doc in reranked_docs[:5]:
        text = doc.get('page_content', '')
        data = {
            "Name": re.search(r"Patient (.*?) \(", text).group(1) if re.search(r"Patient (.*?) \(", text) else "N/A",
            "Age": re.search(r"Age: (\d+)", text).group(1) if re.search(r"Age: (\d+)", text) else "N/A",
            "Gender": re.search(r"Gender: (\w+)", text).group(1) if re.search(r"Gender: (\w+)", text) else "N/A",
            "Condition": re.search(r"for (.*?)\.", text).group(1) if re.search(r"for (.*?)\.", text) else "N/A",
            "Admission Date": re.search(r"admitted on ([\d\-]+)", text).group(1) if re.search(r"admitted on ([\d\-]+)", text) else "N/A",
            "Discharge Date": re.search(r"Discharged on ([\d\-]+)", text).group(1) if re.search(r"Discharged on ([\d\-]+)", text) else "N/A",
            "Test Results": re.search(r"test results were ([\w ]+)\.", text).group(1) if re.search(r"test results were ([\w ]+)\.", text) else "N/A",
            "Billing Amount": re.search(r"\$(\d+\.?\d*)", text).group(1) if re.search(r"\$(\d+\.?\d*)", text) else "N/A",
        }
        comparison_data.append(data)
    if not comparison_data:
        return None
    df = pd.DataFrame(comparison_data)
    return df

def render_insights_tab(response: dict, full_df: pd.DataFrame) -> None:
    """
    Renders the Insights tab in Streamlit with a visualization or table based on the
    tool used in the agent's last step.

    Args:
        response: Agent response dict (should include 'intermediate_steps').
        full_df: Full DataFrame used for the session context.
    """
    last_step = _get_last_tool_step(response)
    if not last_step:
        st.info("No tools were used for this response, so no specific insights are available.")
        return
    agent_action_dict = last_step[0]
    tool_name = agent_action_dict['tool']
    tool_output_str = last_step[1]
    try:
        tool_output_data = json.loads(tool_output_str)
    except (json.JSONDecodeError, TypeError):
        st.warning("Could not parse the tool's output to generate insights.")
        st.text_area("Raw Tool Output", tool_output_str, height=150)
        return
    with st.spinner("Generating insight..."):
        insight_generated = False
        if tool_name == "PandasDataFrameAnalyzer" and "executed_code" in tool_output_data:
            fig = _generate_pandas_insight(tool_output_data["executed_code"], full_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                insight_generated = True
        elif tool_name == "PatientRecordSemanticSearch" and "reranked_docs" in tool_output_data:
            df_insight = _generate_vector_search_insight(tool_output_data["reranked_docs"])
            if df_insight is not None and not df_insight.empty:
                st.dataframe(make_arrow_compatible(df_insight), use_container_width=True)
                insight_generated = True
        if not insight_generated:
            st.info("An insight could not be automatically generated for this specific query. Please check the 'Reasoning' tab for details on how the agent arrived at the answer.")
