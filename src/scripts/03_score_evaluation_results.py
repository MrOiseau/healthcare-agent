import pandas as pd
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core import config


def score_tool_selection(row: pd.Series) -> int:
    """Scores 1 if the agent picked the correct tool, 0 otherwise."""
    if row['query_type'] == 'guardrail':
        # Correct for a guardrail is to use NO tool
        return 1 if pd.isna(row['tool_used']) or row['tool_used'] == 'N/A' else 0
    if pd.notna(row['ideal_tool']) and pd.notna(row['tool_used']):
        # Checks if the used tool name is contained within the ideal tool name
        return 1 if row['ideal_tool'] in row['tool_used'] else 0
    return 0

def score_analytical_answer(row: pd.Series) -> int | None:
    """Scores analytical answers by comparing generated answer to ground truth."""
    if row['query_type'] != 'analytical' or pd.isna(row['ground_truth']) or row['ground_truth'] == 'N/A':
        return None  # Not applicable

    gen_text = str(row['generated_answer']).lower()
    gt_text = str(row['ground_truth']).lower()

    # For list-based answers (e.g. top 3 conditions), check for keyword presence
    if ',' in gt_text and not any(char.isdigit() for char in gt_text):
        gt_items = [item.strip() for item in gt_text.split(',')]
        return 1 if all(item in gen_text for item in gt_items) else 0

    # For numeric answers, extract all numbers and compare.
    # This is robust to formatting ($ ,) and surrounding text
    gen_nums = re.findall(r'-?[\d\.]+', gen_text)
    gt_nums = re.findall(r'-?[\d\.]+', gt_text)
    
    if not gt_nums: return 0
    return 1 if all(num in gen_nums for num in gt_nums) else 0

def get_llm_as_judge_chain():
    """Initializes the LLM-as-a-Judge chain for semantic evaluation."""
    llm = ChatOpenAI(model=config.AGENT_LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        """You are an impartial AI judge evaluating an answer's faithfulness to provided context.
        
        **User Question:** {question}
        **Retrieved Context:**
        {context}
        **Generated Answer:**
        {answer}

        **Instructions:**
        1. Compare the "Generated Answer" to the "Retrieved Context".
        2. Does the answer accurately and exclusively use information from the context?
        3. Output a score of 1 for a faithful answer, or 0 for an unfaithful/hallucinated one.
        4. Provide a brief justification.

        **Your Evaluation (Score and Justification):**
        SCORE: [1 or 0]
        JUSTIFICATION: [Your brief reason]
        """
    )
    return prompt | llm | StrOutputParser()

def score_semantic_answer(row: pd.Series, judge_chain) -> str | None:
    """Scores semantic answers using an LLM-as-a-Judge."""
    if row['query_type'] != 'semantic' or pd.isna(row['retrieved_context']) or row['retrieved_context'] == 'N/A':
        return None
    
    print(f"Judging semantic question (ID: {row.name})...")
    result = judge_chain.invoke({
        "question": row['question'],
        "context": row['retrieved_context'],
        "answer": row['generated_answer']
    })
    
    try:
        score = int(re.search(r'SCORE:\s*(\d)', result, re.IGNORECASE).group(1))
        justification = result.split("JUSTIFICATION:")[1].strip()
        return f"{score} - {justification}"
    except Exception:
        return f"0 - Failed to parse judge's response: {result}"

def generate_summary_report(df: pd.DataFrame):
    """Generates and saves a summary report in TXT and XLSX formats."""
    print("\nGenerating summary report...")

    # Calculate overall metrics
    metrics = {
        "Tool Selection": df['tool_selection_score'].mean(),
        "Analytical Correctness": df['analytical_correctness_score'].mean(),
        "Semantic Faithfulness": df['semantic_numeric_score'].mean()
    }
    
    # Identify failures for detailed analysis
    failures = {
        "Tool Selection": df[df['tool_selection_score'] == 0],
        "Analytical": df[df['analytical_correctness_score'] == 0],
        "Semantic": df[df['semantic_numeric_score'] == 0]
    }

    # --- Build TXT Report ---
    with open(config.SUMMARY_REPORT_TXT_PATH, 'w') as f:
        f.write("--- AGENT EVALUATION SUMMARY ---\n\n")
        f.write("--- OVERALL PERFORMANCE ---\n")
        for name, score in metrics.items():
            f.write(f"{name:<25} {score:.2%}\n")
        
        f.write("\n--- FAILURE ANALYSIS ---\n")
        f.write("\n1. Tool Selection Failures:\n")
        if not failures["Tool Selection"].empty:
            for _, row in failures["Tool Selection"].iterrows():
                f.write(f"  - Q: '{row['question']}' | Ideal: {row['ideal_tool']} | Used: {row['tool_used']}\n")
        else:
            f.write("  - No failures found.\n")
        # (Repeat for other failure types)

    print(f"Text summary report saved to {config.SUMMARY_REPORT_TXT_PATH}")

    # --- Build XLSX Report ---
    summary_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Accuracy'])
    summary_df['Accuracy'] = summary_df['Accuracy'].map('{:.2%}'.format)

    with pd.ExcelWriter(config.SUMMARY_REPORT_XLSX_PATH, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary')
        failures["Tool Selection"].to_excel(writer, sheet_name='Tool_Failures', index=False)
        failures["Analytical"].to_excel(writer, sheet_name='Analytical_Failures', index=False)
        failures["Semantic"].to_excel(writer, sheet_name='Semantic_Failures', index=False)
    
    print(f"Excel report saved to {config.SUMMARY_REPORT_XLSX_PATH}")

def main():
    """Main function to score evaluation results."""
    print(f"Loading evaluation results from {config.EVAL_RESULTS_PATH}...")
    df = pd.read_csv(config.EVAL_RESULTS_PATH)
    
    print("Scoring tool selection...")
    df['tool_selection_score'] = df.apply(score_tool_selection, axis=1)
    
    print("Scoring analytical answers...")
    df['analytical_correctness_score'] = df.apply(score_analytical_answer, axis=1)
    
    print("Initializing LLM-as-a-Judge for scoring semantic answers...")
    judge_chain = get_llm_as_judge_chain()
    df['semantic_quality_score'] = df.apply(lambda row: score_semantic_answer(row, judge_chain), axis=1)
    df['semantic_numeric_score'] = pd.to_numeric(df['semantic_quality_score'].str.extract(r'(\d)', expand=False), errors='coerce')
    
    df.to_csv(config.SCORED_RESULTS_PATH, index=False)
    print(f"\nDetailed scored results saved to {config.SCORED_RESULTS_PATH}")

    generate_summary_report(df)

if __name__ == "__main__":
    main()

# Test it:
# PYTHONPATH=$PYTHONPATH:. python src/scripts/03_score_evaluation_results.py
