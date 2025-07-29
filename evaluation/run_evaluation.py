"""
Runs the main Healthcare Q&A Agent over a set of predefined evaluation questions,
capturing its answer, tool selection, and all intermediate steps. This script
executes the agent against the generated evaluation set, logs the results, and
saves detailed output for further scoring and analysis.

Intended usage:
    - Execute after generating the evaluation set with 02_generate_evaluation_set.py.
    - Produces a CSV with agent responses and tool usage for each question.

Usage:
    PYTHONPATH=$PYTHONPATH:. python evaluation/run_evaluation.py
"""

import pandas as pd
import json
from src.agent.main_agent import create_main_agent
from src.core import config


def run_evaluation():
    """
    Runs a set of predefined questions against the healthcare agent using a
    data sample for fast and efficient testing.
    """
    print("Initializing agent for evaluation...")
    print(f"--- The agent will run on a sample of {config.EVAL_SAMPLE_SIZE} rows. ---")

    # Pass the sample size to the agent creator for a lightweight evaluation run
    agent_executor = create_main_agent(sample_size=config.EVAL_SAMPLE_SIZE)

    print(f"Reading evaluation questions from: {config.EVAL_SET_PATH}")
    try:
        eval_df = pd.read_csv(config.EVAL_SET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Evaluation file not found at {config.EVAL_SET_PATH}. "
              "Please run the generation script first: 'python src/scripts/02_generate_evaluation_set.py'")
        return

    results = []
    for index, row in eval_df.iterrows():
        question = row['question']
        print(f'\nRunning question {index + 1}/{len(eval_df)}: "{question}"')

        try:
            response = agent_executor.invoke({"input": question, "chat_history": []})

            # Safely extract intermediate steps for analysis
            tool_used, tool_input, retrieved_context = "N/A", "N/A", "N/A"

            steps = response.get("intermediate_steps", [])
            if steps:
                # For multi‑tool chains the last step is the one that matters
                last_step = steps[-1]
                tool_used = getattr(last_step[0], "tool", "N/A")
                tool_output_str = last_step[1]

                try:
                    # NEW: Parse the JSON output from the tool
                    tool_output_data = json.loads(tool_output_str)

                    if tool_used == "PandasDataFrameAnalyzer":
                        tool_input = tool_output_data.get("tool_input", "N/A")
                        # For pandas, context is the final text output
                        retrieved_context = tool_output_data.get("output", "N/A")

                    elif tool_used == "PatientRecordSemanticSearch":
                        tool_input = tool_output_data.get("tool_input", "N/A")
                        # For vector search, context is the final string passed to the LLM
                        retrieved_context = tool_output_data.get("final_output", "N/A")

                except (json.JSONDecodeError, TypeError):
                    # Fallback for any unexpected non-JSON output
                    tool_input = str(first_step[0].tool_input)
                    retrieved_context = str(tool_output_str)

            results.append({
                "generated_answer": response.get('output', 'No output found.'),
                "tool_used": tool_used,
                "tool_input": tool_input,
                "retrieved_context": retrieved_context
            })

        except Exception as e:
            print(f"  ERROR processing question: {e}")
            results.append({
                "generated_answer": f"ERROR: {e}",
                "tool_used": "ERROR", "tool_input": "ERROR", "retrieved_context": "ERROR"
            })

    results_df = pd.DataFrame(results)
    final_df = pd.concat([eval_df, results_df], axis=1)

    final_df.to_csv(config.EVAL_RESULTS_PATH, index=False)
    print(f"\\nEvaluation complete. Results saved to {config.EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
