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
        print(f"\nRunning question {index + 1}/{len(eval_df)}: \"{question}\"")
        
        try:
            response = agent_executor.invoke({"input": question, "chat_history": []})
            
            # Safely extract intermediate steps for analysis
            tool_used, tool_input, tool_output = "N/A", "N/A", "N/A"
            if "intermediate_steps" in response and response["intermediate_steps"]:
                first_step = response["intermediate_steps"][0]
                tool_used = first_step[0].tool
                tool_input_obj = first_step[0].tool_input
                
                # Serialize tool input to a string for consistent CSV storage
                tool_input = json.dumps(tool_input_obj) if isinstance(tool_input_obj, dict) else str(tool_input_obj)
                tool_output = str(first_step[1])

            results.append({
                "generated_answer": response.get('output', 'No output found.'),
                "tool_used": tool_used,
                "tool_input": tool_input,
                # Context is only relevant if the vector search tool was used
                "retrieved_context": tool_output if tool_used == "PatientRecordSemanticSearch" else "N/A"
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
    print(f"\nEvaluation complete. Results saved to {config.EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()

# Run it
# PYTHONPATH=$PYTHONPATH:. python evaluation/run_evaluation.py
