# ⚕️ Advanced Healthcare Q&A Agent

This project implements an advanced Question-Answering system for a healthcare dataset. It leverages a hybrid agentic architecture to handle both analytical and semantic queries, providing a robust and accurate interface for data exploration.

[![Built with LangChain](https://img.shields.io/badge/Built%20with-LangChain-blue.svg)](https://www.langchain.com/)
[![Powered by OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-black.svg)](https://openai.com/)

## Demo

<div align="center">
  <img src="assets/healthcare_agent_demo.gif" width="700" alt="App Demo"/>
</div>

## Architecture

The core of this system is a hybrid agent that acts as an intelligent router. It analyzes the user's query and directs it to one of two specialized tools:

1.  **`PandasDataFrameAnalyzer`**:  
    A tool powered by a Text-to-Pandas agent. It handles analytical queries requiring precise calculations, aggregations, and filtering (e.g., "How many patients have cancer?").

2.  **`PatientRecordSemanticSearch`**:  
    A RAG tool that performs **two-stage retrieval** for semantic queries:
    - **Stage 1:** Vector similarity search retrieves relevant records from a FAISS index.
    - **Stage 2:** A **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) further refines the results for answer faithfulness and context relevance.

This dual-tool approach ensures that each type of query is handled by the most appropriate and accurate method.  
**The semantic search pipeline (vector retrieval + reranker) maximizes answer precision and minimizes hallucinations.**

## Features

- **Hybrid Query Handling:** Seamlessly answers both analytical and semantic questions.
- **Two-Stage Semantic Search:** Uses a vector store and cross-encoder reranker for high-fidelity semantic matching.
- **Robust Tool Selection:** An LLM-powered router agent intelligently chooses the correct tool for the job.
- **Built-in Guardrails:** Politely declines to answer out-of-scope questions or provide medical advice.
- **Comprehensive Evaluation Suite:** Includes scripts to generate a test set, run evaluations, and score the agent's performance on tool selection and answer faithfulness.
- **Interactive UI:** A simple and intuitive web interface built with Streamlit.

## Getting Started

### 1. Prerequisites

- Python 3.13 (or compatible with your packages—see `requirements.txt`)
- An OpenAI API Key
- Node.js (optional, for some LangChain dependencies)

### 2. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/healthcare-agent.git
    cd healthcare-agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    > If you are using Python 3.13 and have compatibility issues, ensure your dependencies in `requirements.txt` are up-to-date.

3.  **Set up environment variables:**
    - Copy the example `.env` file:
      ```bash
      cp .env.example .env
      ```
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY="sk-..."
      ```

4.  **Download the dataset:**
    - Download the [Healthcare Dataset](https://www.kaggle.com/datasets/prasadkharkar/healthcare-dataset) from Kaggle.
    - Place the `healthcare_dataset.csv` file inside the `data/` directory.

### 3. Usage

1.  **Build the Vector Store:**  
    This script preprocesses the data and creates a FAISS index for semantic search. This only needs to be run once.
    ```bash
    PYTHONPATH=$PYTHONPATH:. python src/scripts/01_build_vector_store.py
    ```

2.  **Run the Streamlit Application:**  
    Start the interactive web application.
    ```bash
    PYTHONPATH=$PYTHONPATH:. streamlit run src/app/app.py
    ```
    Navigate to `http://localhost:8501` in your browser to start asking questions.

### 4. Running the Evaluation

To assess the agent's performance, you can run the full evaluation pipeline:

1.  **Generate the evaluation question set:**
    ```bash
    PYTHONPATH=$PYTHONPATH:. python src/scripts/02_generate_evaluation_set.py
    ```

2.  **Run the agent against the evaluation set:**
    ```bash
    PYTHONPATH=$PYTHONPATH:. python evaluation/run_evaluation.py
    ```

3.  **Score the results:**  
    This uses an LLM-as-a-Judge to score semantic faithfulness and generates a summary report.
    ```bash
    PYTHONPATH=$PYTHONPATH:. python src/scripts/03_score_evaluation_results.py
    ```
    The final scored results and summary reports will be saved in the `evaluation/` directory.

### 5. Assets

- Demo GIFs and screenshots should be placed in the `assets/` directory at the root of the repo.
- Example: `assets/healthcare_agent_demo.gif`

## Project Structure

The repository is organized for clarity and scalability:

- `data/`: Holds the raw CSV dataset.
- `evaluation/`: Contains all scripts and data related to agent evaluation.
- `src/`: The main source code.
  - `agent/`: The core agent logic.
  - `app/`: The Streamlit UI code.
  - `core/`: Shared components like data loading, retrieval, and prompts.
  - `scripts/`: One-off scripts for setup and evaluation.
  - `tools/`: Implementations of the agent's tools.
- `vector_store/`: Stores the generated FAISS index.
- `assets/`: Screenshots, GIFs, and other media assets for documentation and demos.

## Technical Details

- **Semantic Search Pipeline:**  
  - Uses [FAISS](https://github.com/facebookresearch/faiss) for fast vector-based retrieval over patient summaries.
  - Results are **reranked** with the cross-encoder model [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), ensuring answers are both relevant and faithful to retrieved context.
- **Evaluation:**  
  - Includes scripts for automated evaluation and LLM-as-a-Judge scoring of semantic answers.

## License

[MIT](LICENSE)
