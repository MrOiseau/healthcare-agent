import os

# --- Environment & API Keys ---
# Load from .env file, which should be in the root directory
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM & Embedding Models ---
AGENT_LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Data & Storage Paths ---
DATA_DIR = "data"
EVAL_DIR = "evaluation"
VECTOR_STORE_DIR = "vector_store"

DATASET_FILENAME = "healthcare_dataset.csv"
VECTOR_STORE_INDEX_NAME = "faiss_index"
EVAL_SET_FILENAME = "evaluation_set.csv"
EVAL_RESULTS_FILENAME = "evaluation_results.csv"
SCORED_RESULTS_FILENAME = "evaluation_scored_results.csv"
SUMMARY_REPORT_TXT_FILENAME = "evaluation_summary_report.txt"
SUMMARY_REPORT_XLSX_FILENAME = "evaluation_summary_report.xlsx"

# Construct full paths
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, VECTOR_STORE_INDEX_NAME)
EVAL_SET_PATH = os.path.join(EVAL_DIR, EVAL_SET_FILENAME)
EVAL_RESULTS_PATH = os.path.join(EVAL_DIR, EVAL_RESULTS_FILENAME)
SCORED_RESULTS_PATH = os.path.join(EVAL_DIR, SCORED_RESULTS_FILENAME)
SUMMARY_REPORT_TXT_PATH = os.path.join(EVAL_DIR, SUMMARY_REPORT_TXT_FILENAME)
SUMMARY_REPORT_XLSX_PATH = os.path.join(EVAL_DIR, SUMMARY_REPORT_XLSX_FILENAME)

# --- Retrieval & Embedding Parameters ---
EMBEDDING_BATCH_SIZE = 200
TOP_K_RETRIEVED_DOCS = 5

# --- Evaluation Parameters ---
EVAL_SAMPLE_SIZE = 5000
NUM_PANDAS_QUESTIONS = 20
NUM_VECTOR_QUESTIONS = 20
NUM_GUARDRAIL_QUESTIONS = 10
