import os
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.core import config
from src.core.data_loader import load_and_preprocess_data


def main():
    """Builds the FAISS vector store from the dataset and saves it to disk."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    print("Loading and preprocessing the full dataset...")
    df = load_and_preprocess_data(config.DATASET_PATH)
    texts = df['semantic_summary'].tolist()
    metadatas = [{"row_index": i} for i in df.index]
    print(f"Data loaded. {len(texts)} records to process.")

    print(f"Initializing embedding model ({config.EMBEDDING_MODEL}) with batch size {config.EMBEDDING_BATCH_SIZE}.")
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        chunk_size=config.EMBEDDING_BATCH_SIZE
    )

    print("\nGenerating embeddings and creating FAISS index...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        # Using tqdm for progress bar on large datasets
        ids=None # Let FAISS handle IDs
    )

    print("\nVector store created successfully.")
    os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
    print(f"Saving vector store to: {config.VECTOR_STORE_PATH}...")
    vectorstore.save_local(config.VECTOR_STORE_PATH)
    print("Vector store saved successfully.")

if __name__ == "__main__":
    main()

# Run it
# PYTHONPATH=$PYTHONPATH:. python src/scripts/01_build_vector_store.py
