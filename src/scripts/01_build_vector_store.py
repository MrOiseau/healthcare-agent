import os
import time
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
    
    # Combine texts and metadatas for easier batching
    texts = df['semantic_summary'].tolist()
    metadatas = [{"row_index": i} for i in df.index]
    num_texts = len(texts)
    print(f"Data loaded. {num_texts} records to process.")

    print(f"Initializing embedding model ({config.EMBEDDING_MODEL}).")
    # The chunk_size in OpenAIEmbeddings is for its internal batching to the API,
    # which is different from our tqdm batching, so we kept both
    embeddings_model = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        chunk_size=config.EMBEDDING_BATCH_SIZE
    )

    print("\nGenerating embeddings and creating FAISS index (with progress bar)...")
    
    # Batch size for tqdm loop
    batch_size = 200
    vectorstore = None

    # Use tqdm to iterate over the data in batches
    for i in tqdm(range(0, num_texts, batch_size), desc="Embedding texts"):
        # Select the batch of texts and metadatas
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        # Embed the current batch
        embeddings = embeddings_model.embed_documents(batch_texts)
        
        # Create the FAISS index from the first batch
        if vectorstore is None:
            vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(batch_texts, embeddings)), # FAISS needs text-embedding pairs
                embedding=embeddings_model, # Pass the embedding function instance
                metadatas=batch_metadatas
            )
        # For subsequent batches, add them to the existing index
        else:
            vectorstore.add_embeddings(
                text_embeddings=list(zip(batch_texts, embeddings)),
                metadatas=batch_metadatas
            )
        
        # Optional: A small delay to avoid hitting API rate limits
        # time.sleep(0.1)

    if vectorstore:
        print("\nVector store created successfully.")
        os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
        print(f"Saving vector store to: {config.VECTOR_STORE_PATH}...")
        vectorstore.save_local(config.VECTOR_STORE_PATH)
        print("Vector store saved successfully.")
    else:
        print("\nERROR: Vector store could not be created.")


if __name__ == "__main__":
    main()

# Run it
# PYTHONPATH=$PYTHONPATH:. python src/scripts/01_build_vector_store.py
