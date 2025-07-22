import os
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.core import config
from src.core.data_loader import load_and_preprocess_data


def main():
    """Builds the FAISS vector store from the dataset and saves it to disk."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    print("Loading and preprocessing the full dataset...")
    df = load_and_preprocess_data(config.DATASET_PATH)

    print("Creating Document objects from semantic summaries...")
    documents = [
        Document(page_content=row['semantic_summary'], metadata={"row_index": i})
        for i, row in df.iterrows()
    ]
    num_docs = len(documents)
    print(f"Data loaded. {num_docs} records to process.")

    print(f"Initializing embedding model ({config.EMBEDDING_MODEL}).")
    embeddings_model = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        chunk_size=config.EMBEDDING_BATCH_SIZE
    )

    # Batched embedding with progress bar
    print("\nGenerating embeddings with progress bar...")
    batch_size = 200
    all_embeddings = []
    for i in tqdm(range(0, num_docs, batch_size), desc="Embedding docs"):
        batch_docs = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    print("\nCreating FAISS index...")
    # FAISS expects tuples of (text, embedding), so reassemble as needed
    text_embeddings = list(zip([doc.page_content for doc in documents], all_embeddings))
    metadatas = [doc.metadata for doc in documents]
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings_model,
        metadatas=metadatas
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
