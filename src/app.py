"""
RAG Application Entry Point
---------------------------

This module serves as the main entry point for executing the
Retrieval-Augmented Generation (RAG) application.

It integrates the different components of the RAG pipeline:

    1. Data Loading:
       Loads supported documents from the data directory.

    2. Embedding Pipeline:
       Provides document chunking and embedding generation functionality.

    3. Vector Store:
       Initializes and loads the persisted FAISS vector index.

    4. RAG Search:
       Retrieves relevant document chunks from FAISS and uses a
       Groq-hosted LLM to generate context-aware summaries.

Execution Flow:
    - Load source documents.
    - Initialize the FAISS vector store.
    - Load the persisted vector index.
    - Initialize the RAG search pipeline.
    - Submit a natural language query.
    - Retrieve relevant document chunks.
    - Generate and print the final summary.

Project Dependencies:
    - rag_langchain.data_loader
    - rag_langchain.embedding
    - rag_langchain.vector_store
    - rag_langchain.search

Configuration:
    - Source documents should be available in the data directory.
    - The FAISS index should be available in the faiss_store directory.
    - GROQ_API_KEY should be configured in the environment or .env file.

Usage:
    Run the application from the project root:

        python main.py

Example:
    Query:
        What is attention mechanism?

    Output:
        Summary: <LLM-generated response>
"""
from rag_langchain.data_loader import load_all_documents
from rag_langchain.embedding import EmbeddingPipeline
from rag_langchain.vector_store import FaissVectorStore
from rag_langchain.search import RAGSearch

if __name__ == "__main__":
    """
    Execute the RAG application.

    Loads source documents, initializes the vector store, and performs
    semantic retrieval followed by LLM-based summarization.
    """
    docs = load_all_documents('data')
    #chunks = EmbeddingPipeline().chunk_documents(docs)
    #chunked_vectors = EmbeddingPipeline().embed_chunks(chunks)
    #print(chunked_vectors)

    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs) # Execute if running the application first time
    store.load()
    #print(store.query("what is attention mechanism", top_k=3))

    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)