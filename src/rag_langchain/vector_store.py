"""
FAISS Vector Store Module
-------------------------

This module implements a persistent vector store using FAISS (Facebook AI
Similarity Search) for efficient similarity-based document retrieval.

It integrates SentenceTransformer for generating text embeddings and
LangChain's EmbeddingPipeline for document chunking and embedding generation.

Workflow:
    1. Initialize the FAISS vector store and embedding model.
    2. Load raw documents from the data source.
    3. Split documents into smaller chunks using EmbeddingPipeline.
    4. Generate vector embeddings for each document chunk.
    5. Store embeddings in a FAISS index.
    6. Store associated document metadata using Pickle.
    7. Persist the FAISS index and metadata to disk.
    8. Retrieve relevant document chunks using similarity search.

Storage Structure:
    faiss_store/
        faiss.index     : Persisted FAISS vector index.
        metadata.pkl    : Associated document metadata.

Dependencies:
    - faiss
    - numpy
    - pickle
    - sentence_transformers
    - rag_langchain.embedding

Main Components:
    FaissVectorStore:
        Handles vector generation, indexing, persistence, and retrieval.

Example:
    >>> store = FaissVectorStore("faiss_store")
    >>> store.build_from_documents(documents)
    >>> results = store.query("What is attention mechanism?", top_k=3)
"""
import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from rag_langchain.embedding import EmbeddingPipeline

class FaissVectorStore:
    """
    A persistent vector database implementation using FAISS.

    This class manages the creation, storage, loading, and retrieval of
    document embeddings using FAISS similarity search.

    It uses SentenceTransformer to generate query embeddings and
    EmbeddingPipeline to split documents and generate chunk embeddings.

    Document metadata is maintained separately using a Pickle file,
    while the FAISS index stores the corresponding numerical vectors.

    Attributes:
        persist_dir (str): Directory used to persist the vector index
            and metadata.

        index: FAISS index used for similarity search.

        metadata (List[Any]): Metadata associated with indexed vectors.

        embedding_model (str): Name of the SentenceTransformer model.

        model (SentenceTransformer): Loaded embedding model used for
            query embedding generation.

        chunk_size (int): Maximum character length of each document chunk.

        chunk_overlap (int): Number of overlapping characters between
            consecutive document chunks.

    Example:
        >>> store = FaissVectorStore()
        >>> store.build_from_documents(documents)
        >>> results = store.query("Explain RAG", top_k=5)
    """

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the FAISS vector store.

        Creates the persistence directory if it does not exist and
        initializes the SentenceTransformer embedding model.

        Args:
            persist_dir (str):
                Directory where FAISS index and metadata will be stored.
                Defaults to "faiss_store".

            embedding_model (str):
                Pre-trained SentenceTransformer model used for generating
                embeddings. Defaults to "all-MiniLM-L6-v2".

            chunk_size (int):
                Maximum number of characters in each document chunk.
                Defaults to 1000.

            chunk_overlap (int):
                Number of overlapping characters between consecutive
                chunks. Defaults to 200.

        Returns:
            None
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """
        Build a FAISS vector store from raw documents.

        This method uses EmbeddingPipeline to split the input documents
        into smaller chunks and generate their numerical embeddings.

        The generated embeddings are converted into float32 format
        before being added to the FAISS index.

        Associated document text is stored as metadata.

        After indexing, the FAISS index and metadata are persisted
        to the configured storage directory.

        Args:
            documents (List[Any]):
                List of LangChain Document objects to be indexed.

        Returns:
            None

        Example:
            >>> store.build_from_documents(documents)
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """
        Add numerical embeddings and their metadata to the FAISS index.

        Initializes a FAISS IndexFlatL2 index if one does not already
        exist. Then adds the supplied embeddings to the index and
        appends the corresponding metadata.

        Args:
            embeddings (np.ndarray):
                Two-dimensional NumPy array containing document embeddings.
                Expected data type is float32.

            metadatas (List[Any], optional):
                Metadata associated with each embedding.
                Defaults to None.

        Returns:
            None

        Example:
            >>> store.add_embeddings(embeddings, metadatas)
        """
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        """
        Persist the FAISS index and metadata to disk.

        Saves the FAISS index as a binary index file and serializes
        the associated metadata using Python Pickle.

        Generated files:
            - faiss.index: FAISS vector index.
            - metadata.pkl: Serialized document metadata.

        Returns:
            None

        Example:
            >>> store.save()
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        """
        Load the persisted FAISS index and metadata from disk.

        Reads the previously saved FAISS index and deserializes
        the associated metadata using Pickle.

        The loaded index and metadata are assigned to the current
        vector store instance.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the FAISS index or metadata file does not exist.

            Exception:
                If the index or metadata cannot be loaded successfully.

        Example:
            >>> store.load()
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Perform similarity search using a query embedding.

        Searches the FAISS index using the supplied query vector and
        retrieves the top_k nearest vectors based on L2 (Euclidean)
        distance.

        Each result contains the vector index, calculated distance,
        and its corresponding document metadata.

        Args:
            query_embedding (np.ndarray):
                Two-dimensional NumPy array representing the query embedding.

            top_k (int):
                Maximum number of nearest vectors to retrieve.
                Defaults to 5.

        Returns:
            List[dict]:
                List of search results containing:
                    - index: Position of the matched vector.
                    - distance: L2 distance from the query vector.
                    - metadata: Associated document metadata.

        Example:
            >>> query_embedding = store.model.encode(
            ...     ["Explain attention"]
            ... ).astype("float32")
            >>> results = store.search(query_embedding, top_k=3)
        """
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        """
        Convert a text query into an embedding and retrieve relevant documents.

        Uses the configured SentenceTransformer model to generate
        an embedding for the input query.

        The generated embedding is passed to the search method,
        which performs similarity search against the FAISS index.

        Args:
            query_text (str):
                Natural language query entered by the user.

            top_k (int):
                Maximum number of relevant document chunks to retrieve.
                Defaults to 5.

        Returns:
            List[dict]:
                List of the most relevant document chunks along with
                their similarity distances and metadata.

        Example:
            >>> results = store.query(
            ...     "What is attention mechanism?", top_k=3
            ... )
            >>> print(results)
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))