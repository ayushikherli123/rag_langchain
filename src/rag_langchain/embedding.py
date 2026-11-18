"""
Embedding Pipeline Module
-------------------------

This module provides functionality to transform loaded documents into
numerical vector embeddings using a SentenceTransformer model.

The pipeline consists of two main stages:

1. Document Chunking:
   Splits large documents into smaller, overlapping text chunks using
   LangChain's RecursiveCharacterTextSplitter.

2. Embedding Generation:
   Converts each text chunk into a numerical vector representation
   using a pre-trained SentenceTransformer model.

Dependencies:
    - langchain_text_splitters
    - sentence_transformers
    - numpy
    - rag_langchain.data_loader

Main Components:
    EmbeddingPipeline:
        Handles document chunking and embedding generation.

Example:
    >>> docs = load_all_documents("data")
    >>> pipeline = EmbeddingPipeline()
    >>> chunks = pipeline.chunk_documents(docs)
    >>> embeddings = pipeline.embed_chunks(chunks)
"""
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from rag_langchain.data_loader import load_all_documents

class EmbeddingPipeline:
    """
    A pipeline for splitting documents into chunks and generating embeddings.

    This class uses LangChain's RecursiveCharacterTextSplitter to divide
    documents into manageable text chunks and SentenceTransformer to
    convert those chunks into numerical vector representations.

    Attributes:
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        model (SentenceTransformer): Pre-trained embedding model.

    Example:
        >>> pipeline = EmbeddingPipeline()
        >>> chunks = pipeline.chunk_documents(documents)
        >>> embeddings = pipeline.embed_chunks(chunks)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline.

        Args:
            model_name (str): Name of the SentenceTransformer model.
                Defaults to "all-MiniLM-L6-v2".
            chunk_size (int): Maximum number of characters per chunk.
                Defaults to 1000.
            chunk_overlap (int): Number of overlapping characters
                between consecutive chunks. Defaults to 200.

        Returns:
            None
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller, overlapping text chunks.

        Uses RecursiveCharacterTextSplitter to split documents while
        attempting to preserve natural text boundaries.

        Args:
            documents (List[Any]): List of LangChain Document objects.

        Returns:
            List[Any]: List of smaller Document chunks with metadata preserved.

        Example:
            >>> chunks = pipeline.chunk_documents(documents)
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate numerical embeddings for document chunks.

        Extracts text from each document chunk and passes it to the
        SentenceTransformer model to generate dense vector embeddings.

        Args:
            chunks (List[Any]): List of LangChain Document chunks.

        Returns:
            np.ndarray: Two-dimensional NumPy array containing embeddings.
                Each row represents the vector embedding of one chunk.

        Example:
            >>> embeddings = pipeline.embed_chunks(chunks)
            >>> print(embeddings.shape)
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)