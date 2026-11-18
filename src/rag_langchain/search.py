"""
RAG Search Module
-----------------

This module implements a Retrieval-Augmented Generation (RAG) search
pipeline that retrieves relevant document chunks from a FAISS vector
store and uses a Groq-hosted LLM to generate query-specific summaries.

Workflow:
    1. Initialize the FAISS vector store.
    2. Check whether the vector store already exists on disk.
    3. Load the existing vector store or build it from source documents.
    4. Convert the user query into an embedding and retrieve relevant
       document chunks.
    5. Combine retrieved text into a context.
    6. Pass the context and query to the Groq LLM.
    7. Return the generated summary.

Dependencies:
    - python-dotenv
    - langchain-groq
    - rag_langchain.vector_store

Environment Variables:
    GROQ_API_KEY: API key required to access the Groq LLM service.

Main Components:
    RAGSearch:
        Handles document retrieval and LLM-based summarization.

Example:
    >>> rag_search = RAGSearch()
    >>> summary = rag_search.search_and_summarize(
    ...     "What is attention mechanism?", top_k=3
    ... )
    >>> print(summary)
"""
import os
from dotenv import load_dotenv
from rag_langchain.vector_store import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    """
    Retrieval-Augmented Generation (RAG) search implementation.

    This class combines semantic search using a FAISS vector store
    with a Groq-hosted Large Language Model (LLM) to generate
    context-aware responses based on retrieved documents.

    The vector store is loaded from persistent storage if available.
    Otherwise, it is built using documents from the data directory.

    Attributes:
        vectorstore (FaissVectorStore):
            FAISS vector store responsible for document retrieval.

        llm (ChatGroq):
            Groq-hosted LLM used to generate summaries from retrieved
            document context.

    Example:
        >>> rag = RAGSearch()
        >>> result = rag.search_and_summarize(
        ...     "Explain RAG architecture", top_k=5
        ... )
        >>> print(result)
    """
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        """
        Initialize the RAG search pipeline.

        Initializes the FAISS vector store and Groq LLM.

        If a previously persisted FAISS index and metadata exist,
        the vector store is loaded. Otherwise, documents are loaded
        from the data directory and used to build the vector store.

        Args:
            persist_dir (str):
                Directory containing the persisted FAISS index and
                metadata files. Defaults to "faiss_store".

            embedding_model (str):
                SentenceTransformer embedding model used by the
                vector store. Defaults to "all-MiniLM-L6-v2".

            llm_model (str):
                Groq-supported LLM model identifier.
                Defaults to "openai/gpt-oss-120b".

        Returns:
            None

        Raises:
            ValueError: If the required GROQ_API_KEY is not configured,
                depending on the underlying Groq client validation.

            Exception: If vector store initialization, document loading,
                or LLM initialization fails.
        """
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant documents and generate an LLM-based summary.

        Performs semantic similarity search against the FAISS vector
        store to retrieve the most relevant document chunks.

        The retrieved text is combined into a context and passed to
        the Groq LLM along with the original user query.

        Args:
            query (str):
                User's natural language question or search query.

            top_k (int):
                Number of most relevant document chunks to retrieve.
                Defaults to 5.

        Returns:
            str:
                LLM-generated summary based on the retrieved context.
                Returns "No relevant documents found." if retrieval
                produces no usable text.

        Example:
            >>> summary = rag.search_and_summarize(
            ...     "What is attention mechanism?", top_k=3
            ... )
            >>> print(summary)
        """
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)