# RAG LangChain – Retrieval-Augmented Generation

A Python-based Retrieval-Augmented Generation (RAG) application that enables users to query custom documents using semantic search, FAISS vector storage, and Large Language Models (LLMs) powered by Groq.

The application loads documents from multiple file formats, splits them into smaller chunks, generates embeddings using Sentence Transformers, stores the vectors in FAISS, and retrieves relevant context to generate AI-powered responses.

## 🚀 Features

- **Multi-format Document Loading:** Supports PDF, TXT, CSV, Excel, Word, and JSON files.
- **Document Chunking:** Splits large documents into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
- **Text Embeddings:** Generates dense vector embeddings using Sentence Transformers.
- **Vector Database:** Uses FAISS for efficient similarity-based document retrieval.
- **Persistent Storage:** Saves and loads FAISS indexes and associated metadata.
- **Semantic Search:** Retrieves relevant document chunks based on user queries.
- **LLM Integration:** Uses Groq-hosted LLMs to generate context-aware responses.
- **Environment Configuration:** Uses environment variables to manage API credentials.
- **Dependency Management:** Uses UV for Python package and dependency management.

## 🏗️ Architecture

```text
                User Query
                    |
                    v
             RAG Search Module
                    |
                    v
             Query Embedding
                    |
                    v
              FAISS Search
                    |
                    v
           Retrieve Relevant Chunks
                    |
                    v
             Context Creation
                    |
                    v
                Groq LLM
                    |
                    v
             Generated Answer
```

### Document Ingestion Pipeline

```text
       Source Documents
              |
              v
          Data Loader
              |
              v
      Document Processing
              |
              v
       Text Chunking
              |
              v
     Sentence Transformers
              |
              v
        Vector Embeddings
              |
              v
          FAISS Index
              |
              v
       Persistent Storage
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| UV | Python package and dependency manager |
| LangChain | RAG application framework |
| Sentence Transformers | Text embedding generation |
| FAISS | Vector indexing and similarity search |
| NumPy | Numerical and vector operations |
| Groq | LLM inference |
| python-dotenv | Environment variable management |
| Git | Version control |

## 📁 Project Structure

```text
rag_langchain/
│
├── src/
│   ├── data/
│   │
│   ├── faiss_store/
│   │
│   ├── rag_langchain/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── embedding.py
│   │   ├── search.py
│   │   └── vector_store.py
│   │
│   ├── __init__.py
│   └── app.py
│
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── requirements.txt
├── README.md
└── uv.lock
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ayushikherli123/rag_langchain.git

cd rag_langchain
```

### 2. Install UV

Install UV using the official installation instructions:

https://docs.astral.sh/uv/getting-started/installation/

Verify installation:

```bash
uv --version
```

### 3. Install Python dependencies

Synchronize the project dependencies:

```bash
uv sync
```

This command creates the virtual environment and installs the dependencies specified in `pyproject.toml` and `uv.lock`.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/

**Important:** Never commit your actual `.env` file to GitHub.

### 5. Add source documents

Place your documents inside the `src/data/` directory.

Supported formats:

- PDF (.pdf)
- Text (.txt)
- CSV (.csv)
- Excel (.xlsx)
- Word (.docx)
- JSON (.json)

### 6. Run the application

From the project root:

```bash
uv run python src/app.py
```

## 🔄 RAG Workflow

### 1. Document Loading

The data loader recursively scans the configured data directory and loads supported documents using their respective LangChain document loaders.

### 2. Document Chunking

Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.

Default configuration:

| Parameter | Value |
|---|---|
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Separators | Paragraph, newline, space |

### 3. Embedding Generation

The Sentence Transformer model converts document chunks into dense numerical vectors.

Default model:

`all-MiniLM-L6-v2`

### 4. Vector Storage

FAISS stores the generated embeddings using `IndexFlatL2` for L2 distance-based similarity search.

The vector index and associated metadata are persisted locally.

Storage files:

- `faiss.index`
- `metadata.pkl`

### 5. Semantic Retrieval

When a user submits a query:

1. The query is converted into an embedding.
2. FAISS searches for the nearest vectors.
3. The top-k relevant document chunks are retrieved.
4. Retrieved text is combined into context.

### 6. Response Generation

The retrieved context and user query are passed to the Groq LLM.

The LLM generates a summary based on the retrieved information.

## 💡 Example Usage

Example query:

```python
query = "What is attention mechanism?"

summary = rag_search.search_and_summarize(
    query,
    top_k=3
)

print("Summary:", summary)
```

Example output:

```text
Query: What is attention mechanism?

Summary:
Attention mechanism is a technique used in neural networks
to focus on the most relevant parts of input data while
generating an output.
```

*The output above is illustrative.*

## 🔧 Configuration

The application supports the following configurable parameters:

| Parameter | Default Value | Description |
|---|---|---|
| Embedding Model | all-MiniLM-L6-v2 | Sentence embedding model |
| Chunk Size | 1000 | Maximum chunk size |
| Chunk Overlap | 200 | Overlapping characters |
| Top K | 5 | Number of retrieved chunks |
| Vector Store | FAISS | Vector database |
| LLM | openai/gpt-oss-120b | Groq-hosted model |

## 📦 Dependency Management with UV

Common commands:

```bash
# Add a dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Synchronize dependencies
uv sync

# Run the application
uv run python src/app.py

# View installed dependency tree
uv tree

# Update lockfile
uv lock
```

## 🔐 Security

- API credentials are stored in environment variables.
- `.env` is excluded through `.gitignore`.
- Virtual environments are not committed.
- Generated vector indexes and private source documents are excluded from Git.

## 🚧 Future Enhancements

- [ ] Add support for additional document formats.
- [ ] Implement metadata-based filtering.
- [ ] Add conversational memory.
- [ ] Implement hybrid search (semantic + keyword).
- [ ] Integrate reranking for improved retrieval accuracy.
- [ ] Add streaming LLM responses.
- [ ] Implement document-level source citations.
- [ ] Add FastAPI endpoints.
- [ ] Build a Streamlit user interface.
- [ ] Add unit and integration tests.
- [ ] Implement Docker-based deployment.
- [ ] Add CI/CD using GitHub Actions.

## 👨‍💻 Author

**Ayushi**

GitHub: https://github.com/ayushikherli123

---

⭐ If you find this project useful, consider starring the repository!