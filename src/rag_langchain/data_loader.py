"""
Document Loader Module
----------------------

This module provides functionality to load multiple document formats
from a specified directory using LangChain document loaders.

Supported file formats:
    - PDF (.pdf)
    - Text (.txt)
    - CSV (.csv)
    - Excel (.xlsx)
    - Word (.docx)
    - JSON (.json)

The module uses a file-type-to-loader mapping to dynamically select
the appropriate LangChain loader for each file type.

Main functionality:
    load_all_documents(data_dir):
        Recursively scans the given directory, loads all supported
        documents, and returns a combined list of LangChain Document
        objects.

Features:
    - Recursive directory scanning.
    - Generic loader selection based on file extension.
    - Individual file-loading exception handling.
    - Consolidated document collection.
"""
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    file_type_to_loader_mapping = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'csv': CSVLoader,
        'xlsx': UnstructuredExcelLoader,
        'docx': Docx2txtLoader,
        'json': JSONLoader
    }

    for type, loader_class in file_type_to_loader_mapping.items():
        # files
        files = list(data_path.glob(f"**/*.{type}"))
        print(f"[DEBUG] Found {len(files)} {type} files: {[str(f) for f in files]}")
        for file in files:
            print(f"[DEBUG] Loading PDF: {file}")
            try:
                loader = loader_class(str(file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)