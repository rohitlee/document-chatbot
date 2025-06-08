import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import chromadb
from chromadb.types import Collection

class DocumentProcessor:
    """Handles loading, processing, and storing documents."""
    def __init__(self):
        print("Initializing DocumentProcessor...")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.chroma_client = chromadb.Client() # Uses a transient in-memory DB by default
        self.collection = self.chroma_client.get_or_create_collection("multilingual_documents")

    def process_and_store(self, file_path: str, file_name: str):
        """Loads a document, splits it into chunks, and stores it."""
        print(f"Processing document: {file_name}")
        # Task for Ingestion Engineer: Add logic to load different file types
        # Task: Split documents into chunks
        # Task: Store the chunks (documents, embeddings, metadatas) in self.collection
        pass

    def get_collection(self) -> Collection:
        """Returns the ChromaDB collection object."""
        return self.collection