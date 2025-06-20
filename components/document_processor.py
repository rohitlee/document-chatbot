import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from sentence_transformers import SentenceTransformer
import chromadb

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("multilingual_documents")

    def process_document(self, file_path: str) -> List[Dict]:
        """Load and process documents based on file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        documents = []

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

            text_to_embed = f"passage: {chunk.page_content}"

            doc_data = {
                'id': f"{os.path.basename(file_path)}_{i}",
                'content': chunk.page_content,
                'metadata': metadata,
                'embedding': self.embedding_model.encode(text_to_embed)
            }
            documents.append(doc_data)

        return documents
    
    def store_documents(self, documents: List[Dict]):
        """Store documents in vector database."""
        if not documents:
            return
            
        self.collection.add(
            ids=[doc['id'] for doc in documents],
            embeddings=[doc['embedding'].tolist() for doc in documents],
            documents=[doc['content'] for doc in documents],
            metadatas=[doc['metadata'] for doc in documents]
        )