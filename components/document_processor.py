from chromadb.types import Collection
import chromadb

class DocumentProcessor:
    """Initialises a Document Processor that handles document loading, splitting, and storage in ChromaDB."""
    def __init__(self, chroma_client=None, collection_name="multilingual_documents"):
        """
        Initializes the DocumentProcessor with a ChromaDB client and collection.
        :param chroma_client: An instance of ChromaDB client (optional, for dependency injection).
        :param collection_name: Name of the ChromaDB collection.
        """
        self.chroma_client = chroma_client or chromadb.Client()  # Use provided client or default
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

    def process_and_store(self, file_path: str, file_name: str):
        """Processes a document, splits it into chunks, and stores it in the ChromaDB collection."""
        print(f"Processing document: {file_name}")
        # Task for Ingestion Engineer: Add logic to load different file types
        # Task: Split documents into chunks
        # Task: Store the chunks (documents, embeddings, metadatas) in self.collection
        pass

    def get_collection(self) -> Collection:
        """Returns the ChromaDB collection."""
        return self.collection