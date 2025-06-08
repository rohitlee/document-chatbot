from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2') # use same model used for embedding

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on documents."""
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Performs a similarity search. A true hybrid would require a different backend
        like Weaviate or a more complex Chroma setup. For now, semantic search is powerful enough
        with good multilingual models.
        """
        # For simplicity and power, we rely on the strong multilingual semantic search
        return self.similarity_search(query, k)

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into a list of dictionaries."""
        formatted = []
        return formatted