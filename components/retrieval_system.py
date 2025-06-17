from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on documents."""
        query_embedding=self.embedding_model.encode(query)
        results=self.collection.query(
            query_embedding=[query_embedding.tolist()],
            n_results=k
            )
        return self._format_results(results)

    
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
        if not results['documents'] or not results['documents'][0]:
            return []
        
        docs=results['documents'][0]
        metadatas=results['metadatas'][0]
        distances=results['distances'][0]

        for i in range(len(docs)):
            formatted.append({
                'content': docs[i],
                #score is1-disatnce(cosine distance)
                'score': 1 -distances[i] if distances else 0,
                'metadata': metadatas[i] if metadatas else {}
            })
    
        return formatted