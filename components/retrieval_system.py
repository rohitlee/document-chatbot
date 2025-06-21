from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class DocumentRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on documents."""
        query_to_embed = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_to_embed)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        return self._format_results(results)
    
    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        all_docs = self.collection.get(include=["metadatas", "documents"])

        keywords = query.lower().split()
        if not keywords:
            return []
        
        matched_docs = []
        if all_docs and all_docs.get('ids'):
            for i, doc_content in enumerate(all_docs['documents']):
                score = 0
                content_lower = doc_content.lower()
                # Simple scoring: count how many unique keywords appear in the document
                for keyword in set(keywords):
                    if re.search(r'\b' + re.escape(keyword) + r'\b', content_lower):
                        score += 1
                if score > 0:
                    matched_docs.append({
                        'id': all_docs['ids'][i],
                        'content': doc_content,
                        'score': score, # The score is the number of keyword hits
                        'metadata': all_docs['metadatas'][i]
                    })

        # Sort by score (higher is better) and return top k
        matched_docs.sort(key=lambda x: x['score'], reverse=True)
        return matched_docs[:k]

    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Performs a robust hybrid search using Reciprocal Rank Fusion (RRF)
        to combine semantic and keyword search results.
        """
        # 1. Fetch results from both search methods
        semantic_results = self.similarity_search(query, k=20)
        keyword_results = self.keyword_search(query, k=20)

        # 2. Fuse the results using RRF
        fused_scores = self._reciprocal_rank_fusion([semantic_results, keyword_results])

        if not fused_scores:
            return []
        
        # 3. Create a final sorted list of results based on the fused scores
        all_retrieved_ids = list(fused_scores.keys())
        retrieved_data = self.collection.get(ids=all_retrieved_ids, include=["documents", "metadatas"])

        # Create a dictionary for quick lookup
        docs_by_id = {retrieved_data['ids'][i]: {'content': retrieved_data['documents'][i], 'metadata': retrieved_data['metadatas'][i]} for i in range(len(retrieved_data['ids']))}
        final_results = []
        for doc_id, score in fused_scores.items():
            if doc_id in docs_by_id:
                final_results.append({
                    'id': doc_id,
                    'content': docs_by_id[doc_id]['content'],
                    'metadata': docs_by_id[doc_id]['metadata'],
                    'score': score # Use the raw RRF score for ranking
                })

        # Sort the final list by the RRF score in descending order
        final_results.sort(key=lambda x: x['score'], reverse=True)  

        # 4. Normalize the scores for the top k results
        top_k_results = final_results[:k]
        scores_for_norm = [doc['score'] for doc in top_k_results]   

        if not scores_for_norm:
            return []

        min_score, max_score = min(scores_for_norm), max(scores_for_norm)

        for doc in top_k_results:
            normalized_score = 0.0
            if max_score > min_score:
                normalized_score = (doc['score'] - min_score) / (max_score - min_score)
            elif max_score > 0:
                normalized_score = 1.0
            doc['score'] = normalized_score # Replace raw RRF score with normalized score  

        return top_k_results 
    
    def _reciprocal_rank_fusion(self, result_sets: List[List[Dict]], rrf_k: int = 60) -> Dict[str, float]:
        """
        Combines multiple search result sets using the RRF formula.
        The rrf_k parameter is a constant used to diminish the impact of lower-ranked documents.
        """
        fused_scores = {}
        # Iterate through each list of search results (e.g., semantic, keyword)
        for results in result_sets:
            # Iterate through each document in the result list
            for rank, doc in enumerate(results):
                doc_id = doc.get('id')
                if doc_id:
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = 0
                    # Add the reciprocal rank score to the document's fused score
                    fused_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
        
        return fused_scores
        

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into a list of dictionaries."""
        formatted = []
        if not results['documents'] or not results['documents'][0]:
            return []

        ids = results['ids'][0]
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i in range(len(docs)):
            formatted.append({
                'id': ids[i],
                'content': docs[i],
                # Score is 1 - distance (cosine distance)
                'score': max(0, 1 - distances[i]) if distances else 0, 
                'metadata': metadatas[i] if metadatas else {}
            })
        return formatted