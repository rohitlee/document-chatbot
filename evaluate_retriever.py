# evaluate_retriever.py
import os
import json
import pandas as pd
from tqdm import tqdm
import chromadb

# Import the components from your project
from components.retrieval_system import DocumentRetriever
from components.nlp_processor import NLPProcessor
from components.document_processor import DocumentProcessor # Import the DocumentProcessor

# --- Configuration ---
# Make sure this points to the correct dataset file
EVALUATION_DATASET_FILE = "data/evaluation_dataset.json" 
# Directory containing the documents used to create the dataset
DOCS_DIR = "evaluation_docs/"
# The number of top results to retrieve for each query
K_VALUES = [1, 3, 5, 10] 

class RetrievalEvaluator:
    def __init__(self, dataset_path, docs_path):
        """
        Initializes the evaluator and builds a dedicated, in-memory vector database
        for the evaluation run.
        """
        print("Initializing self-contained evaluation environment...")
        
        # --- Build an independent RAG system for this evaluation run ---
        # 1. Initialize the document processor
        self.doc_processor = DocumentProcessor()
        
        # 2. Process and store the evaluation documents
        print(f"Processing evaluation documents from: '{docs_path}'")
        self.setup_database(docs_path)
        
        # 3. Initialize the retriever with the newly created collection
        self.retriever = DocumentRetriever(self.doc_processor.collection)
        
        # 4. Initialize the NLP processor for query translation
        self.nlp_processor = NLPProcessor()
        
        # Load the evaluation dataset
        self.dataset = self.load_dataset(dataset_path)
        print(f"Loaded {len(self.dataset)} Q&A pairs for evaluation.")

    def setup_database(self, path):
        """Finds all documents, processes them, and stores them in the in-memory DB."""
        supported_files = [f for f in os.listdir(path) if f.endswith(('.pdf', '.docx', '.txt'))]
        if not supported_files:
            print(f"FATAL: No documents found in evaluation directory '{path}'.")
            exit()
            
        print(f"Found {len(supported_files)} documents to process for evaluation.")
        for filename in tqdm(supported_files, desc="Processing Docs"):
            file_path = os.path.join(path, filename)
            try:
                documents = self.doc_processor.process_document(file_path)
                if documents:
                    self.doc_processor.store_documents(documents)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        print("In-memory vector database is ready for evaluation.")

    def load_dataset(self, path):
        """Loads the JSON evaluation dataset."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"FATAL: Evaluation dataset not found at '{path}'.")
            print("Please ensure you have created the dataset.")
            exit()

    def run_evaluation(self):
        """
        Runs the full evaluation process across the dataset and prints the results.
        """
        results = []
        max_k = max(K_VALUES)

        print(f"\nRunning evaluation for Top K = {max_k}...")
        for item in tqdm(self.dataset, desc="Evaluating Queries"):
            question = item['question']
            ground_truth_id = item['chunk_id']
            
            # Initialize variables for this loop iteration to ensure they always exist
            retrieved_ids = []
            hit = False
            rank = 0

            try:
                # 1. Translate the query to English
                english_query = self.nlp_processor.translate_text(question, target_lang='en-IN', source_lang='auto')
                
                # If translation fails or returns an empty string, skip this item
                if not english_query or not english_query.strip():
                    print(f"\nWarning: Translation failed for question, skipping: '{question[:50]}...'")
                    continue

                # 2. Perform the hybrid search
                retrieved_docs = self.retriever.hybrid_search(english_query, k=max_k)

                # 3. Process the results if any documents were retrieved
                if retrieved_docs:
                    retrieved_ids = [doc['id'] for doc in retrieved_docs]
                    if ground_truth_id in retrieved_ids:
                        hit = True
                        rank = retrieved_ids.index(ground_truth_id) + 1
            
            except Exception as e:
                print(f"\nAn unexpected error occurred for question '{question[:50]}...': {e}")
                # The variables will keep their default values (empty list, False, 0)
                # and the loop will continue to the next item.

            # 4. Append the results for this iteration
            results.append({
                'question': question,
                'ground_truth_id': ground_truth_id,
                'retrieved_ids': retrieved_ids,
                'hit': hit,
                'rank': rank
            })
            
        # --- Calculate and Display Metrics ---
        self.calculate_and_print_metrics(results)

    def calculate_and_print_metrics(self, results):
        """Calculates and prints key retrieval metrics."""
        df = pd.DataFrame(results)
        total_queries = len(df)
        
        print("\n--- Retrieval Evaluation Results ---")
        for k in K_VALUES:
            hits_at_k = df[df['rank'].between(1, k, inclusive='both')].shape[0]
            hit_rate_at_k = hits_at_k / total_queries
            print(f"Hit Rate @{k}: {hit_rate_at_k:.2%}")
        
        print("-" * 35)
        df['reciprocal_rank'] = 1 / df['rank'].replace(0, float('inf'))
        mrr = df['reciprocal_rank'].mean()
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print("-" * 35)
        
        print("Precision@k is equivalent to Hit Rate@k in this single-answer-per-question scenario.")
        print("------------------------------------\n")

        df.to_csv("retrieval_evaluation_details.csv", index=False, encoding='utf-8-sig')
        print("Detailed results saved to 'retrieval_evaluation_details.csv'")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator(dataset_path=EVALUATION_DATASET_FILE, docs_path=DOCS_DIR)
    evaluator.run_evaluation()