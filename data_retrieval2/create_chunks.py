import os
import json
from components.document_processor import DocumentProcessor

# Directory containing your multilingual documents
DOCS_DIR = "evaluation_docs/" 
# Output file to save the chunks
CHUNKS_OUTPUT_FILE = "data/document_chunks.json"

def process_all_documents():
    """
    Processes all documents in a directory and saves their chunks and IDs to a file.
    """
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Directory '{DOCS_DIR}' not found. Please create it and add your documents.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(CHUNKS_OUTPUT_FILE), exist_ok=True)

    doc_processor = DocumentProcessor()
    all_chunks = []
    
    # List all supported files in the directory
    supported_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.docx', '.txt'))]

    if not supported_files:
        print(f"No supported documents found in '{DOCS_DIR}'.")
        return

    print(f"Found {len(supported_files)} documents to process...")

    for filename in supported_files:
        file_path = os.path.join(DOCS_DIR, filename)
        print(f"  - Processing {filename}...")
        try:
            # The process_document method returns a list of dictionaries for each chunk
            chunks = doc_processor.process_document(file_path)
            
            # We only need the ID and content for dataset generation
            for chunk_data in chunks:
                all_chunks.append({
                    "chunk_id": chunk_data['id'],
                    "chunk_content": chunk_data['content'],
                    "source_document": os.path.basename(file_path)
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the consolidated chunks to a JSON file
    with open(CHUNKS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"\nSuccessfully processed all documents.")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"All chunks and their IDs have been saved to: {CHUNKS_OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_documents()