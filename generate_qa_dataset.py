import os
import json
import time
from groq import Groq
from tqdm import tqdm
import re
import toml

# --- Configuration ---
CHUNKS_INPUT_FILE = "data/document_chunks.json"
QA_OUTPUT_FILE = "data/evaluation_dataset_groq.json"
SECRETS_FILE_PATH = ".streamlit/secrets.toml"

# --- Groq Configuration ---
MODEL_ID = "llama-3.1-8b-instant"

def load_api_key(secret_key: str):
    try:
        with open(SECRETS_FILE_PATH, "r") as f:
            secrets = toml.load(f)
        return secrets.get(secret_key)
    except FileNotFoundError:
        print(f"Error: Secrets file not found at '{SECRETS_FILE_PATH}'")
        return None
    except Exception as e:
        print(f"Error loading secrets file: {e}")
        return None

# --- Load API Key and Initialize Client ---
GROQ_API_KEY = load_api_key("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Groq API key not found in .streamlit/secrets.toml.")
    print("Please add 'GROQ_API_KEY = \"gsk_YourKeyHere\"' to the file.")
    exit()

client = Groq(api_key=GROQ_API_KEY)

# --- The rest of the script remains the same ---
SYSTEM_PROMPT_TEMPLATE = """
You are an expert data generator for evaluating a question-answering system.
Your task is to generate one high-quality, relevant question and a concise answer based *only* on the provided text chunk.

Rules:
1.  Generate exactly ONE question-answer pair.
2.  The question must be answerable *solely* from the information within the provided text. Do not use any external knowledge.
3.  The answer must be a direct and concise summary of the information in the text that answers the question.
4.  The question and answer should be in the SAME language as the provided text chunk.
5.  Respond ONLY with a valid JSON object containing the keys "question" and "answer". Do not add any other text, explanations, or markdown.

Text Chunk:
"{chunk_content}"

Your JSON Output:
"""

def robust_json_parser(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

def generate_qna_for_chunk(chunk_content: str, retries=3, delay=5):
    prompt_content = SYSTEM_PROMPT_TEMPLATE.format(chunk_content=chunk_content)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.5,
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            qna_json = robust_json_parser(result_text)
            if qna_json:
                return qna_json
        except Exception as e:
            print(f"  - Groq API Error on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(delay)
    return None

def create_evaluation_dataset():
    try:
        with open(CHUNKS_INPUT_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Chunks file not found at '{CHUNKS_INPUT_FILE}'.")
        print("Please run `create_chunks.py` first to generate it.")
        return

    evaluation_dataset = []
    print(f"Starting Q&A generation for {len(chunks)} chunks using model: {MODEL_ID} on Groq")
    for chunk in tqdm(chunks, desc="Generating Q&A"):
        qna_pair = generate_qna_for_chunk(chunk['chunk_content'])
        if qna_pair and 'question' in qna_pair and 'answer' in qna_pair:
            evaluation_dataset.append({
                "chunk_id": chunk['chunk_id'],
                "source_document": chunk['source_document'],
                "question": qna_pair['question'],
                "ground_truth_answer": qna_pair['answer'],
                "context": chunk['chunk_content']
            })
        else:
            print(f"  - Failed to generate valid Q&A for chunk_id: {chunk['chunk_id']}")

    with open(QA_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_dataset, f, ensure_ascii=False, indent=4)

    print(f"\nDataset generation complete! Saved to: {QA_OUTPUT_FILE}")
    print(f"Total Q&A pairs created: {len(evaluation_dataset)}")

if __name__ == "__main__":
    create_evaluation_dataset()