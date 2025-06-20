import requests
import os
import streamlit as st
from .nlp_processor import NLPProcessor

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

class ResponseGenerator:
    def __init__(self):
        try:
            hf_token = st.secrets["HF_TOKEN"]
            self.headers = {"Authorization": f"Bearer {hf_token}"}
        except (FileNotFoundError, KeyError):
            self.headers = {}
            st.error("Hugging Face token not found. Please add HF_TOKEN to your secrets.")
            st.stop()
    
    def generate_response(self, query: str, retrieved_docs: list, nlp_processor: NLPProcessor, target_language: str = 'en-IN') -> str:
        context = self._create_context(retrieved_docs)
        
        if not self.headers.get("Authorization"):
            return "Cannot generate response because Hugging Face API token is missing."

        english_response = self._generate_with_hf_api(query, context)

        if target_language != 'en-IN' and english_response:
            return nlp_processor.translate_text(english_response, source_lang='en-IN', target_lang=target_language)
        
        return english_response or "I could not generate a response based on the provided documents."
    
    def _create_context(self, docs: list, max_length: int = 4000) -> str:
        context_parts = []
        current_length = 0
        for doc in docs:
            content = doc.get('content', '')
            if current_length + len(content) > max_length:
                break
            source = doc.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"Source: {os.path.basename(source)}\nContent: {content}\n---")
            current_length += len(content)
        return "\n".join(context_parts)

    def _generate_with_hf_api(self, query: str, context: str) -> str:
        """Generate response using the Hugging Face Inference API with the correct prompt format."""
        
        system_prompt = "You are a helpful AI assistant. Answer the user's question based *only* on the provided context. If the context does not contain the answer, state that you could not find the information in the documents. Be concise."
        user_prompt = f"""CONTEXT:
        {context}

        QUESTION: {query}"""

        prompt = f"<s>[INST] {system_prompt} \n\n{user_prompt} [/INST]"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 350,
                "temperature": 0.3,
                "return_full_text": False,
            }
        }
        
        try:
            response = requests.post(API_URL, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'].strip()
            elif response.status_code == 503:
                st.toast("Model is loading, please wait a moment and try again...", icon="⏳")
                return "The AI model is currently loading. This can take up to a minute. Please ask your question again shortly."
            else:
                error_message = f"Hugging Face API Error: {response.status_code} - {response.text}"
                print(error_message) # This will print the exact error to your terminal
                return "I encountered an error while trying to reach the AI model. Please check the terminal logs."

        except requests.exceptions.RequestException as e:
            print(f"Error calling Hugging Face API: {e}")
            return "I could not connect to the Hugging Face Inference API. Please check your internet connection."