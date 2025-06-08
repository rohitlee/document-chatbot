import requests
import os
import streamlit as st
from .nlp_processor import NLPProcessor

# Try using different model check which gives better results
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1" # This is a Mistral Instruct model it can be changes based on comparison results

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
        """Generates a response using an LLM with the provided context."""
        
    
    def _create_context(self, docs: list, max_length: int = 4000) -> str:
        """Generates a contextual response from the Hugging Face API."""

    def _generate_with_hf_api(self, query: str, context: str) -> str:
        """Generate response using the Hugging Face Inference API with the correct prompt format."""
        