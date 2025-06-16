import streamlit as st
from sarvamai import SarvamAI

class NLPProcessor:
    def __init__(self):
        """Initializes the NLP Processor using the official SarvamAI SDK."""
        try:
            self.client = SarvamAI(api_subscription_key=st.secrets["SARVAM_API_KEY"])
        except Exception as e:
            self.client = None
            st.error(f"Failed to initialize Sarvam AI client: {e}")

    def translate_text(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """
        Translate text using the official Sarvam AI SDK.
        """
        if not self.client or not text or not text.strip():
            return text
        if source_lang == target_lang and source_lang != "auto":
            return text

        try:
            print(f"--- Calling Sarvam Translate SDK ---")
            print(f"Input: '{text[:50]}...', Source: {source_lang}, Target: {target_lang}")

            response = self.client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
            )
            
            print(f"Sarvam Translate SDK Response: {response}")
            
            return response.translated_text

        except Exception as e:
            print(f"ERROR in Sarvam SDK for translation: {e}")
            return text
