import streamlit as st
import tempfile
import os
from datetime import datetime
import pandas as pd

from components.document_processor import DocumentProcessor
from components.nlp_processor import NLPProcessor
from components.retrieval_system import DocumentRetriever
from components.response_generator import ResponseGenerator

st.set_page_config(
    page_title="Document AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
 <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message-container {
        display: flex;
        justify-content: flex-end;
    }
    .bot-message-container {
        display: flex;
        justify-content: flex-start;
    }
    .user-message {
        background-color: #1189d1;
        text-align: right;
        color: #f7f7f7;
    }
    .bot-message {
        background-color: transparent;
        color: #f7f7f7;
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
 </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False
    st.session_state.messages = []
    st.session_state.query_count = 0
    st.session_state.confidence_history = []
    st.session_state.processed_files = set()

@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot components (cached for performance)"""
    try:
        doc_processor = DocumentProcessor()
        nlp_processor = NLPProcessor()
        retriever = DocumentRetriever(doc_processor.collection)
        response_generator = ResponseGenerator()
        return doc_processor, nlp_processor, retriever, response_generator, True
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None, None, None, None, False
    
def main():
    st.markdown("""
        <div class="main-header">
            <h1>🤖 Document-Driven AI Chatbot</h1>
            <p>Upload documents and ask questions in multiple languages!</p>
        </div>
    """, unsafe_allow_html=True)

    doc_processor, nlp_processor, retriever, response_generator, init_success = initialize_chatbot()
    if not init_success:
        st.error("Failed to initialize chatbot. Please check API keys and refresh the page.")
        return
    
    with st.sidebar:
        st.header("📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to chat with"
        )
        if uploaded_files:
            process_documents(uploaded_files, doc_processor)
        
        st.divider()

        # language selection dropown
        st.header("🌐 Language Settings")
        language_options = {
            "en-IN": "English",
            "hi-IN": "Hindi",
            "bn-IN": "Bengali",
            "gu-IN": "Gujarati",
            "kn-IN": "Kannada",
            "ml-IN": "Malayalam",
            "mr-IN": "Marathi",
            "od-IN": "Odia",
            "pa-IN": "Punjabi",
            "ta-IN": "Tamil",
            # Add other supported codes from the error message if needed
        }
        selected_language = st.selectbox(
            "Choose Response Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0 # Default to English
        )
        
        st.divider()
        st.header("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.processed_files))
        with col2:
            st.metric("Queries", st.session_state.query_count)
        
        if st.session_state.confidence_history:
            avg_confidence = sum(st.session_state.confidence_history) / len(st.session_state.confidence_history)
            st.metric("Avg Relevance", f"{avg_confidence:.1%}")
        
        st.divider()
        st.header("⚡ Quick Actions")
        if st.button("Clear Chat & History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.confidence_history = []
            st.rerun()

    st.header("💬 Chat Interface")
    chat_container = st.container(height=500)
    with chat_container:
        display_chat_messages()

    # passing selected lang to handler 
    handle_chat_input(nlp_processor, retriever, response_generator, selected_language)

def process_documents(uploaded_files, doc_processor):
    """Process uploaded documents, avoiding duplicates."""
    with st.spinner("Processing documents... This may take a moment."):
        new_files_processed = 0
        total_chunks = 0
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    documents = doc_processor.process_document(temp_file_path)
                    doc_processor.store_documents(documents)
                    total_chunks += len(documents)
                    st.session_state.processed_files.add(uploaded_file.name)
                    new_files_processed += 1
                    os.unlink(temp_file_path)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if new_files_processed > 0:
            st.success(f"Successfully processed {new_files_processed} new document(s) into {total_chunks} chunks.")
            st.rerun()

def display_chat_messages():
    """Display chat message history with improved styling."""
    if not st.session_state.messages:
        st.info("👋 Upload a document and start asking questions!")
        return

    for message in st.session_state.messages:
        is_user = message["role"] == "user"
        container_class = "user-message-container" if is_user else "bot-message-container"
        message_class = "user-message" if is_user else "bot-message"
        
        with st.container():
            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            if is_user:
                 st.markdown(f"""
                    <div class="chat-message {message_class}">
                            <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                confidence_bar = ""
                if "confidence" in message:
                    confidence = message["confidence"] * 100
                
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <strong>🤖 Assistant:</strong> {message['content']}
                        {confidence_bar}
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def handle_chat_input(nlp_processor, retriever, response_generator, language: str):
    """Handle chat input and generate responses."""
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Pass the selected language down to the response generation pipeline
                response_data = generate_chatbot_response(user_input, nlp_processor, retriever, response_generator, language)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "confidence": response_data["confidence"]
                })
                st.session_state.query_count += 1
                st.session_state.confidence_history.append(response_data["confidence"])
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error. Please try again.",
                    "confidence": 0.0
                })
        st.rerun()

# Response logic for Multilingual flow
def generate_chatbot_response(query: str, nlp_processor, retriever, response_generator, language: str):
    """Orchestrate the full RAG pipeline for a multilingual response."""
    
    # 1. Translate user's query to English for searching.
    # The source language is auto-detected. The target is our consistent pivot language, 'en-IN'.
    english_query = nlp_processor.translate_text(query, source_lang="auto", target_lang='en-IN')

    if not english_query or not english_query.strip():
        return {"response": "I could not understand your question. Please try rephrasing.", "confidence": 0.0}

    # 2. Retrieve relevant documents using the English query
    retrieved_docs = retriever.hybrid_search(english_query, k=5)

    # 3. Handle the case where no relevant documents are found
    if not retrieved_docs:
        not_found_message = "I couldn't find relevant information in your documents to answer that. Please try rephrasing your question."
        # Translate the "not found" message back to the user's CHOSEN language
        translated_not_found = nlp_processor.translate_text(not_found_message, source_lang='en-IN', target_lang=language)
        return {"response": translated_not_found, "confidence": 0.0}
    
    # 4. Generate a response using the LLM. Pass the user's chosen language for the final translation step.
    response = response_generator.generate_response(
        english_query, retrieved_docs, nlp_processor, target_language=language
    )

    # 5. Calculate confidence based on relevance of retrieved docs
    confidence = sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
    
    return {"response": response, "confidence": confidence}

if __name__ == "__main__":
    main()