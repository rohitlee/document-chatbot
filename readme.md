# Introduction

A Multilingual Intelligent Document Question-Answering System.
This project presents a multilingual document processing system that enables users to upload documents in various languages and interact with their content through natural language queries. The system leverages Retrieval-Augmented Generation (RAG) architecture combined with multilingual language models to provide accurate, contextually relevant answers extracted from user-provided documents, regardless of the source language.
This solution addresses the growing need for efficient multilingual document analysis in global academic, professional, and research contexts. By combining cross-lingual NLP techniques with user-friendly interfaces, the system democratizes access to complex multilingual document analysis, enabling users to quickly extract insights from diverse document collections without manual translation or language-specific searching.

## 🔗 Live Demo
[Click here to try the app](https://document-chatbot-iisc.streamlit.app)


## ✨ Features

-   **Multi-Lingual Support:** Upload documents and ask questions in English, Hindi, Tamil, and more.
-   **Multiple File Formats:** Supports PDF (`.pdf`), Microsoft Word (`.docx`), and Text (`.txt`) files.
-   **AI-Powered Responses:** Uses state-of-the-art open-source models from Hugging Face for question-answering.
-   **Cloud Translation:** Leverages Sarvam AI for fast and accurate language detection and translation.
-   **Local Vector Storage:** Uses ChromaDB to store document embeddings locally for privacy and speed.
-   **Interactive UI:** A clean and modern user interface built with Streamlit.

## 🛠️ Tech Stack

-   **Framework:** Streamlit
-   **LLM (via API):** Hugging Face Inference API (Mixtral / Llama 3)
-   **Translation:** Sarvam AI SDK
-   **Embeddings:** `sentence-transformers` (multilingual models)
-   **Vector Database:** ChromaDB
-   **Language:** Python
---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.


### 1. Prerequisites

-   Python 3.9+
-   An account on [Hugging Face](https://huggingface.co/)
-   An account and API key from [Sarvam AI](https://www.sarvam.ai/)

### 2. Installation

Clone the repository to your local machine:
```
git clone https://github.com/rohitlee/document-chat-ai.git
cd document-chatbot
```

Install all the dependencies mentioned in requirements.txt

```
pip install -r requirements.txt
```

# Team members

- Rohit Singh
- Santosh Grampurohit
- Keshav Kumar
- SK Mohammad Arif
- Sourajit Bhar
- Chandan Kumar Singh
