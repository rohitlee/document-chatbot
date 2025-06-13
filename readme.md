# Introduction

DocuQuery: A Multilingual Intelligent Document Question-Answering System
This project presents a multilingual document processing system that enables users to upload documents in various languages and interact with their content through natural language queries. The system leverages Retrieval-Augmented Generation (RAG) architecture combined with multilingual language models to provide accurate, contextually relevant answers extracted from user-provided documents, regardless of the source language.
This solution addresses the growing need for efficient multilingual document analysis in global academic, professional, and research contexts. By combining cross-lingual NLP techniques with user-friendly interfaces, the system democratizes access to complex multilingual document analysis, enabling users to quickly extract insights from diverse document collections without manual translation or language-specific searching.

# Getting started

Install all the dependencies mentioned in requirements.txt

```
pip install -r requirements.txt
```

# Walkthrough

## document_processor.py

This file contains the code related to processing of documents into vector db, and methods to do the semantic search using the query provided by the user

## References
- [Getting started with CromaDB](https://docs.trychroma.com/docs/overview/getting-started)
- 

## Running UTs
```
python -m unittest tests\components\test_document_processor.py
```

# Team members

- Rohit Singh
- Santosh Grampurohit
- Keshav Kumar
- SK Mohammad Arif
- Sourajit Bhar
- Chandan Kumar Singh