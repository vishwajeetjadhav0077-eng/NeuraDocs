# 🧠 NeuraDocs

### Intelligent Document AI powered by Retrieval-Augmented Generation

NeuraDocs is an AI-powered document intelligence application that uses Retrieval-Augmented Generation (RAG) to allow users to upload documents, retrieve relevant information using semantic search, and generate context-aware answers using a local Large Language Model.

##  Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate semantic embeddings
- Store embeddings using FAISS
- Retrieve relevant document chunks
- Generate answers using Llama 3.2
- Local LLM inference using Ollama
- Streamlit user interface
- Runs locally without requiring paid OpenAI API credits

##  Architecture

PDF
↓
PyPDFLoader
↓
Text Chunking
↓
Sentence Transformers
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
Prompt + Retrieved Context
↓
Llama 3.2
↓
Answer

##  Technologies

- Python
- LangChain
- Ollama
- Llama 3.2
- Sentence Transformers
- FAISS
- PyPDF
- Streamlit

##  Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd RAG_PDF_QA