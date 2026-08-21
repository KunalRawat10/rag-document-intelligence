# 📚 DocuMind RAG — Document Retrieval-Augmented Generation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Groq API](https://img.shields.io/badge/LLM-Groq%20Cloud-green.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-grade Retrieval-Augmented Generation (RAG) system engineered to eliminate LLM hallucinations by grounding responses in private documents (PDF & TXT).

---

## 📌 Architecture & Pipeline

```text
Uploaded Document (PDF / TXT)
        │
        ▼
Sliding Window Chunking (Size: 500, Overlap: 100)
        │
        ▼
Dense Vector Embedding (`all-MiniLM-L6-v2`)
        │
        ▼
In-Memory Vector Store & Cosine Indexing
        │
        ▼ (User Query)
Top-K Semantic Vector Search
        │
        ▼
Context-Augmented Prompt Construction
        │
        ▼
Groq Accelerated LLM Inference (Streaming Output + Citations)
