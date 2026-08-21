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


🚀 Key Features
📄 Multi-Format Ingestion: Extracts text from multi-page PDFs and text files with page metadata preservation.

✂️ Configurable Sliding-Window Chunking: Sliders to tune chunk size and overlap dynamically.

🧠 Dense Semantic Vector Embeddings: Uses all-MiniLM-L6-v2 for 384-dimensional vector embeddings and normalized cosine search.

🛡️ Zero-Hallucination Prompt Architecture: Constrains LLMs to answer strictly using retrieved context chunks with fallbacks.

🔍 Transparent Source Citations: Inspect the exact chunks and similarity confidence scores used to answer each question.

⚙️ Quick Start
Bash
git clone [https://github.com/KunalRawat10/rag-document-chat.git](https://github.com/KunalRawat10/rag-document-chat.git)
cd rag-document-chat
pip install -r requirements.txt
streamlit run app.py
👨‍💻 Author
Kunal Rawat

GitHub: @KunalRawat10


---

**Phase 3: Deploy on Streamlit Community Cloud (Free)**

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **Create app**.
3. Select your repository: `KunalRawat10/rag-document-chat`, branch: `main`, file: `app.py`.
4. Click **Deploy!**

---

**Phase 4: LinkedIn Post Caption**

Publish the following caption on LinkedIn and attach a screenshot of the app answering a question from an uploaded document:

```text
🚀 Module 4 Completed: Engineering a Production Document RAG (Retrieval-Augmented Generation) Engine!

Excited to share the completion of Module 4: "RAG, Embeddings & Vector Databases" as part of my Generative AI Internship at @Codomax Digital Solutions!

I built and deployed DocuMind RAG — a full-stack document intelligence application engineered to eliminate LLM hallucinations by grounding responses in private PDFs and text files.

Key Technical Implementations:
• Ingestion & Sliding-Window Chunking: Built customizable text splitting with sliding overlap to preserve contextual continuity across document boundaries.
• Dense Vector Embeddings: Integrated Hugging Face's `all-MiniLM-L6-v2` to map document chunks into 384-dimensional dense semantic vector space.
• Vector Search & Cosine Indexing: Implemented normalized vector similarity retrieval to rank and retrieve top-k context passages in milliseconds.
• Hallucination Mitigation Prompting: Designed strict boundary-enforced system prompts that cite source chunks and prevent ungrounded assertions.
• Transparent Citations & Telemetry: Real-time inspection of source chunk similarity scores, retrieval latency, and LLM generation time.

💻 GitHub Repository: https://github.com/KunalRawat10/rag-document-chat

Ready for the final stretch! 🚀

#GenerativeAI #RAG #VectorDatabase #Embeddings #Python #Streamlit #Groq #OpenSource #Fu
