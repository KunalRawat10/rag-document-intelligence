"""
DocuMind RAG - Production Document Retrieval-Augmented Generation Engine
========================================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, SentenceTransformers, NumPy Vector Store, PyPDF

Pipeline: Document -> Chunking -> Vector Embeddings -> Semantic Retrieval -> LLM Reasoning
"""

import os
import time
import json
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="DocuMind RAG | Chat with Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Monochromatic Obsidian & Starfield Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        color: #f4f4f5 !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(1px 1px at 25px 35px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 85px 130px, rgba(255,255,255,0.7) 100%, transparent),
            radial-gradient(1.5px 1.5px at 170px 50px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 260px 210px, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(2px 2px at 340px 280px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 430px 90px, rgba(255,255,255,0.8) 100%, transparent),
            radial-gradient(1.5px 1.5px at 510px 240px, #ffffff 100%, transparent);
        background-size: 550px 550px !important;
    }

    .title-3d {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        text-shadow: 0 1px 0 #52525b, 0 2px 0 #3f3f46, 0 4px 0 #18181b, 0 8px 24px rgba(255, 255, 255, 0.12);
        margin-bottom: 2px;
    }

    .card-3d {
        background: linear-gradient(180deg, #111113 0%, #080809 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid #000000;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }

    .citation-box {
        background: #09090b;
        border: 1px solid #27272a;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0px;
        font-size: 0.84rem;
        color: #d4d4d8;
    }

    .stButton > button {
        background: linear-gradient(180deg, #1f1f23 0%, #121215 100%) !important;
        color: #f4f4f5 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.5) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.8) !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 0 #09090b, 0 6px 14px rgba(0, 0, 0, 0.6) !important;
    }

    section[data-testid="stSidebar"] {
        background: #050507 !important;
        border-right: 1px solid #18181b !important;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #09090b;
        color: #d4d4d8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid #27272a;
        margin-bottom: 16px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px #22c55e;
    }

    .telemetry-chip {
        display: inline-block;
        background: #09090b;
        color: #a1a1aa;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-top: 8px;
        border: 1px solid #27272a;
    }
</style>
""", unsafe_allow_html=True)


# Load Embedding Model (Cached in memory)
@st.cache_resource(show_spinner="Loading dense embedding model (all-MiniLM-L6-v2)...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


embedder = load_embedder()


# Document Ingestion & Chunking Pipeline
def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from uploaded PDF or TXT files."""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += f"\n[Page {i+1}]\n" + extracted
        return text
    else:
        return uploaded_file.read().decode("utf-8")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, str]]:
    """Chunks documents using sliding window overlap with metadata preservation."""
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append({
                "id": chunk_id,
                "text": chunk_str,
                "start_idx": start,
                "end_idx": end
            })
            chunk_id += 1
        start += (chunk_size - chunk_overlap)
    return chunks


def build_vector_store(chunks: List[Dict[str, str]]) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Generates normalized dense vector embeddings for semantic similarity."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, chunks


def retrieve_top_k(query: str, embeddings: np.ndarray, chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict]:
    """Performs cosine semantic search between query embedding and stored document vectors."""
    query_vector = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    similarity_scores = np.dot(embeddings, query_vector)
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": float(similarity_scores[idx])
        })
    return results


# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# Sidebar Controls
with st.sidebar:
    st.markdown('<div class="status-badge"><span class="status-dot"></span> RAG PIPELINE ACTIVE</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at console.groq.com"
    )
    
    model_name = st.selectbox(
        "Inference LLM",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])
    
    with st.expander("🎛️ Chunking & Retrieval Parameters", expanded=False):
        chunk_size = st.slider("Chunk Size (Characters)", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap (Characters)", 0, 300, 100, 20)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 6, 3, 1)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)

    if uploaded_file and (st.session_state.processed_filename != uploaded_file.name):
        with st.spinner("Extracting text and computing vector embeddings..."):
            raw_text = extract_text_from_file(uploaded_file)
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            vectors, chunks_metadata = build_vector_store(chunks)
            
            st.session_state.vector_store = vectors
            st.session_state.doc_chunks = chunks_metadata
            st.session_state.processed_filename = uploaded_file.name
            st.success(f"Indexed {len(chunks)} chunks into Vector DB!")

    st.markdown("---")
    st.markdown("### 📊 Index Telemetry")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card-3d">
            <span style="font-size:0.72rem; color:#71717a; font-weight:600;">INDEXED CHUNKS</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#fafafa;">{len(st.session_state.doc_chunks)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card-3d">
            <span style="font-size:0.72rem; color:#71717a; font-weight:600;">QUERIES</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#fafafa;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Reset Chat & Knowledge Base", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.doc_chunks = []
        st.session_state.processed_filename = ""
        st.session_state.query_count = 0
        st.rerun()


# Main Header
st.markdown('<div class="title-3d">📚 DOCUMIND | RAG ENGINE</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#a1a1aa; font-size:0.95rem; margin-top:-4px; margin-bottom:20px;'>"
    "Eliminate LLM hallucinations by grounding answers directly in your uploaded documents using dense vector embeddings."
    "</p>",
    unsafe_allow_html=True
)

# Display Active Document Banner
if st.session_state.processed_filename:
    st.info(f"📂 Active Knowledge Base: **{st.session_state.processed_filename}** ({len(st.session_state.doc_chunks)} searchable chunks)")
else:
    st.warning("⚠️ No document indexed. Please upload a PDF or TXT file in the sidebar to activate RAG grounding.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("🔍 View Retrieved Context Snippets", expanded=False):
                for c in msg["citations"]:
                    st.markdown(
                        f'<div class="citation-box"><b>[Chunk {c["chunk"]["id"]}] Similarity Score: {c["score"]:.4f}</b><br>{c["chunk"]["text"]}</div>',
                        unsafe_allow_html=True
                    )


# Streaming Text Generator
def stream_text_chunks(raw_stream):
    for chunk in raw_stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                yield delta.content


# Query Submission Pipeline
user_query = st.chat_input("Ask a question grounded in your uploaded document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    effective_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")

    if not effective_key:
        err = "Please enter a valid Groq API key in the sidebar."
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
    elif st.session_state.vector_store is None or len(st.session_state.doc_chunks) == 0:
        err = "Please upload and index a document in the sidebar before asking questions."
        with st.chat_message("assistant"):
            st.warning(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
    else:
        # Step 1: Semantic Retrieval
        retrieval_start = time.time()
        retrieved_results = retrieve_top_k(user_query, st.session_state.vector_store, st.session_state.doc_chunks, top_k)
        retrieval_latency = round(time.time() - retrieval_start, 3)

        # Step 2: Context Augmentation Prompt Engineering
        context_str = "\n\n".join([
            f"--- Context Chunk {r['chunk']['id']} (Relevance Score: {r['score']:.3f}) ---\n{r['chunk']['text']}"
            for r in retrieved_results
        ])

        system_instruction = (
            "You are a strict, grounded Retrieval-Augmented Generation (RAG) assistant. "
            "Your task is to answer the user's question using ONLY the provided document context below.\n"
            "Rules:\n"
            "1. If the provided context does NOT contain enough information to answer, state: 'The provided document does not contain this information.' Do not hallucinate.\n"
            "2. Cite the chunk IDs where relevant.\n"
            "3. Keep answers clear, technical, and concise.\n\n"
            f"Provided Document Context:\n{context_str}"
        )

        # Step 3: LLM Inference
        client = Groq(api_key=effective_key)
        messages_payload = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query}
        ]

        with st.chat_message("assistant"):
            start_time = time.time()
            try:
                raw_stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages_payload,
                    temperature=temperature,
                    stream=True
                )
                
                response_content = st.write_stream(stream_text_chunks(raw_stream))
                inference_latency = round(time.time() - start_time, 2)
                
                st.session_state.query_count += 1

                # Display Citations Expander
                with st.expander("🔍 View Retrieved Context Snippets", expanded=False):
                    for c in retrieved_results:
                        st.markdown(
                            f'<div class="citation-box"><b>[Chunk {c["chunk"]["id"]}] Similarity Score: {c["score"]:.4f}</b><br>{c["chunk"]["text"]}</div>',
                            unsafe_allow_html=True
                        )

                st.markdown(
                    f'<div class="telemetry-chip">⚡ Retrieval: <b>{retrieval_latency}s</b> | '
                    f'Inference: <b>{inference_latency}s</b> | '
                    f'Top-K Chunks: <b>{len(retrieved_results)}</b></div>',
                    unsafe_allow_html=True
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "citations": retrieved_results
                })

            except RateLimitError:
                err = "Rate limit reached on Groq API. Please wait a few moments."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIConnectionError:
                err = "Network connection failed. Verify internet link."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIStatusError as e:
                err = f"API Error ({e.status_code}): {e.message}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
