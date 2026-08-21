"""
DocuMind RAG Studio - Quantum Neural Document Intelligence
==========================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, SentenceTransformers, NumPy, PyPDF
Theme: Quantum Cyber-Slate & Neon HUD
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
    page_title="DocuMind RAG | Neural Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Quantum Cyber-Slate & Neon HUD Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #e2e8f0 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #03040a !important;
        background-image: 
            radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(14, 165, 233, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(139, 92, 246, 0.03) 0%, transparent 60%),
            linear-gradient(to right, rgba(14, 165, 233, 0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(14, 165, 233, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 50px 50px, 50px 50px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 28, 0.92) 0%, rgba(6, 8, 18, 0.95) 100%) !important;
        backdrop-filter: blur(24px) saturate(1.4) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.12) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(255, 255, 255, 0.03) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    .cyber-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 30%, #38bdf8 60%, #818cf8 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 6s linear infinite;
        margin-bottom: 4px;
        line-height: 1.1;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .cyber-sub {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 28px;
        letter-spacing: 0.01em;
    }

    .hud-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(10, 15, 30, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.06),
            0 0 0 1px rgba(255, 255, 255, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .hud-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
    }

    .hud-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 
            0 12px 40px rgba(99, 102, 241, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 0 0 1px rgba(255, 255, 255, 0.03);
        transform: translateY(-1px);
    }

    .citation-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.6) 0%, rgba(10, 15, 30, 0.75) 100%);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-left: 3px solid #0ea5e9;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0px;
        font-size: 0.85rem;
        color: #cbd5e1;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.25s ease;
    }

    .citation-card:hover {
        border-left-color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.25);
        box-shadow: 0 6px 24px rgba(14, 165, 233, 0.1);
    }

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        color: #67e8f9;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(14, 165, 233, 0.12) 100%) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.08) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28) 0%, rgba(14, 165, 233, 0.28) 100%) !important;
        border-color: rgba(165, 180, 252, 0.5) !important;
        color: #ffffff !important;
        box-shadow: 
            0 0 24px rgba(99, 102, 241, 0.25),
            0 0 48px rgba(14, 165, 233, 0.1) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    .telemetry-chip {
        display: inline-block;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%);
        color: #64748b;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 12px;
        border: 1px solid rgba(99, 102, 241, 0.12);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        letter-spacing: 0.01em;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        color: #22d3ee;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(34, 211, 238, 0.25);
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.1);
        letter-spacing: 0.04em;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #22d3ee;
        border-radius: 50%;
        box-shadow: 0 0 8px #22d3ee, 0 0 16px rgba(34, 211, 238, 0.4);
        animation: pulse-dot 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.85); }
    }

    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.12) !important;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(10, 15, 30, 0.8) 100%) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }

    .stAlert [data-testid="stAlertContent"] {
        color: #cbd5e1 !important;
    }

    .stFileUploader > div > div {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.5) 0%, rgba(10, 15, 30, 0.6) 100%) !important;
        border: 2px dashed rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
    }

    .stFileUploader > div > div:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.6) 0%, rgba(10, 15, 30, 0.7) 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


# Load Dense Embedding Model
@st.cache_resource(show_spinner="Initializing all-MiniLM-L6-v2 dense vector embedder...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


embedder = load_embedder()


# Document Processing & Vector Search Helpers
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
    """Chunks documents using a sliding window overlap."""
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
    """Generates normalized vector embeddings for cosine similarity retrieval."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, chunks


def retrieve_top_k(query: str, embeddings: np.ndarray, chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict]:
    """Cosine semantic search between query vector and stored document vectors."""
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


# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("""
        <div class="status-badge" style="margin-bottom:18px;">
            <span class="status-dot"></span>
            NEURAL ENGINE ONLINE
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ System Config")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Obtain a free key at console.groq.com"
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

    rag_mode = st.radio(
        "RAG Shield Mode",
        ["Strict Grounding (Zero Hallucination)", "Analytical Synthesis"],
        index=0,
        help="Strict mode will never guess outside provided document chunks."
    )

    st.markdown("---")
    st.markdown("### 📂 Ingest Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])

    with st.expander("🎛️ Vector & Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Overlap", 0, 300, 100, 20)
        top_k = st.slider("Top-K Citations", 1, 6, 3, 1)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1 if "Strict" in rag_mode else 0.7, 0.05)

    if uploaded_file and (st.session_state.processed_filename != uploaded_file.name):
        with st.spinner("Extracting text and generating vector embeddings..."):
            raw_text = extract_text_from_file(uploaded_file)
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            vectors, chunks_metadata = build_vector_store(chunks)
            
            st.session_state.vector_store = vectors
            st.session_state.doc_chunks = chunks_metadata
            st.session_state.processed_filename = uploaded_file.name
            st.success(f"Indexed {len(chunks)} chunks into Vector Space!")

    st.markdown("---")
    st.markdown("### 📊 Index HUD")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="hud-card">
            <span style="font-size:0.7rem; color:#64748b; font-weight:700; letter-spacing:0.08em;">CHUNKS</span><br>
            <span style="font-size:1.3rem; font-weight:800; color:#818cf8; letter-spacing:-0.02em;">{len(st.session_state.doc_chunks)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="hud-card">
            <span style="font-size:0.7rem; color:#64748b; font-weight:700; letter-spacing:0.08em;">QUERIES</span><br>
            <span style="font-size:1.3rem; font-weight:800; color:#38bdf8; letter-spacing:-0.02em;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Flush Vector Database", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.doc_chunks = []
        st.session_state.processed_filename = ""
        st.session_state.query_count = 0
        st.rerun()


# Main View Header
st.markdown('<div class="cyber-title">⚡ DOCUMIND RAG STUDIO</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cyber-sub">Quantum semantic search & context-grounded LLM inference engine.</div>',
    unsafe_allow_html=True
)

if st.session_state.processed_filename:
    st.info(f"📂 Active Knowledge Base: **{st.session_state.processed_filename}** ({len(st.session_state.doc_chunks)} dense vector vectors indexed)")
else:
    st.warning("⚠️ Knowledge base empty. Upload a PDF or TXT document in the sidebar to activate semantic grounding.")

# Render Conversation Turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("🔍 Semantic Context Citations & Cosine Scores", expanded=False):
                for c in msg["citations"]:
                    pct = max(0, min(100, int(c["score"] * 100)))
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <b style="color:#e2e8f0; font-size:0.9rem;">Chunk #{c["chunk"]["id"]}</b>
                                <span class="confidence-badge">Similarity: {c["score"]:.4f} ({pct}%)</span>
                            </div>
                            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5;">{c["chunk"]["text"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# Streaming Text Generator
def stream_text_chunks(raw_stream):
    for chunk in raw_stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                yield delta.content


# Query Input & Retrieval Execution
user_query = st.chat_input("Ask a grounded question about your document...")

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
        # Step 1: Semantic Vector Retrieval
        retrieval_start = time.time()
        retrieved_results = retrieve_top_k(user_query, st.session_state.vector_store, st.session_state.doc_chunks, top_k)
        retrieval_latency = round(time.time() - retrieval_start, 3)

        # Step 2: Context Augmentation
        context_str = "\n\n".join([
            f"[Chunk #{r['chunk']['id']} | Cosine Match: {r['score']:.3f}]\n{r['chunk']['text']}"
            for r in retrieved_results
        ])

        if "Strict" in rag_mode:
            system_instruction = (
                "You are an ultra-precise, hallucination-free RAG agent. "
                "Answer the user's question using ONLY the retrieved document chunks below. "
                "If the context does not provide sufficient proof to answer, state: "
                "'I cannot find this information within the provided document.' Never make assumptions.\n\n"
                f"Retrieved Context:\n{context_str}"
            )
        else:
            system_instruction = (
                "You are an advanced technical analyst. Answer the user's inquiry using the retrieved "
                "document context as your primary source, providing reasoned synthesis where appropriate.\n\n"
                f"Retrieved Context:\n{context_str}"
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
                with st.expander("🔍 Semantic Context Citations & Cosine Scores", expanded=False):
                    for c in retrieved_results:
                        pct = max(0, min(100, int(c["score"] * 100)))
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <b style="color:#e2e8f0; font-size:0.9rem;">Chunk #{c["chunk"]["id"]}</b>
                                    <span class="confidence-badge">Similarity: {c["score"]:.4f} ({pct}%)</span>
                                </div>
                                <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5;">{c["chunk"]["text"]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown(
                    f'<div class="telemetry-chip">⚡ Retrieval: <b>{retrieval_latency}s</b> | '
                    f'🤖 Generation: <b>{inference_latency}s</b> | '
                    f'📚 Top-K: <b>{len(retrieved_results)}</b></div>',
                    unsafe_allow_html=True
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "citations": retrieved_results
                })

            except RateLimitError:
                err = "Groq API rate limit reached. Please wait a moment."
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
