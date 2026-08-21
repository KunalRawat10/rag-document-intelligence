"""
DocuMind RAG Studio - Quantum Neural Document Intelligence
==========================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, SentenceTransformers, NumPy, PyPDF
Theme: Neo-Cyber Iris & Light Glassmorphism
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

# Custom Light Glassmorphism & Cyber Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --ink: #172033;
        --muted: #65708a;
        --paper: #fbfcff;
        --surface: rgba(255, 255, 255, 0.78);
        --line: rgba(94, 104, 140, 0.16);
        --violet: #7357ff;
        --cyan: #00b8d9;
        --pink: #ff5caa;
        --lime: #b8e44c;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--ink) !important;
    }

    code, pre { font-family: 'Space Mono', monospace !important; }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(255, 92, 170, .12), transparent 24%),
            radial-gradient(circle at 94% 2%, rgba(0, 184, 217, .14), transparent 27%),
            linear-gradient(135deg, #f8f9ff 0%, #ffffff 48%, #f7fbff 100%) !important;
        background-attachment: fixed !important;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .24;
        background-image: linear-gradient(rgba(115, 87, 255, .035) 1px, transparent 1px), linear-gradient(90deg, rgba(115, 87, 255, .035) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(to bottom, black, transparent 82%);
    }

    section[data-testid="stSidebar"] {
        background: rgba(250, 251, 255, .84) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid var(--line) !important;
        box-shadow: 12px 0 34px rgba(58, 68, 106, .07) !important;
    }

    section[data-testid="stSidebar"] > div { padding-top: 2rem !important; }
    section[data-testid="stSidebar"] h3 { color: var(--ink) !important; letter-spacing: -.02em; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown p { color: var(--muted) !important; }

    .cyber-title {
        font-size: clamp(2.1rem, 4vw, 3.5rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -.065em;
        background: linear-gradient(100deg, var(--ink) 10%, var(--violet) 48%, var(--pink) 92%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1.3rem 0 .35rem;
    }

    .cyber-sub { color: var(--muted); font-size: 1rem; margin-bottom: 1.35rem; }

    .hud-card, .citation-card {
        background: var(--surface) !important;
        backdrop-filter: blur(18px);
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 12px 35px rgba(68, 77, 115, .08), inset 0 1px 0 rgba(255,255,255,.9) !important;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease !important;
    }

    .hud-card { padding: 15px !important; margin-bottom: 11px !important; }
    .hud-card:hover, .citation-card:hover { transform: translateY(-3px); border-color: rgba(115,87,255,.34) !important; box-shadow: 0 18px 38px rgba(68, 77, 115, .13) !important; }

    .citation-card { border-left: 4px solid var(--cyan) !important; padding: 14px 17px !important; margin: 9px 0 !important; color: var(--ink) !important; }
    .citation-card b { color: var(--ink) !important; }
    .citation-card div[style*="color:#94a3b8"] { color: var(--muted) !important; }

    .confidence-badge, .telemetry-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(184, 228, 76, .18) !important;
        color: #547400 !important;
        border: 1px solid rgba(120, 165, 27, .28) !important;
        border-radius: 999px !important;
        font-size: .72rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    .confidence-badge { padding: 4px 9px; }
    .telemetry-chip { padding: 8px 15px; margin-top: 12px; box-shadow: 0 8px 20px rgba(88, 113, 20, .08); }

    .stButton > button {
        background: linear-gradient(105deg, var(--violet), #967cff) !important;
        color: #fff !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 18px rgba(115,87,255,.22) !important;
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; filter: saturate(1.12) !important; box-shadow: 0 13px 25px rgba(115,87,255,.3) !important; }
    .stButton > button:active { transform: translateY(0) scale(.985) !important; }

    .stTextInput input, .stSelectbox [data-baseweb="select"], .stTextArea textarea, .stFileUploader {
        border-radius: 12px !important;
        border-color: var(--line) !important;
        background: rgba(255,255,255,.72) !important;
    }
    .stChatInput { border-color: rgba(115,87,255,.28) !important; box-shadow: 0 10px 30px rgba(115,87,255,.10) !important; }
    [data-testid="stAlert"] { border-radius: 14px !important; border: 1px solid var(--line) !important; box-shadow: 0 8px 24px rgba(68,77,115,.06); }
    hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)


# Load Dense Embedding Model
@st.cache_resource(show_spinner="Initializing dense vector embedder...")
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
        <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(0,184,217,0.12); color:#00b8d9; padding:4px 12px; border-radius:9999px; font-size:0.75rem; font-weight:700; border:1px solid rgba(0,184,217,0.3); margin-bottom:14px;">
            <span style="width:6px; height:6px; background:#00b8d9; border-radius:50%;"></span>
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
            <span style="font-size:0.7rem; color:var(--muted); font-weight:700;">CHUNKS</span><br>
            <span style="font-size:1.2rem; font-weight:800; color:var(--violet);">{len(st.session_state.doc_chunks)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="hud-card">
            <span style="font-size:0.7rem; color:var(--muted); font-weight:700;">QUERIES</span><br>
            <span style="font-size:1.2rem; font-weight:800; color:var(--cyan);">{st.session_state.query_count}</span>
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
    st.info(f"📂 Active Knowledge Base: **{st.session_state.processed_filename}** ({len(st.session_state.doc_chunks)} dense vectors indexed)")
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
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <b>Chunk #{c["chunk"]["id"]}</b>
                                <span class="confidence-badge">Similarity: {c["score"]:.4f} ({pct}%)</span>
                            </div>
                            <div style="color:var(--muted); font-size:0.82rem;">{c["chunk"]["text"]}</div>
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
                                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                    <b>Chunk #{c["chunk"]["id"]}</b>
                                    <span class="confidence-badge">Similarity: {c["score"]:.4f} ({pct}%)</span>
                                </div>
                                <div style="color:var(--muted); font-size:0.82rem;">{c["chunk"]["text"]}</div>
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
