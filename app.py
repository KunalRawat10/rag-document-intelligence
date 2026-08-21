"""
DocuMind RAG Studio - Neural Document Intelligence
==================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, SentenceTransformers, NumPy, PyPDF
Theme: Minimal Aurora - bright accents, soft glass, subtle motion
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

# ──────────────────────────────────────────────────────────────
#  MINIMAL AURORA THEME
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #06070d;
        --ink: #e8ecf6;
        --muted: #8b93a7;
        --line: rgba(255,255,255,0.07);
        --glass: rgba(255,255,255,0.035);
        --c1: #22d3ee;   /* cyan   */
        --c2: #a78bfa;   /* violet */
        --c3: #fb7185;   /* rose   */
        --c4: #34d399;   /* mint   */
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: var(--ink) !important;
    }
    code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* Living aurora canvas */
    .stApp {
        background: var(--bg) !important;
        background-image:
            radial-gradient(60rem 40rem at 12% -10%, rgba(34,211,238,0.18), transparent 60%),
            radial-gradient(50rem 36rem at 92% 4%,  rgba(167,139,250,0.16), transparent 60%),
            radial-gradient(48rem 34rem at 55% 108%, rgba(251,113,133,0.12), transparent 60%) !important;
        background-attachment: fixed !important;
        animation: auroraDrift 22s ease-in-out infinite alternate;
    }
    @keyframes auroraDrift {
        0%   { background-position: 0% 0%, 100% 0%, 50% 100%; }
        100% { background-position: 8% 6%, 88% 10%, 44% 92%; }
    }

    /* Fade-up entrance for main content */
    section.main .block-container {
        padding-top: 2.4rem;
        max-width: 1100px;
        animation: riseIn .7s cubic-bezier(.22,.8,.25,1) both;
    }
    @keyframes riseIn { from { opacity:0; transform: translateY(14px);} to { opacity:1; transform:none; } }

    /* Sidebar: quiet frosted panel */
    section[data-testid="stSidebar"] {
        background: rgba(9,11,18,0.72) !important;
        backdrop-filter: blur(22px) saturate(140%) !important;
        border-right: 1px solid var(--line) !important;
    }
    section[data-testid="stSidebar"] * { font-size: 0.9rem; }

    /* Title */
    .cyber-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1.4px;
        line-height: 1.05;
        background: linear-gradient(100deg, #ffffff 0%, var(--c1) 38%, var(--c2) 66%, var(--c3) 100%);
        background-size: 240% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: sweep 9s ease-in-out infinite;
        margin-bottom: 4px;
    }
    @keyframes sweep { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }

    .cyber-sub { color: var(--muted); font-size: 1rem; margin-bottom: 26px; font-weight: 500; }

    /* Status dot */
    .status-row { display:flex; align-items:center; gap:8px; font-size:.72rem; letter-spacing:.14em;
                  text-transform:uppercase; color:var(--muted); font-weight:700; margin-bottom:18px; }
    .dot { width:8px; height:8px; border-radius:50%; background:var(--c4);
           box-shadow:0 0 0 0 rgba(52,211,153,.7); animation: pulse 2s infinite; }
    @keyframes pulse {
        0%   { box-shadow:0 0 0 0 rgba(52,211,153,.55); }
        70%  { box-shadow:0 0 0 10px rgba(52,211,153,0); }
        100% { box-shadow:0 0 0 0 rgba(52,211,153,0); }
    }

    /* Stat cards */
    .hud-card {
        background: var(--glass);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: transform .25s cubic-bezier(.22,.8,.25,1), border-color .25s, background .25s;
    }
    .hud-card:hover { transform: translateY(-3px); border-color: rgba(34,211,238,.4); background: rgba(255,255,255,.06); }
    .hud-label { font-size:.65rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:700; }
    .hud-value { font-size:1.6rem; font-weight:800; margin-top:2px;
                 background:linear-gradient(120deg,var(--c1),var(--c2));
                 -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }

    /* Citation cards */
    .citation-card {
        position: relative;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px 14px 18px;
        margin: 10px 0;
        font-size: .87rem;
        color: #c6ccdb;
        overflow: hidden;
        animation: riseIn .45s ease both;
        transition: border-color .25s, transform .25s;
    }
    .citation-card::before {
        content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
        background: linear-gradient(180deg, var(--c1), var(--c2));
    }
    .citation-card:hover { border-color: rgba(167,139,250,.4); transform: translateX(3px); }
    .cite-head { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px; }
    .cite-id { font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--c1); font-weight:600; }
    .cite-text { color:#aab2c5; line-height:1.6; font-size:.83rem; margin:0; }

    /* Confidence pill + score bar */
    .confidence-badge {
        display:inline-flex; align-items:center; gap:6px;
        background: rgba(34,211,238,.08); color: var(--c1);
        border: 1px solid rgba(34,211,238,.28);
        padding: 3px 10px; border-radius: 999px;
        font-size:.7rem; font-weight:600; font-family:'JetBrains Mono',monospace;
    }
    .score-track { height:3px; border-radius:99px; background:rgba(255,255,255,.07); margin:10px 0 4px; overflow:hidden; }
    .score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--c1),var(--c2));
                  animation: grow .9s cubic-bezier(.22,.8,.25,1) both; }
    @keyframes grow { from { width:0 !important; } }

    /* Buttons */
    .stButton > button {
        background: rgba(255,255,255,0.04) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: .86rem !important;
        padding: .5rem 1rem !important;
        transition: all .22s cubic-bezier(.22,.8,.25,1) !important;
    }
    .stButton > button:hover {
        border-color: rgba(34,211,238,.55) !important;
        color: #fff !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -10px rgba(34,211,238,.6) !important;
    }
    .stButton > button:active { transform: translateY(0); }

    /* Inputs */
    input, textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,.035) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        color: var(--ink) !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput input:focus, textarea:focus {
        border-color: rgba(34,211,238,.6) !important;
        box-shadow: 0 0 0 3px rgba(34,211,238,.12) !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.03) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        animation: riseIn .45s ease both;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(10,12,20,.85) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(16px);
    }
    [data-testid="stChatInput"]:focus-within { border-color: rgba(34,211,238,.5) !important; }

    /* Alerts */
    [data-testid="stAlert"] {
        background: rgba(255,255,255,.035) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        color: var(--ink) !important;
    }

    /* Expander */
    details, [data-testid="stExpander"] {
        background: rgba(255,255,255,.025) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
    }

    [data-testid="stSlider"] [role="slider"] { background: var(--c1) !important; border-color: var(--c1) !important; }

    .telemetry-chip {
        display:inline-flex; gap:14px; align-items:center;
        background: rgba(255,255,255,.03);
        color: var(--muted);
        padding: 8px 16px; border-radius: 999px;
        font-size:.76rem; margin-top:12px;
        border: 1px solid var(--line);
        font-family:'JetBrains Mono', monospace;
        animation: riseIn .5s ease both;
    }

    hr { border-color: var(--line) !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.09); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,.4); }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def citation_html(c) -> str:
    """Renders one citation card with animated confidence bar."""
    pct = max(0, min(100, int(c["score"] * 100)))
    return f"""
    <div class="citation-card">
      <div class="cite-head">
        <span class="cite-id">CHUNK #{c["chunk"]["id"]}</span>
        <span class="confidence-badge">{c["score"]:.4f} · {pct}%</span>
      </div>
      <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
      <p class="cite-text">{c["chunk"]["text"]}</p>
    </div>
    """


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
    st.markdown(
        '<div class="status-row"><span class="dot"></span> Neural engine online</div>',
        unsafe_allow_html=True
    )

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
        st.markdown(
            f'<div class="hud-card"><div class="hud-label">Chunks</div>'
            f'<div class="hud-value">{len(st.session_state.doc_chunks)}</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="hud-card"><div class="hud-label">Queries</div>'
            f'<div class="hud-value">{st.session_state.query_count}</div></div>',
            unsafe_allow_html=True
        )

    if st.button("🗑️ Flush Vector Database", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.doc_chunks = []
        st.session_state.processed_filename = ""
        st.session_state.query_count = 0
        st.rerun()


# Main View Header
st.markdown('<div class="cyber-title">DocuMind RAG Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cyber-sub">Semantic search & context-grounded LLM inference engine.</div>',
    unsafe_allow_html=True
)

if st.session_state.processed_filename:
    st.info(f"📂 Active Knowledge Base: **{st.session_state.processed_filename}** ({len(st.session_state.doc_chunks)} vectors indexed)")
else:
    st.warning("⚠️ Knowledge base empty. Upload a PDF or TXT document in the sidebar to activate semantic grounding.")

# Render Conversation Turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("🔍 Semantic Context Citations & Cosine Scores", expanded=False):
                for c in msg["citations"]:
                    st.markdown(citation_html(c), unsafe_allow_html=True)


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
                        st.markdown(citation_html(c), unsafe_allow_html=True)

                st.markdown(
                    f'<div class="telemetry-chip"><span>⚡ Retrieval {retrieval_latency}s</span>'
                    f'<span>🤖 Generation {inference_latency}s</span>'
                    f'<span>📚 Top-K {len(retrieved_results)}</span></div>',
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
