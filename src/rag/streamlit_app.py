import sys
import shutil
import time
from pathlib import Path
import streamlit as st

# Setup system path
PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag.search import RAGSearch, AVAILABLE_MODELS, DEFAULT_GROQ_MODEL
from rag.url_ingester import URLIngester
from evaluate import EVALUATION_DATASET

st.set_page_config(
    page_title="NexusRAG • Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean vector SVG icons (replaces cartoon emojis with crisp vector geometry)
ICON_SPARK_SM = """<svg width="15" height="15" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/></svg>"""

ICON_SPARK_LG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/></svg>"""

ICON_DOC = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>"""

ICON_LINK = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>"""

# --- CHATGPT-STYLE CUSTOM CSS (EXACT SAME COLORS PRESERVED) ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

/* Hide Streamlit Watermarks & Chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden; height: 0px;}
.stDeployButton {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}

/* Background Radial Glow */
.stApp {
    background-color: #080c14;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 45%);
}

/* ChatGPT Centered Max-Width Container */
.block-container {
    max-width: 820px !important;
    padding-top: 1rem !important;
    padding-bottom: 6.5rem !important;
    margin: 0 auto !important;
}

/* ChatGPT Sidebar (Dark Slate) */
section[data-testid="stSidebar"] {
    background-color: #0d131f !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
}

section[data-testid="stSidebar"] .block-container {
    max-width: 100% !important;
    padding: 1.2rem 1rem 2rem 1rem !important;
}

/* New Chat Button (ChatGPT Top Left Signature Button) */
.new-chat-btn button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25)) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease !important;
}

.new-chat-btn button:hover {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    border-color: #6366f1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    margin: 18px 0 8px 4px;
}

/* Document Item Row in Sidebar */
.doc-row-label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-top: 6px;
    overflow: hidden;
}

.doc-name-text {
    font-size: 0.8rem;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 175px;
}

/* Minimal Close Button in Document List */
div[data-testid="stSidebar"] div[data-testid="column"]:last-child button {
    padding: 2px 6px !important;
    min-height: 26px !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: transparent !important;
    border-radius: 6px !important;
}

div[data-testid="stSidebar"] div[data-testid="column"]:last-child button:hover {
    color: #f87171 !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    background: rgba(239, 68, 68, 0.1) !important;
}

/* Minimalist ChatGPT Topbar */
.chatgpt-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 16px 0;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #ffffff;
}

.topbar-brand-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}

.topbar-meta {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 9999px;
    font-size: 0.74rem;
    color: #34d399;
    font-weight: 600;
}

.status-pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.8; transform: scale(0.6); }
}

.model-tag {
    padding: 4px 10px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 9999px;
    font-size: 0.74rem;
    color: #c7d2fe;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* ChatGPT Empty Hero */
.chatgpt-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 1rem 2rem 1rem;
    text-align: center;
}

.chatgpt-hero-icon {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.45);
    margin-bottom: 18px;
}

.chatgpt-hero h2 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 6px;
}

.chatgpt-hero p {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Prompt Suggestion Cards (ChatGPT 2x2 Grid) */
.prompt-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    width: 100%;
    margin-top: 8px;
}

.prompt-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
}

.prompt-card:hover {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
}

.prompt-card-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 4px;
}

.prompt-card-sub {
    font-size: 0.76rem;
    color: #64748b;
    line-height: 1.4;
}

/* ChatGPT Message Flow */
/* ChatGPT / Gemini Message Flow (Completely Borderless & Seamless Canvas) */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 12px 0px !important;
    margin-bottom: 20px !important;
}

/* User Message: Elegant pill bubble like ChatGPT */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 20px !important;
    padding: 12px 18px !important;
    max-width: 85% !important;
    margin-left: auto !important;
}

/* Assistant Message: Zero borders, zero cards, completely natural flow like Gemini/ChatGPT */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
div[data-testid="stChatMessage"]:not(:has(div[data-testid="stChatMessageAvatarUser"])) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0px !important;
    border-radius: 0 !important;
}

div[data-testid="stChatMessageContent"] {
    color: #e2e8f0 !important;
    font-size: 0.98rem !important;
    line-height: 1.75 !important;
}

/* Tables Styling */
table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin: 14px 0 !important;
}

th {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    font-size: 0.86rem !important;
}

td {
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #cbd5e1 !important;
    font-size: 0.84rem !important;
    line-height: 1.6 !important;
}

tr:last-child td {
    border-bottom: none !important;
}

tr:hover td {
    background: rgba(255, 255, 255, 0.02) !important;
}

/* Sidebar Metrics Pill */
.sidebar-metrics-bar {
    display: flex;
    justify-content: space-around;
    padding: 8px 4px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    margin: 10px 0 16px 0;
}

.sidebar-metric-item {
    text-align: center;
}

.sidebar-metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
}

.sidebar-metric-name {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
}

/* Floating Bottom Input Bar */
div[data-testid="stChatInput"] {
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    background: #0f172a !important;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3), 0 10px 35px rgba(0, 0, 0, 0.6) !important;
}

/* Bottom Bar Hosting Custom Gemini Input Component */
div[data-testid="stBottom"] iframe {
    width: 100% !important;
    border: none !important;
    background: transparent !important;
}

/* Authentic Google Gemini Dialog Styling */
div[data-testid="stDialog"] > div[role="dialog"] {
    background: #1e1f20 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 22px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.85) !important;
    color: #e3e3e3 !important;
    padding: 20px 24px !important;
    max-width: 520px !important;
}

div[data-testid="stDialog"] header {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 10px !important;
    margin-bottom: 14px !important;
}

div[data-testid="stDialog"] h2 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}

div[data-testid="stDialog"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

div[data-testid="stDialog"] .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 14px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

div[data-testid="stDialog"] .stTabs [aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.25) !important;
    color: #ffffff !important;
}

.chatgpt-footer-disclaimer {
    text-align: center;
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

import importlib
import inspect
import base64
import streamlit.components.v1 as components
import rag.search

# Declare custom Gemini-style Chat Input Component
FRONTEND_DIR = Path(__file__).parent / "frontend"
gemini_chat_input = components.declare_component("gemini_chat_input", path=str(FRONTEND_DIR.resolve()))

# Gemini Attachment Dialog (Triggered by the + Button in Chat Input)
@st.dialog("Ask NexusRAG")
def open_gemini_attachment_dialog():
    st.markdown("""
        <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 14px;">
            Choose what you would like to upload and vectorize into your private knowledge base:
        </div>
    """, unsafe_allow_html=True)

    tab_files, tab_img, tab_url = st.tabs(["📎 Upload files", "🖼️ Images", "🌐 Web link"])

    with tab_files:
        st.caption("PDF, Word (.docx), Excel (.xlsx), CSV, JSON, PPTX, TXT")
        uploaded_docs = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "doc", "xlsx", "xls", "json", "jsonl", "csv", "pptx", "txt", "md"],
            accept_multiple_files=True,
            key="dlg_doc_uploader",
            label_visibility="collapsed"
        )
        if uploaded_docs:
            if st.button("Index Documents", type="primary", key="btn_dlg_docs", use_container_width=True):
                data_dir = PROJECT_DIR / "data"
                saved = []
                with st.spinner(f"Indexing {len(uploaded_docs)} document(s)..."):
                    for uf in uploaded_docs:
                        ext = Path(uf.name).suffix.lower()
                        target_dir = data_dir / ("pdf" if ext == ".pdf" else "text_files")
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / uf.name
                        with open(target_path, "wb") as f:
                            f.write(uf.getbuffer())
                        rag.add_document_file(target_path)
                        saved.append(uf.name)
                    st.toast(f"Successfully indexed {len(saved)} document(s).")
                    st.rerun()

    with tab_img:
        st.caption("PNG, JPG, JPEG, WEBP (resolutions, EXIF & visual analysis)")
        uploaded_imgs = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="dlg_img_uploader",
            label_visibility="collapsed"
        )
        if uploaded_imgs:
            if st.button("Index Images", type="primary", key="btn_dlg_imgs", use_container_width=True):
                img_dir = PROJECT_DIR / "data" / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                with st.spinner(f"Indexing {len(uploaded_imgs)} image(s)..."):
                    for uf in uploaded_imgs:
                        target_path = img_dir / uf.name
                        with open(target_path, "wb") as f:
                            f.write(uf.getbuffer())
                        rag.add_document_file(target_path)
                        saved.append(uf.name)
                    st.toast(f"Successfully indexed {len(saved)} image(s).")
                    st.rerun()

    with tab_url:
        st.caption("Scrape & index any public webpage, documentation, or article")
        pop_url = st.text_input(
            "Web URL",
            placeholder="https://en.wikipedia.org/... or article link",
            key="dlg_url_input",
            label_visibility="collapsed"
        )
        if st.button("Ingest Web Resource", type="primary", key="btn_dlg_url", use_container_width=True):
            if pop_url and pop_url.strip().startswith(('http://', 'https://')):
                web_docs_dir = PROJECT_DIR / "data" / "web_docs"
                ingester = URLIngester()
                try:
                    with st.spinner("Scraping and indexing web content..."):
                        saved_path, u_title, chars = ingester.ingest_url_to_file(pop_url.strip(), web_docs_dir)
                        rag.add_document_file(saved_path)
                        st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
                        st.rerun()
                except Exception as e_url:
                    st.error(f"Failed to ingest URL: {e_url}")
            else:
                st.warning("Please enter a valid URL starting with http:// or https://")

# Initialize RAG Engine in Streamlit Session State
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    importlib.reload(rag.search)
    return rag.search.RAGSearch(
        persist_dir=str(PROJECT_DIR / "faiss_store"),
        data_dir=str(PROJECT_DIR / "data"),
        llm_model=DEFAULT_GROQ_MODEL
    )

with st.spinner("Connecting to FAISS & BM25 Core..."):
    rag = load_rag_engine()

def safe_search_context(rag_engine, query, top_k, mode, use_reranker=True, enable_web_search=True):
    """Safely invoke search_context handling any in-flight module caching gracefully."""
    sig = inspect.signature(rag_engine.search_context).parameters
    kwargs = {"top_k": top_k, "mode": mode}
    if "use_reranker" in sig:
        kwargs["use_reranker"] = use_reranker
    if "enable_web_search" in sig:
        kwargs["enable_web_search"] = enable_web_search
    return rag_engine.search_context(query, **kwargs)

def safe_stream_query(rag_engine, query, chat_history, top_k, model_name, retrieval_mode, use_reranker=True, enable_web_search=True):
    """Safely invoke stream_query handling any in-flight module caching gracefully."""
    sig = inspect.signature(rag_engine.stream_query).parameters
    kwargs = {
        "chat_history": chat_history,
        "top_k": top_k,
        "model_name": model_name,
        "retrieval_mode": retrieval_mode
    }
    if "use_reranker" in sig:
        kwargs["use_reranker"] = use_reranker
    if "enable_web_search" in sig:
        kwargs["enable_web_search"] = enable_web_search
    return rag_engine.stream_query(query, **kwargs)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DYNAMIC ACTION HANDLERS ---
def handle_file_deletion(doc_name: str):
    """Instantly delete a document from knowledge base and disk in milliseconds."""
    try:
        if hasattr(rag, "remove_document"):
            rag.remove_document(doc_name)
        else:
            # Fallback
            data_dir = PROJECT_DIR / "data"
            for p in data_dir.rglob(doc_name):
                if p.is_file():
                    p.unlink()
            rag.reindex_all()
        st.toast(f"Removed '{doc_name}' instantly.")
        return True
    except Exception as e:
        st.error(f"Error removing '{doc_name}': {e}")
        return False

# =========================================================
# --- 1. CHATGPT SIDEBAR: NAVIGATION & KNOWLEDGE ASSETS ---
# =========================================================
with st.sidebar:
    # New Chat Button (Prominent at top like ChatGPT)
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar Knowledge Base Metrics
    stats = rag.vectorstore.get_stats()
    docs_count = stats.get("total_documents", 0)
    chunks_count = stats.get("total_chunks", 0)
    
    st.markdown(f"""
        <div class="sidebar-metrics-bar">
            <div class="sidebar-metric-item">
                <div class="sidebar-metric-num">{docs_count}</div>
                <div class="sidebar-metric-name">Docs</div>
            </div>
            <div class="sidebar-metric-item">
                <div class="sidebar-metric-num">{chunks_count}</div>
                <div class="sidebar-metric-name">Chunks</div>
            </div>
            <div class="sidebar-metric-item">
                <div class="sidebar-metric-num">384-D</div>
                <div class="sidebar-metric-name">FAISS</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Document Inventory (Clean Vector Icon + Truncated Name)
    st.markdown('<div class="sidebar-section-title">Knowledge Documents</div>', unsafe_allow_html=True)
    active_docs = stats.get("documents", [])
    if not active_docs:
        st.caption("No files uploaded yet.")
    else:
        for doc in active_docs:
            col_doc, col_action = st.columns([5, 1])
            with col_doc:
                st.markdown(
                    f'<div class="doc-row-label">{ICON_DOC}<span class="doc-name-text" title="{doc}">{doc}</span></div>',
                    unsafe_allow_html=True
                )
            with col_action:
                if st.button("×", key=f"del_{doc}", help=f"Remove {doc} from knowledge base"):
                    handle_file_deletion(doc)
                    st.rerun()

    # System & Retrieval Settings (Clean Expander)
    st.markdown('<div class="sidebar-section-title">Settings & Engine</div>', unsafe_allow_html=True)
    
    # Model Selection
    model_options = {m["name"]: m["id"] for m in AVAILABLE_MODELS}
    selected_model_name = st.selectbox("Model", list(model_options.keys()), index=0)
    selected_model_id = model_options[selected_model_name]

    # Retrieval Mode
    retrieval_mode_label = st.selectbox(
        "Retrieval Method",
        ["Hybrid Search (BM25 + FAISS)", "Dense Vector Only (FAISS)", "Sparse Keyword Only (BM25)"],
        index=0
    )
    mode_map = {
        "Hybrid Search (BM25 + FAISS)": "hybrid",
        "Dense Vector Only (FAISS)": "dense",
        "Sparse Keyword Only (BM25)": "sparse"
    }
    retrieval_mode = mode_map[retrieval_mode_label]

    use_reranker = st.toggle(
        "Cross-Encoder Re-Ranking",
        value=True,
        help="Applies a two-stage pipeline: retrieves candidate pool and re-scores with MS-MARCO Cross-Encoder for maximum accuracy"
    )

    enable_web_search = st.toggle(
        "Live Multi-Source Search",
        value=True,
        help="Augments document retrieval with live findings from Wikipedia, ArXiv, and global search providers."
    )

    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=10, value=4)

    # Actions: Reindex & Benchmark
    col_reindex, col_bench = st.columns(2)
    with col_reindex:
        if st.button("Reindex", use_container_width=True, help="Re-scan all files"):
            with st.spinner("Rebuilding..."):
                rag.reindex_all()
            st.toast("Index rebuilt successfully.")
            st.rerun()

    with col_bench:
        if st.button("Benchmark", use_container_width=True, help="Test retrieval Hit Rate"):
            with st.spinner("Benchmarking..."):
                results = []
                for item in EVALUATION_DATASET:
                    q = item["query"]
                    exp = item["expected_doc"].lower()
                    cits = safe_search_context(
                        rag,
                        q,
                        top_k=3,
                        mode=retrieval_mode,
                        use_reranker=use_reranker,
                        enable_web_search=enable_web_search
                    )
                    hit = any(exp in c["source"].lower() for c in cits)
                    results.append({"Query": q[:35] + "...", "Hit": "Passed" if hit else "Failed"})
                st.dataframe(results, use_container_width=True)

    # Export Chat
    if st.session_state.messages:
        chat_markdown = "# NexusRAG Chat Export\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].upper()}:\n{m['content']}\n\n"
        st.download_button(
            label="Export Chat (.md)",
            data=chat_markdown,
            file_name="nexusrag_chat_transcript.md",
            mime="text/markdown",
            use_container_width=True
        )

# =========================================================
# --- 2. MAIN STAGE: CHATGPT FLOATING HEADER & CONVERSATION
# =========================================================

# Floating Minimalist Topbar (Vector SVG Spark Icon)
if enable_web_search and use_reranker:
    pipeline_status = "Omni-Source RAG (Docs + Web + Re-Ranker)"
elif enable_web_search:
    pipeline_status = "Omni-Source RAG (Docs + Live Web)"
elif use_reranker:
    pipeline_status = "Two-Stage RAG (Cross-Encoder)"
else:
    pipeline_status = f"{retrieval_mode.capitalize()} RAG (Local Docs Only)"

is_pg_live = bool(hasattr(rag, "pg_db") and rag.pg_db and rag.pg_db.is_connected)
db_label = "PostgreSQL (pgvector)" if is_pg_live else "FAISS VectorStore"

st.markdown(f"""
    <div class="chatgpt-topbar">
        <div class="topbar-brand">
            <div class="topbar-brand-icon">{ICON_SPARK_SM}</div>
            <span>NexusRAG</span>
        </div>
        <div class="topbar-meta">
            <div class="status-pill">
                <span class="status-pulse"></span>
                <span>{pipeline_status}</span>
            </div>
            <div class="model-tag" style="border-color: {'rgba(16, 185, 129, 0.35)' if is_pg_live else 'rgba(99, 102, 241, 0.3)'}; color: {'#34d399' if is_pg_live else '#c7d2fe'};">
                {db_label}
            </div>
            <div class="model-tag">
                {selected_model_name.split(' ')[0]}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Helper function to clean raw HTML tags that leak into markdown tables
def clean_markdown_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return text.replace("<br>•", "\n• ").replace("<br>", " ")

# Display Conversation History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(clean_markdown_content(msg["content"]))
        if "citations" in msg and msg["citations"]:
            with st.expander(f"View {len(msg['citations'])} Verified Source Citations"):
                for i, c in enumerate(msg["citations"], 1):
                    if c.get("is_web"):
                        st.markdown(f"**Web Source {i}: [{c['source']}]({c['page']})** — *Confidence: {c['relevance_pct']}%* `[{c['retrieval_mode']}]`")
                    else:
                        st.markdown(f"**Doc {i}: `{c['source']}`** (Page {c['page']}) — *Relevance: {c['relevance_pct']}%* `[{c['retrieval_mode']}]`")
                    st.code(c["full_text"], language="text")

# Toolbar beneath the last assistant message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col_regen, _ = st.columns([2, 5])
    with col_regen:
        if st.button("Regenerate Answer", help="Re-query model with current settings"):
            st.session_state.messages.pop()
            last_user_query = st.session_state.messages.pop()["content"]
            st.session_state["pending_prompt"] = last_user_query
            st.rerun()

# Check for query from input or suggestion card
pending_query = st.session_state.pop("pending_prompt", None)

with st.bottom:
    component_output = gemini_chat_input(key="gemini_bottom_input")

# Extract query text and handle any file or URL ingestion from the component
active_query = None

if component_output:
    action = component_output.get("action")
    if action == "open_gemini_menu":
        open_gemini_attachment_dialog()
    elif action == "send_message":
        active_query = component_output.get("text", "").strip() or None
        raw_files = component_output.get("files", []) or []
        if raw_files:
            data_dir = PROJECT_DIR / "data"
            saved_names = []
            with st.spinner(f"Ingesting {len(raw_files)} uploaded resource(s)..."):
                for rf in raw_files:
                    fname = rf.get("name", "uploaded_file")
                    b64_data = rf.get("data", "")
                    if b64_data:
                        file_bytes = base64.b64decode(b64_data)
                        ext = Path(fname).suffix.lower()
                        if ext == ".pdf":
                            target_dir = data_dir / "pdf"
                        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                            target_dir = data_dir / "images"
                        else:
                            target_dir = data_dir / "text_files"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / fname
                        with open(target_path, "wb") as f:
                            f.write(file_bytes)
                        rag.add_document_file(target_path)
                        saved_names.append(fname)
            if saved_names:
                st.toast(f"Ingested and indexed {len(saved_names)} document/image resource(s).")
    elif action == "ingest_url":
        target_url = component_output.get("url", "").strip()
        if target_url and target_url.startswith(('http://', 'https://')):
            web_docs_dir = PROJECT_DIR / "data" / "web_docs"
            ingester = URLIngester()
            try:
                with st.spinner(f"Ingesting web resource: {target_url[:45]}..."):
                    saved_path, u_title, chars = ingester.ingest_url_to_file(target_url, web_docs_dir)
                    rag.add_document_file(saved_path)
                    st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
                    st.rerun()
            except Exception as e_url:
                st.error(f"Failed to ingest URL: {e_url}")

if not active_query and pending_query:
    active_query = pending_query

# Auto-detect and ingest any web URLs entered or pasted into the chat bar
if active_query:
    import re
    url_pattern = r'https?://[^\s<>"]+'
    found_urls = re.findall(url_pattern, active_query)
    if found_urls:
        web_docs_dir = PROJECT_DIR / "data" / "web_docs"
        ingester = URLIngester()
        for u in found_urls:
            try:
                with st.spinner(f"Ingesting web resource: {u[:45]}..."):
                    saved_path, u_title, chars = ingester.ingest_url_to_file(u, web_docs_dir)
                    rag.add_document_file(saved_path)
                    st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
            except Exception as e_url:
                print(f"[WARN] Failed to auto-ingest URL {u}: {e_url}")

# If user pasted a standalone URL without query, refresh inventory
is_standalone_url = bool(active_query and active_query.strip().startswith(('http://', 'https://')) and len(active_query.split()) == 1)
if is_standalone_url:
    st.rerun()

# =========================================================
# --- 3. CHATGPT EMPTY STATE: HERO & 2x2 SUGGESTION CARDS --
# =========================================================
if len(st.session_state.messages) == 0 and not active_query:
    st.markdown(f"""
        <div class="chatgpt-hero">
            <div class="chatgpt-hero-icon">{ICON_SPARK_LG}</div>
            <h2>How can I help you today?</h2>
            <p>Ask anything about your private knowledge base</p>
        </div>
    """, unsafe_allow_html=True)

    # ChatGPT 2x2 Grid of Prompt Starters (Clean Typography, No Raw Emojis)
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("Supervised vs Unsupervised\nCompare algorithms, labeled data & objectives", use_container_width=True, key="card_1"):
            st.session_state["pending_prompt"] = "What are the core differences between Supervised and Unsupervised Learning?"
            st.rerun()

        if st.button("AI & Data Science Convergence\nHow machine learning & analytics intersect in business", use_container_width=True, key="card_2"):
            st.session_state["pending_prompt"] = "How do AI and Data Science converge in business solutions?"
            st.rerun()

    with col_b:
        if st.button("Neural Network Architectures\nExplore CNNs, RNNs, and LSTMs in deep learning", use_container_width=True, key="card_3"):
            st.session_state["pending_prompt"] = "Explain Neural Network Architectures mentioned in the docs."
            st.rerun()

        if st.button("Python Programming Fundamentals\nCore syntax, data types, and libraries", use_container_width=True, key="card_4"):
            st.session_state["pending_prompt"] = "What are the key concepts and applications of Python?"
            st.rerun()

# =========================================================
# --- 4. QUERY DISPATCHER & STREAMING GENERATION ----------
# =========================================================
if active_query:
    # 1. User Message (Streamlit built-in sleek vector silhouette)
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Assistant Message (Streamlit built-in sleek vector silhouette)
    with st.chat_message("assistant"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        full_response = ""
        citations = []

        try:
            status_text = f"Searching Local Docs" + (" + Cross-Encoder" if use_reranker else "") + (" + Live Web & ArXiv..." if enable_web_search else "...")
            with st.spinner(status_text):
                citations = safe_search_context(
                    rag,
                    active_query,
                    top_k=top_k,
                    mode=retrieval_mode,
                    use_reranker=use_reranker,
                    enable_web_search=enable_web_search
                )

            def token_stream():
                try:
                    for token in safe_stream_query(
                        rag,
                        query=active_query,
                        chat_history=history,
                        top_k=top_k,
                        model_name=selected_model_id,
                        retrieval_mode=retrieval_mode,
                        use_reranker=use_reranker,
                        enable_web_search=enable_web_search
                    ):
                        if token:
                            yield token
                except Exception as s_err:
                    yield f"\n\n*Notice: {s_err}*"

            full_response = st.write_stream(token_stream())

            # Render Citation Drawer / Expander only if relevant citations exist
            if citations:
                with st.expander(f"View {len(citations)} Verified Source Citations"):
                    for i, c in enumerate(citations, 1):
                        if c.get("is_web"):
                            st.markdown(f"**Web Source {i}: [{c['source']}]({c['page']})** — *Confidence: {c['relevance_pct']}%* `[{c['retrieval_mode']}]`")
                        else:
                            st.markdown(f"**Doc {i}: `{c['source']}`** (Page {c['page']}) — *Relevance: {c['relevance_pct']}%* `[{c['retrieval_mode']}]`")
                        st.code(c["full_text"], language="text")

        except Exception as err:
            full_response = f"**Backend Error:** {err}"
            st.error(full_response)

        # 3. Save Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations
        })

# Bottom Disclaimer
st.markdown(
    '<div class="chatgpt-footer-disclaimer">NexusRAG synthesizes answers from your verified documents. Citations are grounded in source text.</div>',
    unsafe_allow_html=True
)
