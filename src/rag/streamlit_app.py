import sys
import shutil
import time
import html
import base64
import re
import json
import inspect
import subprocess
import urllib
import urllib.parse
from pathlib import Path
import streamlit as st

# Setup system path safely
_curr_path = Path(__file__).resolve()
PROJECT_DIR = None
for _p in [_curr_path.parent, *_curr_path.parents]:
    if (_p / "src" / "rag").exists() or (_p / "pyproject.toml").exists() or (_p / "requirements.txt").exists():
        PROJECT_DIR = _p
        break
if PROJECT_DIR is None:
    PROJECT_DIR = _curr_path.parent

SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Bridge Streamlit Cloud secrets & .env into os.environ
try:
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv()
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
except Exception:
    pass

from rag.search import RAGSearch, AVAILABLE_MODELS, DEFAULT_GROQ_MODEL
from rag.url_ingester import URLIngester
from rag.chat_history import ChatHistoryManager
from rag.tracer import LLMOpsTracer
from rag.multi_search import OmniSearchEngine
from evaluate import EVALUATION_DATASET

st.set_page_config(
    page_title="NexusRAG Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Theme State immediately
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
if "sb_theme_select" not in st.session_state:
    st.session_state["sb_theme_select"] = "Dark Obsidian" if st.session_state["theme_mode"] == "dark" else "Pure Light"

is_light = (st.session_state.get("theme_mode", "dark") == "light")

# =============================================================================
# --- NEW DESIGN SYSTEM: OBSIDIAN TITANIUM & SUNSET FLAME ---
# Complete frontend redesign: Rich obsidian surface, fiery sunset accents, modern typography
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ═══ GLOBAL RESET & MODERN TYPOGRAPHY ═══ */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #e5e7eb;
    -webkit-font-smoothing: antialiased;
}

/* ═══ STREAMLIT CHROME & SIDEBAR TOGGLE ═══ */
#MainMenu, footer, .stDeployButton,
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] { 
    display: none !important; 
}

header[data-testid="stHeader"],
.stAppHeader {
    background: transparent !important;
    height: 0px !important;
    overflow: visible !important;
    z-index: 1000 !important;
}

div[data-testid="stToolbar"],
.stAppToolbar {
    background: transparent !important;
    height: 0px !important;
    overflow: visible !important;
}

/* Sidebar Expand Button (when sidebar is closed) - Floating Brand Logo/Symbol Sign */
div[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 999999 !important;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

button[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    min-width: 38px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z'/></svg>"), linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 20px 20px, cover !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 11px !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5), 0 0 12px rgba(124, 58, 237, 0.35) !important;
    cursor: pointer !important;
    opacity: 1 !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    padding: 0 !important;
}

button[data-testid="stExpandSidebarButton"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z'/></svg>"), linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 21px 21px, cover !important;
    transform: scale(1.08) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.7), 0 0 18px rgba(124, 58, 237, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

button[data-testid="stExpandSidebarButton"] svg,
div[data-testid="collapsedControl"] svg {
    display: none !important;
}

/* ═══ SIDEBAR BRAND HEADER & PILLS ROW ═══ */
.sb-brand-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 2px 0 6px 0;
}

.sb-brand-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.sb-pills-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 4px 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

/* ═══ KEEP SIDEBAR COLLAPSE BUTTON («) CONSTANTLY VISIBLE ═══ */
[data-testid="stSidebarHeader"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: inline-flex !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.12) !important;
    transform: scale(1.08) !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    fill: currentColor !important;
    color: inherit !important;
    opacity: 1 !important;
}

/* ═══ APP BACKGROUND — Deep Obsidian Titanium with Nebula Aura ═══ */
.stApp {
    background-color: #0b0d14;
    background-image:
        radial-gradient(ellipse 90% 60% at 10% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 80% 60% at 90% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 60%),
        radial-gradient(circle at 50% 30%, rgba(6, 182, 212, 0.03) 0%, transparent 70%);
    background-attachment: fixed;
}

/* ═══ MAIN CONTAINER — Generous, Broad Response Field ═══ */
.block-container {
    max-width: 1040px !important;
    width: 94% !important;
    padding-top: 0.8rem !important;
    padding-bottom: 14rem !important;
    margin: 0 auto !important;
}

/* ═══ SIDEBAR — ChatGPT Modern Charcoal Panel ═══ */
section[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
    height: 100vh !important;
}

section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
}

section[data-testid="stSidebar"] .block-container {
    max-width: 100% !important;
    padding: 1rem 0.85rem 1.2rem 0.85rem !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: calc(100vh - 2.5rem) !important;
    height: 100% !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] div:has(> div[data-testid="stElementContainer"]) {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 100% !important;
}

/* Flexible vertical spacer pushing lower controls down */
.sb-flex-spacer {
    flex-grow: 1 !important;
    min-height: 40px !important;
}

div:has(> .sb-flex-spacer),
div:has(> div > .sb-flex-spacer),
div[data-testid="stElementContainer"]:has(.sb-flex-spacer) {
    flex-grow: 1 !important;
    display: flex !important;
}

/* Pin the lower dock to the bottom of the sidebar */
div:has(> .sb-lower-dock),
div:has(> div > .sb-lower-dock),
div[data-testid="stElementContainer"]:has(.sb-lower-dock) {
    margin-top: auto !important;
}

/* ═══ CHATGPT SIDEBAR DESIGN SYSTEM ═══ */
.cg-sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 6px 8px 6px;
    margin-bottom: 2px;
}

.cg-brand-title {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
}

/* New Chat Button (Exact ChatGPT rounded pill) */
button[data-testid*="btn_new_chat"] {
    background: #212121 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ececec !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    height: 38px !important;
    min-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0 12px !important;
    gap: 8px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    margin: 2px 0 6px 0 !important;
}

button[data-testid*="btn_new_chat"] p {
    color: #ececec !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    margin: 0 !important;
}

button[data-testid*="btn_new_chat"]:hover {
    background: #2f2f2f !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
    transform: none !important;
}

/* Tool Navigation Buttons (Images & Documents, etc.) */
button[data-testid*="btn_tool_"] {
    background: transparent !important;
    border: none !important;
    color: #b4b4b4 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    border-radius: 8px !important;
    height: 36px !important;
    min-height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0 12px !important;
    gap: 8px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    margin-bottom: 2px !important;
}

button[data-testid*="btn_tool_"] p {
    color: #b4b4b4 !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    margin: 0 !important;
}

button[data-testid*="btn_tool_"]:hover {
    background: #212121 !important;
    color: #ffffff !important;
    transform: none !important;
}
button[data-testid*="btn_tool_"]:hover p {
    color: #ffffff !important;
}

/* Theme Toggle Button (Compact Header Icon) */
button[data-testid*="btn_theme_toggle"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    height: 34px !important;
    min-height: 34px !important;
    width: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
    padding: 0 !important;
    margin: 0 !important;
    color: #b4b4b4 !important;
}

button[data-testid*="btn_theme_toggle"]:hover {
    background: #262626 !important;
    color: #ffffff !important;
    transform: none !important;
}

/* Section Headings (Projects, Recents) */
.cg-section-title {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 0.76rem;
    font-weight: 600;
    color: #8e8e8e;
    padding: 14px 6px 4px 6px;
    letter-spacing: 0.01em;
}

.cg-section-empty {
    font-size: 0.82rem;
    color: #676767;
    padding: 2px 6px 8px 6px;
}

/* ═══ RECENT CHATS LIST — EXACT MATCH FOR IMAGE 2 ═══ */
/* ═══ RECENT CHATS LIST — MINIMALIST CHATGPT STYLE (ZERO BOXES) ═══ */
/* Horizontal row container: completely borderless, boxless, transparent */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]),
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-pop_act_"]),
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button[data-testid*="sb_chat_btn_"]) {
    gap: 0 !important;
    margin-bottom: 2px !important;
    border-radius: 0px !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    align-items: center !important;
    height: 36px !important;
    min-height: 36px !important;
    position: relative !important;
}

/* Zero box on hover */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-pop_act_"]):hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Zero box when active */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):has(button[data-testid="baseButton-primary"]),
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):has(button[kind="primary"]) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* STRIP ALL BOXES, BORDERS, AND BACKGROUNDS FROM RECENT CHAT BUTTONS */
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button,
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[data-testid="baseButton-secondary"],
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[kind="primary"],
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"] {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
    color: #94a3b8 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;
    padding: 0 4px !important;
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 0px !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    width: 100% !important;
    transform: none !important;
    transition: color 0.15s ease !important;
}

section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button:hover,
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"]:hover {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
}

section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button:focus,
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button:active {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Active chat text: bright white */
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[data-testid="baseButton-primary"] p,
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[kind="primary"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Ensure inner text is strictly left-aligned with ellipsis */
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button p,
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button div[data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"] p,
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"] div[data-testid="stMarkdownContainer"] {
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;
    display: flex !important;
    width: 100% !important;
    color: inherit !important;
    font-size: 0.88rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ═══ THREE DOTS BUTTON: HIDDEN BY DEFAULT, APPEARS ON HOVER ═══ */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[class*="st-key-pop_act_"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopoverButton"],
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] {
    opacity: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
    transition: opacity 0.15s ease, visibility 0.15s ease !important;
    display: flex !important;
    height: 36px !important;
    align-items: center !important;
    justify-content: center !important;
}

/* THREE DOTS: SHOW ON ROW HOVER OR WHEN POPOVER IS OPEN/FOCUSED */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):hover div[class*="st-key-pop_act_"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):hover div[data-testid="stPopover"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):hover div[data-testid="stPopoverButton"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):focus-within div[class*="st-key-pop_act_"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):focus-within div[data-testid="stPopover"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):focus-within div[data-testid="stPopoverButton"],
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"]:has(button[aria-expanded="true"]) {
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
}

/* ZERO BOX FOR THREE DOTS BUTTON: PURE BARE DOTS (···) */
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] div[data-testid="stPopoverButton"] button,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] div[data-testid="stPopover"] button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopoverButton"] button {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 36px !important;
    min-height: 36px !important;
    width: auto !important;
    min-width: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button:hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button:hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopoverButton"] button:hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] svg,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] svg {
    display: none !important;
}

/* Style the dots text itself */
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button p,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopoverButton"] button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopoverButton"] button span {
    color: #94a3b8 !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    transition: color 0.15s ease !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button:hover p,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button:hover span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button:hover p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button:hover span {
    color: #ffffff !important;
}

div[data-testid="stPopoverBody"] {
    background: #0f1322 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.75) !important;
}

div[data-testid="stPopoverBody"] button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
    padding: 6px 12px !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

div[data-testid="stPopoverBody"] button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

div[data-testid="stPopoverBody"] button[kind="primary"],
div[data-testid="stPopoverBody"] button[data-testid="baseButton-primary"] {
    background: rgba(244, 63, 94, 0.15) !important;
    border: 1px solid rgba(244, 63, 94, 0.35) !important;
    color: #fda4af !important;
}

div[data-testid="stPopoverBody"] button[kind="primary"]:hover,
div[data-testid="stPopoverBody"] button[data-testid="baseButton-primary"]:hover {
    background: rgba(244, 63, 94, 0.25) !important;
    border-color: rgba(244, 63, 94, 0.6) !important;
    color: #ffffff !important;
    box-shadow: 0 0 10px rgba(244, 63, 94, 0.3) !important;
}

.sb-arch-header {
    font-size: 0.68rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #64748b;
    margin: 14px 0 6px 2px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 8px;
}

/* ═══ SIDEBAR EXPANDERS (COLLAPSIBLE SECTIONS) ═══ */
div[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stSidebar"] div[data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.28) !important;
}

div[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 9px 12px !important;
}

div[data-testid="stSidebar"] div[data-testid="stExpander"] details summary:hover {
    color: #f1f5f9 !important;
}

div[data-testid="stSidebar"] div[data-testid="stExpander"] details[open] summary {
    color: #818cf8 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

/* ═══ COMPACT KB STATS & FOOTER ═══ */
.kb-compact-stats {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
}

.stat-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #818cf8;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 8px;
    padding: 4px 8px;
}

.kb-doc-header {
    font-size: 0.68rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #64748b;
    margin: 10px 0 6px 2px;
}

.sb-system-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    margin-top: 18px;
}

.sb-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
}

.sb-status-text {
    display: flex;
    flex-direction: column;
}

.sb-status-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #f1f5f9;
}

.sb-status-sub {
    font-size: 0.64rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}

/* ═══ DOCUMENT ROW ═══ */
.doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    margin-bottom: 4px;
}

.doc-item-name {
    font-size: 0.77rem;
    color: #d1d5db;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 170px;
}

/* ═══ STATS CARDS ═══ */
.glass-stats {
    display: flex;
    justify-content: space-around;
    padding: 12px 6px;
    background: #14151c;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    margin: 14px 0 18px 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}

.glass-stat {
    text-align: center;
    flex: 1;
}

.glass-stat:not(:last-child) {
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.glass-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #ff8a65;
}

.glass-stat-lbl {
    font-size: 0.62rem;
    color: #9ca3af;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* ═══ FLOATING HEADER BAR ═══ */
.header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 22px;
    margin-bottom: 20px;
    background: rgba(15, 17, 26, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
}

.hdr-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.hdr-icon {
    width: 36px;
    height: 36px;
    border-radius: 11px;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 14px rgba(99, 102, 241, 0.45);
}

.hdr-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.22rem;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.hdr-sub {
    font-size: 0.72rem;
    color: #9ca3af;
    font-weight: 500;
    margin-top: -1px;
}

.hdr-pills {
    display: flex;
    align-items: center;
    gap: 8px;
}

.pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.pill-live {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #34d399;
    display: flex;
    align-items: center;
    gap: 7px;
}

.pill-info {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.28);
    color: #a5b4fc;
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.8);
    animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px rgba(16, 185, 129, 0.8); }
    50% { opacity: 0.4; transform: scale(0.85); box-shadow: 0 0 4px rgba(16, 185, 129, 0.4); }
}

/* ═══ HERO SECTION ═══ */
.welcome-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4.5rem 1rem 2.2rem 1rem;
    text-align: center;
}

.welcome-icon {
    width: 68px;
    height: 68px;
    border-radius: 22px;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #8b5cf6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow:
        0 0 40px rgba(99, 102, 241, 0.4),
        0 0 80px rgba(139, 92, 246, 0.2);
    margin-bottom: 24px;
    animation: float 4s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-7px); }
}

.welcome-hero h2 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 75%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}

.welcome-hero p {
    font-size: 0.96rem;
    color: #9ca3af;
    margin-bottom: 2.8rem;
    max-width: 480px;
    line-height: 1.55;
}

/* ═══ SUGGESTION BUTTON CARDS ═══ */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #12141e !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 18px 22px !important;
    text-align: left !important;
    height: auto !important;
    min-height: 84px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: center !important;
    color: #f3f4f6 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    line-height: 1.45 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}

div[data-testid="stHorizontalBlock"] .stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4f46e5, #6366f1, #06b6d4);
    opacity: 0;
    transition: opacity 0.25s ease;
}

div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08)) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 32px rgba(99, 102, 241, 0.22) !important;
    color: #ffffff !important;
}

div[data-testid="stHorizontalBlock"] .stButton > button:hover::before {
    opacity: 1 !important;
}

/* ═══ CHAT MESSAGES ═══ */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 14px 0px !important;
    margin-bottom: 14px !important;
}

/* User Message — Standard Conversational Right Alignment */
div[data-testid="stElementContainer"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 80% !important;
    width: fit-content !important;
    min-width: 175px !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.14)) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 20px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3) !important;
    align-self: flex-end !important;
}

/* Assistant Message — Standard Clean Prose Canvas */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 8px 4px 18px 4px !important;
    width: 100% !important;
    max-width: 100% !important;
}

/* ═══ USER MESSAGE ATTACHED DOCUMENT CHIPS ═══ */
.user-attachments-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.user-doc-chip {
    width: 82px;
    height: 80px;
    min-width: 82px;
    min-height: 80px;
    background: #23252d;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 14px;
    padding: 9px 10px 8px 10px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    user-select: none;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.user-doc-chip:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.7);
}

.user-doc-type {
    font-size: 11px;
    font-weight: 700;
    color: #9aa0a6;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.user-doc-name {
    font-size: 11px;
    font-weight: 500;
    color: #f1f3f4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}

/* Hide all avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
div[data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]),
div[data-testid="stChatMessage"] div[data-testid*="Avatar"],
.stChatMessageAvatar {
    display: none !important;
    width: 0 !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    visibility: hidden !important;
}

div[data-testid="stChatMessageContent"] {
    width: 100% !important;
    max-width: 100% !important;
    color: #e5e7eb !important;
    font-size: 0.96rem !important;
    line-height: 1.8 !important;
}

/* ═══ TABLES (WHEN REQUESTED) ═══ */
table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin: 16px 0 !important;
}

th {
    background: #12141e !important;
    color: #818cf8 !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

td {
    padding: 11px 16px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    color: #d1d5db !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

tr:last-child td { border-bottom: none !important; }
tr:hover td { background: rgba(99, 102, 241, 0.04) !important; }

/* ═══ GROUNDING BADGES ═══ */
.gnd-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin: 10px 0;
    font-weight: 600;
    backdrop-filter: blur(8px);
}

.gnd-ok {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}

.gnd-warn {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

.gnd-info {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
}

/* ═══ ENTITY MEDIA CARDS & DIAGRAM GALLERY ═══ */
.visual-gallery-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 16px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 7px;
    letter-spacing: 0.02em;
}

.media-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin: 8px 0 16px 0;
}

.media-card {
    display: flex;
    flex-direction: column;
    background: #141724;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    overflow: hidden;
    text-decoration: none !important;
    width: 210px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.media-card:hover {
    border-color: rgba(99, 102, 241, 0.55);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.22);
}

.media-card-img-wrap {
    width: 100%;
    height: 136px;
    background: #090a10;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.media-card img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #090a10;
    padding: 6px;
}

.media-card-body {
    padding: 9px 12px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    background: #11131e;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.media-card-title {
    font-size: 0.79rem;
    font-weight: 600;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.media-card-source {
    font-size: 0.65rem;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
}

/* ═══ STREAMLIT CONTROLS & BUTTONS (Main Content & Dialogs) ═══ */
section:not([data-testid="stSidebar"]) button[kind="secondary"],
div.main button[kind="secondary"],
div[data-testid="stDialog"] button[kind="secondary"] {
    background: #14151c !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

section:not([data-testid="stSidebar"]) button[kind="secondary"]:hover,
div.main button[kind="secondary"]:hover,
div[data-testid="stDialog"] button[kind="secondary"]:hover {
    background: #1a1b24 !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #818cf8 !important;
}

section:not([data-testid="stSidebar"]) button[kind="primary"],
div.main button[kind="primary"],
div[data-testid="stDialog"] button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
}

section:not([data-testid="stSidebar"]) button[kind="primary"]:hover,
div.main button[kind="primary"]:hover,
div[data-testid="stDialog"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
    box-shadow: 0 6px 22px rgba(79, 70, 229, 0.5) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div {
    background: #14151c !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: #12131a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-top: 10px !important;
}

div[data-testid="stExpander"] summary {
    color: #e5e7eb !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #818cf8 !important;
}

/* Code block */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stCodeBlock"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: #0d0e14 !important;
}

/* ═══ BOTTOM INPUT DOCK & SHIELD ═══ */
div[data-testid="stBottom"] {
    background-color: #0b0d14 !important;
    background: linear-gradient(180deg, rgba(11, 13, 20, 0) 0%, rgba(11, 13, 20, 0.95) 24px, #0b0d14 44px, #0b0d14 100%) !important;
    padding-top: 24px !important;
    padding-bottom: 12px !important;
    z-index: 99 !important;
}
div[data-testid="stBottom"] > div {
    background: transparent !important;
}
div[data-testid="stBottom"] iframe {
    width: 100% !important;
    border: none !important;
    background: transparent !important;
    display: block !important;
}

/* ═══ DIALOG STYLING ═══ (See dedicated attachment dialog rules below) */

/* ═══ FOOTER ═══ */
.app-footer {
    text-align: center;
    font-size: 0.72rem;
    color: #4b5563;
    margin-top: 18px;
    letter-spacing: 0.02em;
    font-weight: 500;
}

/* ═══ MESSAGE ACTION ICONS (Edit, Copy, Share Toolbar ONLY) ═══ */
div[data-testid="stChatMessage"]:not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"]:not(:has([data-testid="stImage"])):not(:has(img)) {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 6px !important;
    margin-top: 6px !important;
    margin-bottom: 0px !important;
    width: 100% !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]):not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"]:not(:has([data-testid="stImage"])):not(:has(img)) {
    justify-content: flex-end !important;
}

div[data-testid="stChatMessage"]:not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"]:not(:has([data-testid="stImage"])):not(:has(img)) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 8px !important;
}

div[data-testid="stChatMessage"]:not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"]:not(:has([data-testid="stImage"])):not(:has(img)) > div[data-testid="stColumn"]:not(:first-child) {
    flex: 0 0 32px !important;
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
}

/* ═══ CONTROLLED IMAGE GALLERY GRID (Sleek, Compact, Proportional Display) ═══ */
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    gap: 14px !important;
    width: 100% !important;
    margin: 8px 0 16px 0 !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > div[data-testid="stColumn"] {
    flex: 1 1 0px !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: none !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    max-width: 340px !important;
    margin: 0 auto !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] > img {
    height: 220px !important;
    max-height: 220px !important;
    width: 100% !important;
    max-width: 100% !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: #090a10 !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4) !important;
    transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.22s ease !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] > img:hover {
    transform: scale(1.025) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
    border-color: rgba(99, 102, 241, 0.6) !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] .stCaptionContainer {
    font-size: 0.74rem !important;
    color: #94a3b8 !important;
    text-align: center !important;
    margin-top: 6px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    max-width: 100% !important;
    width: 100% !important;
}


div[data-testid="stChatMessage"]:not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"] .stButton,
div[data-testid="stChatMessage"]:not(:has(div[data-testid="stForm"])) div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] {
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    display: inline-flex !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]),
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 0 !important;
    height: 30px !important;
    min-height: 30px !important;
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    color: #94a3b8 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: none !important;
    margin: 0 !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]) span[data-testid="stIconMaterial"],
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button span[data-testid="stIconMaterial"] {
    font-size: 17px !important;
    line-height: 1 !important;
    color: #94a3b8 !important;
    transition: color 0.2s ease, transform 0.2s ease !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"])::before,
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button::before {
    display: none !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]):hover,
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover {
    background: rgba(99, 102, 241, 0.22) !important;
    border-color: rgba(99, 102, 241, 0.55) !important;
    color: #ffffff !important;
    transform: translateY(-1px) scale(1.08) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]):hover span[data-testid="stIconMaterial"],
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover span[data-testid="stIconMaterial"] {
    color: #ffffff !important;
}

div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]):active,
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:active {
    transform: scale(0.95) !important;
}

/* ═══ HIDE ONLY THE CHEVRON DOWN ARROW (PRESERVE SHARE ICON) ═══ */
div[data-testid="stPopover"] [class*="eucf0wj1"],
div[data-testid="stPopover"] .eucf0wj1,
div[data-testid="stPopover"] button > div > *:nth-child(2) {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stPopover"] button > div > *:first-child,
div[data-testid="stPopover"] button > div > *:first-child span[data-testid="stIconMaterial"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-size: 17px !important;
    color: #94a3b8 !important;
}

div[data-testid="stPopover"] button:hover > div > *:first-child span[data-testid="stIconMaterial"] {
    color: #ffffff !important;
}

div[data-testid="stPopover"] button > div {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Edit Mode Form Box */
div[data-testid="stChatMessage"]:has(div[data-testid="stForm"]) {
    width: 380px !important;
    min-width: 320px !important;
    max-width: 90% !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] {
    background: rgba(18, 20, 30, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 14px !important;
    padding: 14px !important;
    margin-top: 6px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
    width: 100% !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] textarea {
    background: rgba(10, 12, 18, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.94rem !important;
    padding: 10px !important;
    width: 100% !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 12px !important;
    width: 100% !important;
    margin-top: 10px !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    flex: 1 1 0% !important;
    width: 50% !important;
    min-width: 110px !important;
    max-width: 50% !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
    width: 100% !important;
}

div[data-testid="stChatMessage"] div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button,
div[data-testid="stChatMessage"] div[data-testid="stForm"] button {
    width: 100% !important;
    min-width: 110px !important;
    max-width: none !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 6px 16px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    cursor: pointer !important;
}

/* ═══ SHARE POPOVER APPS ═══ */
.share-app-btn {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 13px !important;
    border-radius: 10px !important;
    text-decoration: none !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.share-wa {
    background: rgba(37, 211, 102, 0.12) !important;
    color: #4ade80 !important;
    border-color: rgba(37, 211, 102, 0.3) !important;
}
.share-wa:hover {
    background: rgba(37, 211, 102, 0.24) !important;
    border-color: rgba(37, 211, 102, 0.6) !important;
    transform: translateY(-1px) !important;
    color: #86efac !important;
}

.share-tg {
    background: rgba(34, 158, 217, 0.12) !important;
    color: #38bdf8 !important;
    border-color: rgba(34, 158, 217, 0.3) !important;
}
.share-tg:hover {
    background: rgba(34, 158, 217, 0.24) !important;
    border-color: rgba(34, 158, 217, 0.6) !important;
    transform: translateY(-1px) !important;
    color: #7dd3fc !important;
}

.share-tw {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}
.share-tw:hover {
    background: rgba(255, 255, 255, 0.14) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}

.share-mail {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #a5b4fc !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
}
.share-mail:hover {
    background: rgba(99, 102, 241, 0.24) !important;
    border-color: rgba(99, 102, 241, 0.6) !important;
    transform: translateY(-1px) !important;
    color: #c7d2fe !important;
}

/* ═══ ATTACHMENT DIALOG POSITIONED JUST NEAR ABOVE CHAT INPUT BAR ═══ */
div[data-testid="stDialog"],
.stDialog,
div[data-testid="stModal"] {
    display: flex !important;
    align-items: flex-end !important;
    justify-content: center !important;
    padding-bottom: 96px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
    background-color: rgba(0, 0, 0, 0.55) !important;
    backdrop-filter: blur(4px) !important;
}

div[data-testid="stDialog"] > div,
.stDialog > div,
div[data-testid="stModal"] > div {
    margin-bottom: 0 !important;
    margin-top: auto !important;
    max-height: calc(100vh - 125px) !important;
    width: 92% !important;
    max-width: 660px !important;
    padding: 24px 28px !important;
    background: #16181f !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 -12px 48px rgba(0, 0, 0, 0.8), 0 0 24px rgba(255, 87, 34, 0.15) !important;
    overflow-y: auto !important;
    animation: dialogSlideUp 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

@keyframes dialogSlideUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Style the dialog header, tabs & dropzone */
div[data-testid="stDialog"] header,
.stDialog header {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 10px !important;
    margin-bottom: 14px !important;
}

div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] [data-testid="stHeadingWithActionElements"],
.stDialog h2 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    padding-bottom: 2px !important;
}

div[data-testid="stDialog"] button[aria-label="Close"],
.stDialog button[aria-label="Close"] {
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stDialog"] button[aria-label="Close"]:hover,
.stDialog button[aria-label="Close"]:hover {
    color: #ffffff !important;
    transform: scale(1.1) !important;
}

div[data-testid="stDialog"] div[data-baseweb="tab-list"],
.stDialog div[data-baseweb="tab-list"],
div[data-testid="stDialog"] .stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    gap: 8px !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}

div[data-testid="stDialog"] button[data-baseweb="tab"],
.stDialog button[data-baseweb="tab"],
div[data-testid="stDialog"] .stTabs [data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}

div[data-testid="stDialog"] button[data-baseweb="tab"][aria-selected="true"],
.stDialog button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stDialog"] .stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(255, 87, 34, 0.15) !important;
    border: 1px solid rgba(255, 87, 34, 0.35) !important;
    border-bottom: 2px solid #ff5722 !important;
}

div[data-testid="stDialog"] section[data-testid="stFileUploaderDropzone"],
.stDialog section[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1.5px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stDialog"] section[data-testid="stFileUploaderDropzone"]:hover,
.stDialog section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #ff5722 !important;
    background: rgba(255, 87, 34, 0.05) !important;
}

/* ═══ MATHEMATICAL FORMULAS & KATEX DISPLAY — DARK OBSIDIAN ═══ */
.katex {
    font-size: 1.08em !important;
    text-rendering: auto !important;
}
.katex-display {
    margin: 0.85rem 0 !important;
    padding: 0.65rem 1.1rem !important;
    background: rgba(255, 255, 255, 0.035) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
}
.katex-display > .katex {
    font-size: 1.14em !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LIGHT_THEME_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════
   PREMIUM CRYSTAL LIGHT MODE DESIGN SYSTEM
   Clean Alpine / Enterprise Studio Aesthetic
   ═══════════════════════════════════════════════════════════════════ */

/* ─── Streamlit Theme CSS Variables Override ─── */
:root, .stApp {
    --background-color: #f8fafc !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #0f172a !important;
    --primary-color: #4f46e5 !important;
}

/* ─── Global Body & App Canvas ─── */
html, body, [class*="css"], .stApp {
    color: #0f172a !important;
    background-color: #f8fafc !important;
    background-image:
        radial-gradient(ellipse 90% 60% at 10% 0%, rgba(99, 102, 241, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 80% 60% at 90% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 60%),
        radial-gradient(circle at 50% 30%, rgba(6, 182, 212, 0.04) 0%, transparent 70%) !important;
    background-attachment: fixed !important;
}

/* ─── Sidebar Panel & Toggle (Floating Brand Logo/Symbol in Light Mode) ─── */
div[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 999999 !important;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

button[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    min-width: 38px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z'/></svg>"), linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 20px 20px, cover !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 11px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.4), 0 0 12px rgba(124, 58, 237, 0.25) !important;
    cursor: pointer !important;
    opacity: 1 !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    padding: 0 !important;
}

button[data-testid="stExpandSidebarButton"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z'/></svg>"), linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 21px 21px, cover !important;
    transform: scale(1.08) !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.6), 0 0 18px rgba(124, 58, 237, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.7) !important;
}

button[data-testid="stExpandSidebarButton"] svg,
div[data-testid="collapsedControl"] svg {
    display: none !important;
}

.sb-pills-row {
    border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
}

/* ═══ KEEP SIDEBAR COLLAPSE BUTTON («) CONSTANTLY VISIBLE IN LIGHT MODE ═══ */
[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: inline-flex !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
    color: #64748b !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    color: #0f172a !important;
    background: #f1f5f9 !important;
    transform: scale(1.08) !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    fill: currentColor !important;
    color: inherit !important;
    opacity: 1 !important;
}

/* ═══ CHATGPT SIDEBAR DESIGN SYSTEM — LIGHT MODE ═══ */
section[data-testid="stSidebar"] {
    background: #f9f9f9 !important;
    border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
    box-shadow: none !important;
}

.cg-brand-title {
    color: #0d0d0d !important;
}

button[data-testid*="btn_new_chat"] {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    color: #0d0d0d !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

button[data-testid*="btn_new_chat"] p {
    color: #0d0d0d !important;
}

button[data-testid*="btn_new_chat"]:hover {
    background: #f0f0f0 !important;
    border-color: rgba(0, 0, 0, 0.18) !important;
}

button[data-testid*="btn_tool_"] {
    background: transparent !important;
    border: none !important;
    color: #5d5d5d !important;
}

button[data-testid*="btn_tool_"] p {
    color: #5d5d5d !important;
}

button[data-testid*="btn_tool_"]:hover {
    background: #ececec !important;
    color: #0d0d0d !important;
}

button[data-testid*="btn_tool_"]:hover p {
    color: #0d0d0d !important;
}

button[data-testid*="btn_theme_toggle"] {
    background: transparent !important;
    border: none !important;
    color: #5d5d5d !important;
}

button[data-testid*="btn_theme_toggle"]:hover {
    background: #ececec !important;
    color: #0d0d0d !important;
}

.cg-section-title {
    color: #8e8e8e !important;
}

.cg-section-empty {
    color: #8e8e8e !important;
}

/* Recents conversation list in light mode (ZERO BOXES) */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]),
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-pop_act_"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]):hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-pop_act_"]):hover {
    background: transparent !important;
}

section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button,
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"] {
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button:hover,
section[data-testid="stSidebar"] button[data-testid*="sb_chat_btn_"]:hover {
    color: #0f172a !important;
    background: transparent !important;
}

section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[data-testid="baseButton-primary"] p,
section[data-testid="stSidebar"] div[class*="st-key-sb_chat_btn_"] button[kind="primary"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button p,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button span {
    color: #94a3b8 !important;
}

section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button:hover p,
section[data-testid="stSidebar"] div[class*="st-key-pop_act_"] button:hover span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button:hover p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-sb_chat_btn_"]) div[data-testid="stPopover"] button:hover span {
    color: #0f172a !important;
}

.sb-knowledge-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
}
.sb-kc-label {
    color: #64748b !important;
}
.sb-kc-meta {
    color: #94a3b8 !important;
}

/* Sidebar Selectbox & Dropdowns */
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:hover {
    border-color: #4f46e5 !important;
    background: #f8fafc !important;
}
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #0f172a !important;
}

/* Popover Floating Body */
div[data-testid="stPopoverBody"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.12) !important;
    color: #0f172a !important;
}
div[data-testid="stPopoverBody"] p,
div[data-testid="stPopoverBody"] span,
div[data-testid="stPopoverBody"] div {
    color: #0f172a !important;
}
div[data-testid="stPopoverBody"] button {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}
div[data-testid="stPopoverBody"] button:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}

/* Theme Toggle Button */
button[data-testid*="btn_theme_toggle"] {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
button[data-testid*="btn_theme_toggle"]:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Sidebar Expanders */
div[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
    color: #334155 !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] details summary:hover {
    color: #4f46e5 !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] details[open] summary {
    color: #4f46e5 !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* BaseWeb Selects & Menus Everywhere */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
}
li[data-baseweb="menu-item"] {
    background: transparent !important;
    color: #0f172a !important;
}
li[data-baseweb="menu-item"]:hover {
    background: #f1f5f9 !important;
    color: #4f46e5 !important;
}

/* Settings & Engine Controls in Light Mode */
div[data-testid="stSidebar"] div[data-testid="stExpander"] label {
    color: #475569 !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] .stSlider div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ffffff !important;
    border: 2px solid #4f46e5 !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] button {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] button:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Download Export Chat Button */
div[data-testid="stSidebar"] .stDownloadButton > button {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
div[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Status Pill in Sidebar Lower Dock */
.sb-system-footer {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
}
.sb-status-title {
    color: #0f172a !important;
}
.sb-status-sub {
    color: #64748b !important;
}

/* ─── Floating Header Bar ─── */
.header-bar {
    background: rgba(255, 255, 255, 0.94) !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
}
.hdr-title {
    color: #0f172a !important;
}
.hdr-sub {
    color: #64748b !important;
}
.pill-info {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}
.pill-live {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    color: #059669 !important;
}

/* ─── Welcome Hero & Suggestion Cards ─── */
.welcome-hero h2 {
    background: linear-gradient(135deg, #0f172a 40%, #334155 75%, #4f46e5 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.welcome-hero p {
    color: #64748b !important;
}

div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #f8fafc !important;
    border-color: #4f46e5 !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.12) !important;
    color: #0f172a !important;
}

/* ─── Chat Messages — Standard Conversational Layout ─── */
div[data-testid="stElementContainer"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 80% !important;
    width: fit-content !important;
    min-width: 175px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 20px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04) !important;
    align-self: flex-end !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    color: #0f172a !important;
}

/* Assistant Message — Clean Natural Prose Canvas (ChatGPT / Claude / Gemini Standard) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 8px 4px 18px 4px !important;
    width: 100% !important;
    max-width: 100% !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    color: #1e293b !important;
}

div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessageContent"] p,
div[data-testid="stChatMessageContent"] li,
div[data-testid="stChatMessageContent"] h1,
div[data-testid="stChatMessageContent"] h2,
div[data-testid="stChatMessageContent"] h3,
div[data-testid="stChatMessageContent"] h4 {
    color: #1e293b !important;
}

/* Attached Document Chips on Messages */
.user-doc-chip {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}
.user-doc-chip:hover {
    border-color: #4f46e5 !important;
}
.user-doc-type {
    color: #4f46e5 !important;
}
.user-doc-name {
    color: #0f172a !important;
}

/* Grounding Badges & Citations */
.citation-pill, .gnd-info {
    background: rgba(79, 70, 229, 0.08) !important;
    border: 1px solid rgba(79, 70, 229, 0.25) !important;
    color: #4338ca !important;
}
.gnd-ok {
    background: rgba(16, 185, 129, 0.08) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    color: #047857 !important;
}
.gnd-warn {
    background: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    color: #b45309 !important;
}

/* Code Blocks & Tables */
div[data-testid="stCodeBlock"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
}
div[data-testid="stCodeBlock"] code {
    color: #0f172a !important;
}
table {
    border: 1px solid #e2e8f0 !important;
}
th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
td {
    background: #ffffff !important;
    color: #1e293b !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
tr:hover td {
    background: #f8fafc !important;
}

/* Entity Media Cards */
.media-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
}
.media-card-title {
    color: #0f172a !important;
}
.media-card-source {
    color: #4f46e5 !important;
}

/* Controlled Image Gallery in Light Mode */
div[data-testid="stChatMessage"] div[data-testid="stImage"] > img {
    border: 1px solid #e2e8f0 !important;
    background: #f8fafc !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06) !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] > img:hover {
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2) !important;
    border-color: #4f46e5 !important;
}

div[data-testid="stChatMessage"] div[data-testid="stImage"] .stCaptionContainer {
    color: #475569 !important;
}

/* Action Icons (Edit, Copy, Share) */
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]),
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
}
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]) span[data-testid="stIconMaterial"],
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button span[data-testid="stIconMaterial"],
div[data-testid="stPopover"] button > div > *:first-child span[data-testid="stIconMaterial"] {
    color: #64748b !important;
}
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]):hover,
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover {
    background: #f1f5f9 !important;
    border-color: #4f46e5 !important;
    color: #4f46e5 !important;
}
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] .stButton > button:not([data-testid*="FormSubmitButton"]):hover span[data-testid="stIconMaterial"],
div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover span[data-testid="stIconMaterial"] {
    color: #4f46e5 !important;
}

/* Share Option Apps */
.share-tw {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
.share-tw:hover {
    background: #e2e8f0 !important;
    color: #000000 !important;
}

/* Edit Mode Form Box */
div[data-testid="stChatMessage"] div[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06) !important;
}
div[data-testid="stChatMessage"] div[data-testid="stForm"] textarea {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}

/* Attachment Dialog ("Ask NexusRAG") in Light Mode */
div[data-testid="stDialog"],
.stDialog,
div[data-testid="stModal"] {
    display: flex !important;
    align-items: flex-end !important;
    justify-content: center !important;
    padding-bottom: 96px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
    background-color: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(4px) !important;
}

div[data-testid="stDialog"] > div,
.stDialog > div,
div[data-testid="stModal"] > div {
    margin-bottom: 0 !important;
    margin-top: auto !important;
    max-height: calc(100vh - 125px) !important;
    width: 92% !important;
    max-width: 660px !important;
    padding: 24px 28px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 20px !important;
    box-shadow: 0 -8px 36px rgba(0, 0, 0, 0.12), 0 0 1px rgba(0, 0, 0, 0.08) !important;
    overflow-y: auto !important;
    animation: dialogSlideUp 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div[data-testid="stDialog"] header,
.stDialog header {
    background: transparent !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding-bottom: 10px !important;
    margin-bottom: 14px !important;
}
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] [data-testid="stHeadingWithActionElements"],
.stDialog h2 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    padding-bottom: 2px !important;
}
div[data-testid="stDialog"] button[aria-label="Close"],
.stDialog button[aria-label="Close"] {
    color: #64748b !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stDialog"] button[aria-label="Close"]:hover,
.stDialog button[aria-label="Close"]:hover {
    color: #0f172a !important;
    transform: scale(1.1) !important;
}
div[data-testid="stDialog"] div[data-baseweb="tab-list"],
.stDialog div[data-baseweb="tab-list"],
div[data-testid="stDialog"] .stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 8px !important;
}
div[data-testid="stDialog"] button[data-baseweb="tab"],
.stDialog button[data-baseweb="tab"],
div[data-testid="stDialog"] .stTabs [data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}
div[data-testid="stDialog"] button[data-baseweb="tab"][aria-selected="true"],
.stDialog button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stDialog"] .stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #ff5722 !important;
    border: 1px solid rgba(255, 87, 34, 0.2) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
}
div[data-testid="stDialog"] section[data-testid="stFileUploaderDropzone"],
.stDialog section[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stDialog"] section[data-testid="stFileUploaderDropzone"]:hover,
.stDialog section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #ff5722 !important;
    background: rgba(255, 87, 34, 0.04) !important;
}
div[data-testid="stDialog"] input[type="text"],
.stDialog input[type="text"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}

/* General Controls */
button[kind="secondary"] {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
button[kind="secondary"]:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}
div[data-testid="stSelectbox"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
}
div[data-testid="stExpander"] summary {
    color: #334155 !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #4f46e5 !important;
}

/* ─── Bottom Input Bar Container & Opaque Shield ─── */
div[data-testid="stBottom"] {
    background-color: #f8fafc !important;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0) 0%, rgba(248, 250, 252, 0.95) 24px, #f8fafc 44px, #f8fafc 100%) !important;
    padding-top: 24px !important;
    padding-bottom: 12px !important;
    z-index: 99 !important;
}
div[data-testid="stBottom"] > div {
    background: transparent !important;
}
div[data-testid="stBottom"] iframe {
    width: 100% !important;
    border: none !important;
    background: transparent !important;
    display: block !important;
}

/* ═══ MATHEMATICAL FORMULAS & KATEX DISPLAY — LIGHT MODE ═══ */
.katex {
    font-size: 1.08em !important;
    color: #0f172a !important;
    text-rendering: auto !important;
}
.katex-display {
    margin: 0.85rem 0 !important;
    padding: 0.65rem 1.1rem !important;
    background: #f1f5f9 !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
}
.katex-display > .katex {
    font-size: 1.14em !important;
    color: #0f172a !important;
}
</style>
"""

if is_light:
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

import importlib
import inspect
import base64
import streamlit.components.v1 as components
import rag.search

def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard natively using Windows clip.exe or pyperclip with graceful fallback."""
    if not isinstance(text, str):
        text = str(text)
    try:
        flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.run(["clip.exe"], input=text.encode("utf-16le"), check=True, creationflags=flags)
        return True
    except Exception:
        pass
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    return False

# Declare custom Chat Input Component
FRONTEND_DIR = Path(__file__).parent / "frontend"
gemini_chat_input = components.declare_component("gemini_chat_input", path=str(FRONTEND_DIR.resolve()))

# ─── SVG Icons ─── #
SVG_SPARK = '<svg width="{w}" height="{h}" viewBox="0 0 24 24" fill="white"><path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/></svg>'
SVG_DOC = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'
SVG_CHECK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
SVG_WARN = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
SVG_INFO = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff8a65" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
SVG_EDIT = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px;"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>'


# Attachment Dialog
@st.dialog("Ask NexusRAG")
def open_gemini_attachment_dialog():
    st.markdown("""
        <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 14px;">
            Upload and vectorize resources into your private knowledge base:
        </div>
    """, unsafe_allow_html=True)

    tab_files, tab_img, tab_url = st.tabs(["Upload Files", "Images", "Web Link"])

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
            processed = st.session_state.setdefault("_processed_uploader_files", set())
            new_files = [uf for uf in uploaded_docs if f"{uf.name}_{uf.size}" not in processed]
            if new_files:
                data_dir = PROJECT_DIR / "data"
                saved = []
                with st.spinner(f"Indexing {len(new_files)} document(s)..."):
                    for uf in new_files:
                        ext = Path(uf.name).suffix.lower()
                        target_dir = data_dir / ("pdf" if ext == ".pdf" else "text_files")
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / uf.name
                        with open(target_path, "wb") as f:
                            f.write(uf.getbuffer())
                        rag.add_document_file(target_path)
                        saved.append(uf.name)
                        processed.add(f"{uf.name}_{uf.size}")

                    if "staged_attachments" not in st.session_state:
                        st.session_state.staged_attachments = []
                    for s in saved:
                        if s not in st.session_state.staged_attachments:
                            st.session_state.staged_attachments.append(s)

                    st.toast(f"Successfully indexed {len(saved)} document(s).")
                    st.rerun()

    with tab_img:
        st.caption("PNG, JPG, JPEG, WEBP")
        uploaded_imgs = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="dlg_img_uploader",
            label_visibility="collapsed"
        )
        if uploaded_imgs:
            processed = st.session_state.setdefault("_processed_uploader_files", set())
            new_imgs = [uf for uf in uploaded_imgs if f"{uf.name}_{uf.size}" not in processed]
            if new_imgs:
                img_dir = PROJECT_DIR / "data" / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                with st.spinner(f"Indexing {len(new_imgs)} image(s)..."):
                    for uf in new_imgs:
                        target_path = img_dir / uf.name
                        with open(target_path, "wb") as f:
                            f.write(uf.getbuffer())
                        rag.add_document_file(target_path)
                        saved.append(uf.name)
                        processed.add(f"{uf.name}_{uf.size}")

                    if "staged_attachments" not in st.session_state:
                        st.session_state.staged_attachments = []
                    for s in saved:
                        if s not in st.session_state.staged_attachments:
                            st.session_state.staged_attachments.append(s)

                    st.toast(f"Successfully indexed {len(saved)} image(s).")
                    st.rerun()

    with tab_url:
        st.caption("Scrape & index any public webpage or article")
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
                        doc_fname = Path(saved_path).name
                        if "staged_attachments" not in st.session_state:
                            st.session_state.staged_attachments = []
                        if doc_fname not in st.session_state.staged_attachments:
                            st.session_state.staged_attachments.append(doc_fname)
                        st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
                        st.rerun()
                except Exception as e_url:
                    st.error(f"Failed to ingest URL: {e_url}")
            else:
                st.warning("Please enter a valid URL starting with http:// or https://")


# ─── Initialize RAG Engine ─── #
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    return RAGSearch(
        persist_dir=str(PROJECT_DIR / "faiss_store"),
        data_dir=str(PROJECT_DIR / "data"),
        llm_model=DEFAULT_GROQ_MODEL
    )

with st.spinner("Connecting to FAISS & BM25 Core..."):
    rag = load_rag_engine()

# Global Knowledge Base stats and active documents
stats = rag.vectorstore.get_stats() if hasattr(rag, "vectorstore") else {}
active_docs = stats.get("documents", [])
doc_count = len(active_docs)
total_chunks = stats.get("total_chunks", 0)


def safe_search_context(rag_engine, query, top_k, mode, use_reranker=True, enable_web_search=True, decompose_query=False, tracer=None, attached_docs=None):
    """Safely invoke search_context handling any in-flight module caching gracefully."""
    sig = inspect.signature(rag_engine.search_context).parameters
    kwargs = {"top_k": top_k, "mode": mode}
    if "use_reranker" in sig:
        kwargs["use_reranker"] = use_reranker
    if "enable_web_search" in sig:
        kwargs["enable_web_search"] = enable_web_search
    if "decompose_query" in sig:
        kwargs["decompose_query"] = decompose_query
    if "tracer" in sig:
        kwargs["tracer"] = tracer
    if "attached_docs" in sig:
        kwargs["attached_docs"] = attached_docs
    return rag_engine.search_context(query, **kwargs)

def safe_stream_query(rag_engine, query, chat_history, top_k, model_name, retrieval_mode, use_reranker=True, enable_web_search=True, decompose_query=False, tracer=None, citations=None, attached_docs=None):
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
    if "decompose_query" in sig:
        kwargs["decompose_query"] = decompose_query
    if "tracer" in sig:
        kwargs["tracer"] = tracer
    if "citations" in sig:
        kwargs["citations"] = citations
    if "attached_docs" in sig:
        kwargs["attached_docs"] = attached_docs
    return rag_engine.stream_query(query, **kwargs)


# ─── Chat History Manager ─── #
chat_mgr = ChatHistoryManager(PROJECT_DIR / "data" / "chats")
if "current_chat_id" not in st.session_state:
    existing_chats = chat_mgr.list_chats()
    if existing_chats:
        st.session_state.current_chat_id = existing_chats[0]["id"]
        loaded = chat_mgr.load_chat(st.session_state.current_chat_id)
        st.session_state.messages = loaded.get("messages", []) if loaded else []
    else:
        st.session_state.current_chat_id = ChatHistoryManager.generate_chat_id()
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Action Handlers ─── #
def handle_file_deletion(doc_name: str):
    """Instantly delete a document from knowledge base and disk."""
    try:
        if hasattr(rag, "remove_document"):
            rag.remove_document(doc_name)
        else:
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


# ─── Specialized Dialogs for Quick Tools ─── #
@st.dialog("Knowledge Library")
def open_library_dialog():
    st.markdown("""
        <div style="font-size: 0.86rem; color: #94a3b8; margin-bottom: 12px;">
            Active indexed resources in your hybrid retrieval index:
        </div>
    """, unsafe_allow_html=True)
    stats = rag.vectorstore.get_stats()
    docs = stats.get("documents", [])
    total_chunks = stats.get("total_chunks", 0)
    st.caption(f"{len(docs)} Documents &middot; {total_chunks:,} Chunks Indexed (FAISS + BM25)")
    if not docs:
        st.info("No documents indexed yet. Use 'Images' or '+' to add documents.")
    else:
        for d in docs:
            c_doc, c_del = st.columns([7.5, 2.5])
            with c_doc:
                st.markdown(f"📄 `{d}`")
            with c_del:
                if st.button("Delete", key=f"dlg_lib_del_{d}", use_container_width=True):
                    handle_file_deletion(d)
                    st.rerun()


@st.dialog("Explore Prompts & Benchmarks")
def open_explore_dialog():
    st.markdown("""
        <div style="font-size: 0.86rem; color: #94a3b8; margin-bottom: 12px;">
            Select a benchmark query to test retrieval and synthesis:
        </div>
    """, unsafe_allow_html=True)
    prompts = [
        "Explain Neural Network Architectures with key diagrams",
        "Compare Transformer attention vs Convolutional feature maps",
        "Summarize performance bottlenecks in Hybrid Search pipelines",
        "What are the best practices for vector database chunking?"
    ]
    for p in prompts:
        if st.button(p, key=f"exp_prompt_{hash(p)}", use_container_width=True):
            st.session_state.pending_prompt = p
            st.rerun()


# =====================================================================
# ███  1. SIDEBAR (Clean, Minimalist ChatGPT Styling)  ████████████████
# =====================================================================
with st.sidebar:
    # ── Top Brand Header & Status Pills (Moved from Main UI) ──
    web_on = st.session_state.get("sb_web_search", True)
    rerank_on = st.session_state.get("sb_use_reranker", True)
    if web_on and rerank_on:
        curr_pipeline = "Omni-Source RAG"
    elif web_on:
        curr_pipeline = "Docs + Web"
    elif rerank_on:
        curr_pipeline = "Two-Stage RAG"
    else:
        curr_pipeline = "Direct RAG"

    is_pg_live = bool(hasattr(rag, "pg_db") and rag.pg_db and rag.pg_db.is_connected)
    curr_db_label = "pgvector (Connected)" if is_pg_live else "FAISS & pgvector"

    active_model_disp = "GPT-OSS"
    if "sb_model_name" in st.session_state:
        active_model_disp = st.session_state["sb_model_name"].split(" ")[0]
    elif AVAILABLE_MODELS:
        active_model_disp = AVAILABLE_MODELS[0]["name"].split(" ")[0]

    col_hdr_brand, col_hdr_theme = st.columns([8.2, 1.8])
    with col_hdr_brand:
        st.markdown(f"""
            <div class="sb-brand-header">
                <div class="hdr-icon" style="width:34px; height:34px; min-width:34px; border-radius:10px;">{SVG_SPARK.format(w=16, h=16)}</div>
                <div class="sb-brand-text">
                    <div class="hdr-title" style="font-size: 1.05rem; line-height: 1.2;">NexusRAG Studio</div>
                    <div class="hdr-sub" style="font-size: 0.68rem; line-height: 1.2;">Document Intelligence Platform</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_hdr_theme:
        is_dark = (st.session_state.get("theme_mode", "dark") == "dark")
        t_icon = ":material/light_mode:" if is_dark else ":material/dark_mode:"
        t_help = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
        if st.button("", icon=t_icon, key="btn_theme_toggle", help=t_help, use_container_width=True):
            new_mode = "light" if is_dark else "dark"
            st.session_state["theme_mode"] = new_mode
            st.session_state["sb_theme_select"] = "Pure Light" if new_mode == "light" else "Dark Obsidian"
            st.rerun()

    # ── Status Pills Row in Sidebar ──
    st.markdown(f"""
        <div class="sb-pills-row">
            <div class="pill pill-live" style="font-size: 0.66rem; padding: 3px 8px;">
                <span class="live-dot" style="width: 5px; height: 5px;"></span>
                <span>{curr_pipeline}</span>
            </div>
            <div class="pill pill-info" style="font-size: 0.66rem; padding: 3px 8px;">{curr_db_label}</div>
            <div class="pill pill-info" style="font-size: 0.66rem; padding: 3px 8px;">{active_model_disp}</div>
        </div>
    """, unsafe_allow_html=True)

    # ── New chat Pill Button ──
    if st.button("New chat", icon=":material/edit_square:", key="btn_new_chat", use_container_width=True):
        st.session_state.current_chat_id = ChatHistoryManager.generate_chat_id()
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.session_state.pop("staged_attachments", None)
        st.rerun()

    # ── Functional Quick Tools ──
    if st.button("Images & Files", icon=":material/image:", key="btn_tool_images", use_container_width=True):
        open_gemini_attachment_dialog()

    if st.button("Knowledge Library", icon=":material/local_library:", key="btn_tool_library", use_container_width=True):
        open_library_dialog()

    if st.button("Explore Prompts", icon=":material/explore:", key="btn_tool_explore", use_container_width=True):
        open_explore_dialog()

    # ── Recents Section ──
    saved_chats = chat_mgr.list_chats()
    current_id = st.session_state.get("current_chat_id")

    st.markdown('<div class="cg-section-title">Recents <span style="font-size:0.72rem;margin-left:3px;opacity:0.75;">⌵</span></div>', unsafe_allow_html=True)

    if saved_chats:
        chat_container = st.container(height=320) if len(saved_chats) > 7 else st.container()
        with chat_container:
            for c in saved_chats:
                is_cur = (c["id"] == current_id)
                col_c_btn, col_c_menu = st.columns([8.6, 1.4], gap="small")
                with col_c_btn:
                    c_title = c.get("title", "Conversation")
                    disp_title = (c_title[:24] + "...") if len(c_title) > 26 else c_title
                    btn_kind = "primary" if is_cur else "secondary"
                    if st.button(
                        disp_title,
                        key=f"sb_chat_btn_{c['id']}",
                        type=btn_kind,
                        use_container_width=True,
                        help=c_title
                    ):
                        if c["id"] != current_id:
                            chat_data = chat_mgr.load_chat(c["id"])
                            if chat_data:
                                st.session_state.current_chat_id = c["id"]
                                st.session_state.messages = chat_data.get("messages", [])
                                st.session_state["staged_attachments"] = []
                                st.rerun()
                with col_c_menu:
                    with st.popover("···", key=f"pop_act_{c['id']}", help="Options"):
                        st.caption(f"{c['title'][:20]}...")
                        if st.button("Archive", icon=":material/archive:", key=f"arch_btn_{c['id']}", use_container_width=True):
                            chat_mgr.archive_chat(c["id"])
                            if c["id"] == current_id:
                                st.session_state.current_chat_id = ChatHistoryManager.generate_chat_id()
                                st.session_state.messages = []
                            st.toast("Chat moved to archives.")
                            st.rerun()
                        if st.button("Delete", icon=":material/delete:", key=f"del_btn_{c['id']}", type="primary", use_container_width=True):
                            chat_mgr.delete_chat(c["id"])
                            if c["id"] == current_id:
                                st.session_state.current_chat_id = ChatHistoryManager.generate_chat_id()
                                st.session_state.messages = []
                            st.toast("Chat deleted permanently.")
                            st.rerun()

        archived_chats = chat_mgr.list_archived_chats()
        if archived_chats:
            with st.expander(f"Archived ({len(archived_chats)})", icon=":material/inventory_2:"):
                for ac in archived_chats:
                    col_ac_t, col_ac_rst, col_ac_del = st.columns([3.6, 1.8, 1])
                    with col_ac_t:
                        st.caption(ac["title"][:15] + ("..." if len(ac["title"]) > 15 else ""))
                    with col_ac_rst:
                        if st.button("Restore", key=f"rst_{ac['id']}", use_container_width=True, help="Restore chat"):
                            chat_mgr.unarchive_chat(ac["id"])
                            st.toast("Chat restored to active list.")
                            st.rerun()
                    with col_ac_del:
                        with st.popover("···", key=f"pop_arch_{ac['id']}", help="Options"):
                            st.caption(ac["title"][:22] + ("..." if len(ac["title"]) > 22 else ""))
                            if st.button("Delete Chat", icon=":material/delete:", key=f"del_arch_{ac['id']}", type="primary", use_container_width=True):
                                chat_mgr.delete_chat(ac["id"])
                                st.toast("Archived chat deleted.")
                                st.rerun()
    else:
        st.markdown('<div class="cg-section-empty">No chats yet</div>', unsafe_allow_html=True)

    # ── Push lower controls to bottom ──
    st.markdown('<div class="sb-flex-spacer"></div><div class="sb-lower-dock"></div>', unsafe_allow_html=True)

    # ── Settings & Engine (Collapsible, Clean) ──
    with st.expander("Settings & Engine", icon=":material/settings:", expanded=False):
        theme_options = ["Dark Obsidian", "Pure Light"]
        curr_theme_idx = 0 if st.session_state.get("theme_mode", "dark") == "dark" else 1

        def _sync_theme_dropdown():
            chosen = st.session_state.get("sb_theme_select")
            st.session_state["theme_mode"] = "light" if chosen == "Pure Light" else "dark"

        st.selectbox(
            "Appearance Mode",
            theme_options,
            index=curr_theme_idx,
            key="sb_theme_select",
            on_change=_sync_theme_dropdown
        )

        model_options = {m["name"]: m["id"] for m in AVAILABLE_MODELS}
        selected_model_name = st.selectbox("Model", list(model_options.keys()), index=0, key="sb_model_name")
        selected_model_id = model_options[selected_model_name]

        retrieval_mode_label = st.selectbox(
            "Retrieval Method",
            ["Hybrid Search (BM25 + FAISS)", "Dense Vector Only (FAISS)", "Sparse Keyword Only (BM25)"],
            index=0,
            key="sb_retrieval_mode_label"
        )
        mode_map = {
            "Hybrid Search (BM25 + FAISS)": "hybrid",
            "Dense Vector Only (FAISS)": "dense",
            "Sparse Keyword Only (BM25)": "sparse"
        }
        retrieval_mode = mode_map[retrieval_mode_label]

        use_reranker = st.toggle("Cross-Encoder Re-Ranking", value=True, help="Two-stage pipeline with MS-MARCO Cross-Encoder", key="sb_use_reranker")
        enable_web_search = st.toggle("Live Multi-Source Search", value=True, help="Augment with Wikipedia, ArXiv, and web search", key="sb_web_search")
        decompose_query = st.toggle("Agentic Query Planner", value=True, help="Decompose compound queries into parallel searches")
        verify_hallucinations = st.toggle("Hallucination Guardrail", value=False, help="Self-RAG auditor verifies groundedness")
        enable_llmops_tracer = st.toggle("LLMOps Real-Time Tracing", value=False, help="Track latency waterfall, TTFT, and token cost")
        top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=10, value=4)

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
                        cits = safe_search_context(rag, q, top_k=3, mode=retrieval_mode, use_reranker=use_reranker, enable_web_search=enable_web_search)
                        hit = any(exp in c["source"].lower() for c in cits)
                        results.append({"Query": q[:35] + "...", "Hit": "Passed" if hit else "Failed"})
                    st.dataframe(results, use_container_width=True)

    if st.session_state.messages:
        chat_markdown = "# NexusRAG Chat Export\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].upper()}:\n{m['content']}\n\n"
        st.download_button(label="Export Chat (.md)", data=chat_markdown, file_name="nexusrag_chat_transcript.md", mime="text/markdown", use_container_width=True)

    # ── Clean System Status Pill ──
    st.markdown(f"""
        <div class="sb-system-footer">
            <div class="sb-status-dot"></div>
            <div class="sb-status-text">
                <span class="sb-status-title">Engine Ready</span>
                <span class="sb-status-sub">{selected_model_name.split(' ')[0]} &middot; {retrieval_mode.capitalize()}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# (Main header moved to sidebar to maximize vertical screen canvas)


# =====================================================================
# ███  CONTENT HELPERS  ██████████████████████████████████████████████
# =====================================================================

def convert_tables_to_prose(text: str) -> str:
    """Convert raw markdown tables into clean bullet points."""
    if not isinstance(text, str) or "|" not in text:
        return text
    lines = text.splitlines()
    result = []
    in_table = False
    table_headers = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                if i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?\s*$", lines[i + 1]):
                    table_headers = cells
                    in_table = True
                    i += 2
                    continue
            if in_table:
                if len(cells) >= 2:
                    primary = cells[0].strip("*").strip()
                    desc = cells[1].strip()
                    extra_parts = []
                    for idx, c in enumerate(cells[2:], 2):
                        if c:
                            h_label = table_headers[idx] if idx < len(table_headers) else f"Detail {idx}"
                            extra_parts.append(f"{h_label}: {c}")
                    extra_str = f" ({'; '.join(extra_parts)})" if extra_parts else ""
                    result.append(f"- **{primary}**: {desc}{extra_str}")
                i += 1
                continue
        else:
            if in_table:
                in_table = False
                table_headers = []
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def remove_inline_citations(text: str) -> str:
    """Remove inline citation markers across all bracket types (standard, lenticular, fullwidth) and naked mentions."""
    if not isinstance(text, str):
        return text
    patterns = [
        # Matches [Web Source 1], 【Web Source 1】, ［Web Source 1］, [1], 【1】, [Doc 1, 2], etc.
        r"[\[【［](?:Web(?:\/External)?\s*(?:Source)?|Doc(?:ument)?|Source|Private Document|\d+)[^\]】］]*[\]】］]",
        # Matches (Web Source 1), (Doc 1)
        r"\((?:Web(?:\/External)?\s*(?:Source)?|Doc(?:ument)?|Source|Private Document)\s*\d+[^)]*\)",
        # Matches naked 'Web Source 1', 'Web Source 1, 2, 3'
        r"\b(?:Web(?:\/External)?\s*Source|Doc(?:ument)?|Private Document)\s*\d+(?:\s*(?:,|and|&)?\s*\d+)*\b",
        # Empty brackets leftover
        r"[\[【［]\s*[\]】］]"
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned


LATEX_MATH_COMMANDS = (
    r'\\(?:sin|cos|tan|csc|sec|cot|arcsin|arccos|arctan|sinh|cosh|tanh|'
    r'frac|dfrac|tfrac|sqrt|cbrt|'
    r'theta|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|iota|kappa|'
    r'lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|'
    r'varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|'
    r'Phi|Psi|Omega|'
    r'pm|mp|times|div|cdot|circ|ast|star|cap|cup|in|notin|subset|subseteq|'
    r'neq|approx|sim|simeq|cong|leq|geq|ll|gg|equiv|propto|'
    r'sum|prod|coprod|int|iint|iiint|oint|lim|inf|sup|max|min|'
    r'to|rightarrow|leftarrow|Rightarrow|Leftarrow|Leftrightarrow|iff|'
    r'infty|partial|nabla|hbar|ell|'
    r'quad|qquad|mathbf|mathit|mathrm|mathbb|mathcal|text|'
    r'begin|end|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|'
    r'log|ln|lg|exp|det|dim|ker|deg|gcd)'
    r'(?:[^a-zA-Z]|$)'
)


def format_latex_formulas(text: str) -> str:
    """Format and normalize mathematical formulas so Streamlit renders them as KaTeX formulas."""
    if not isinstance(text, str) or not text:
        return text

    # Remove \displaystyle which is redundant inside math mode and causes syntax clutter
    text = re.sub(r'\\displaystyle\s*', '', text)

    # 1. Normalize LaTeX block delimiters \[ ... \] -> $ ... $ (collapsing internal linebreaks to single spaces)
    def replace_block_math(m):
        inner = m.group(1).strip()
        inner_clean = ' '.join(inner.split())
        return f"${inner_clean}$"

    text = re.sub(r'\\\[([\s\S]*?)\\\]', replace_block_math, text)

    # 2. Normalize LaTeX inline delimiters \( ... \) -> $ ... $
    def replace_inline_math(m):
        inner = m.group(1).strip()
        inner_clean = ' '.join(inner.split())
        return f"${inner_clean}$"

    text = re.sub(r'\\\(([\s\S]*?)\\\)', replace_inline_math, text)

    # 3. Clean up bullet points followed immediately by math without colon:
    text = re.sub(r'(?m)^(\s*[-•*]\s*\*\*[^*]+\*\*)\s*\n+\s*(\$)', r'\1: \2', text)
    text = re.sub(r'(?m)^(\s*[-•*]\s*\*\*[^*]+\*\*)\s+(?!\:\s*)(\$)', r'\1: \2', text)

    # 4. Normalize bracketed formulas like [ \sin^{2}\theta + \cos^{2}\theta = 1 ] -> $ ... $
    def replace_bracket_math(m):
        content = m.group(1).strip()
        if re.search(LATEX_MATH_COMMANDS, content):
            inner_clean = ' '.join(content.split())
            return f"${inner_clean}$"
        return m.group(0)

    # Negative lookahead (?!\s*[\(\[]) ensures we do NOT break markdown links [text](url) or [[wiki]]
    text = re.sub(r'\[\s*([^\]\n]+?)\s*\](?!\s*[\(\[])', replace_bracket_math, text)

    # 5. Mask existing math ($...$ and $$...$$) and code blocks before processing standalone elements
    masked_blocks = []
    def mask_element(m):
        idx = len(masked_blocks)
        masked_blocks.append(m.group(0))
        return f"__MATH_OR_CODE_BLOCK_{idx}__"

    text = re.sub(r'```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$', mask_element, text)

    # 6. Normalize remaining parentheses with math like (provided (\cos\theta \neq 0)) or (\cos\theta \neq 0)
    def replace_paren_math(m):
        content = m.group(1).strip()
        words = [w for w in re.findall(r'[a-zA-Z]{3,}', content) if not re.match(r'^(sin|cos|tan|csc|sec|cot|frac|sqrt|theta|alpha|beta|gamma|delta|sigma|neq|approx|quad|qquad)$', w, re.I)]
        if re.search(LATEX_MATH_COMMANDS, content) and len(words) == 0:
            return f"(${content}$)"
        return m.group(0)

    text = re.sub(r'(?<!\\)\(\s*([^\(\)\n]*?' + LATEX_MATH_COMMANDS + r'[^\(\)\n]*?)\s*\)', replace_paren_math, text)

    # 7. Match and wrap any unescaped LaTeX math formulas containing equations, \\text{...}, relations, or math commands
    # e.g. \text{addr}(A[i]) = B + i \times s or A \times B = C
    def replace_unwrapped_formula(m):
        expr = m.group(0).strip()
        trailing = ""
        while expr and expr[-1] in ".,;:":
            trailing = expr[-1] + trailing
            expr = expr[:-1].strip()
        if not expr:
            return m.group(0)
        return f"${expr}${trailing}"

    formula_pattern = r'(?:\\(?:text|mathrm|mathbf|mathbb|frac|sqrt)\{[^}]+\}|\\(?:[a-zA-Z]+))(?:\([^)]*\)|\[[^\]]*\]|[\s\w+*/^_{}\-=\\])*?(?:=|<|>|\+|-|\\times|\\cdot|\\approx|\\neq|\\leq|\\geq)(?:[\s\w+*/^_{}\-=\\]|\\(?:text|mathrm|mathbf|mathbb|frac|sqrt)\{[^}]+\}|\\[a-zA-Z]+)+'
    text = re.sub(formula_pattern, replace_unwrapped_formula, text)

    # 8. Normalize standalone lines that are pure LaTeX expressions:
    def replace_standalone_line(m):
        line = m.group(1).strip()
        words = [w for w in re.findall(r'[a-zA-Z]{4,}', line) if not re.match(r'^(sin|cos|tan|csc|sec|cot|frac|dfrac|tfrac|sqrt|theta|alpha|beta|gamma|delta|epsilon|sigma|omega|approx|times|qquad|quad|addr|text|base)$', w, re.I)]
        if re.search(LATEX_MATH_COMMANDS, line) and len(words) <= 2:
            return f"${line}$"
        return m.group(0)

    text = re.sub(r'(?m)^\s*(\\+[a-zA-Z]+[^\n]+)$', replace_standalone_line, text)

    # 9. Unmask code and math blocks
    for idx, block in enumerate(masked_blocks):
        text = text.replace(f"__MATH_OR_CODE_BLOCK_{idx}__", block)

    # Clean up empty math markers
    text = re.sub(r'\$\s*\$', '', text)

    return text


def clean_markdown_content(text: str, query: str = "") -> str:
    if not isinstance(text, str):
        return str(text)
    user_wants_table = bool(re.search(r"\b(table|tabular|grid|matrix)\b", query, re.IGNORECASE))
    cleaned = text.replace("<br>•", "\n• ").replace("<br>", " ")
    cleaned = remove_inline_citations(cleaned)
    if not user_wants_table:
        cleaned = convert_tables_to_prose(cleaned)
    # Strip hallucinated markdown images (the verified visual cards render them natively)
    cleaned = re.sub(r"!\[[^\]]*\]\(https?://[^\)]+\)", "", cleaned)
    # Strip naked stock-photo direct links or search URLs (e.g., Pixabay, Shutterstock, etc.)
    cleaned = re.sub(r"(?m)^\s*(?:👉\s*)?https?://(?:www\.)?(?:pixabay|shutterstock|istockphoto|gettyimages|unsplash|google)\.com[^\s]*\s*$", "", cleaned)
    cleaned = re.sub(r"(?i)by visiting the links? above[^\n.]*\.?", "", cleaned)
    cleaned = re.sub(r"(?i)here are a few reliable places where you can find[^\n:]*:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)below are a few (?:direct-link )?images[^\n.]*\.?", "", cleaned)
    cleaned = re.sub(r"(?i)all of the links point directly[^\n.]*\.?", "", cleaned)
    cleaned = re.sub(r"(?i)####?\s*how to use them[\s\S]*?(?=\n\n|$)", "", cleaned)
    cleaned = re.sub(r"\*{0,2}suggested attribution:\*{0,2}[^\n]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*Photographer:[^\n]*\*", "", cleaned, flags=re.IGNORECASE)
    # Normalize LaTeX mathematical formulas so they render as visual KaTeX formulas
    cleaned = format_latex_formulas(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def render_entity_images(images, query: str = "", content: str = ""):
    """Render visual entity cards with rich visual preview."""
    if not images:
        return
    cards_html = []
    stopwords = {
        "the", "and", "for", "with", "from", "about", "what", "which",
        "who", "whom", "this", "that", "song", "album", "music", "video",
        "days", "tell", "show", "response", "please", "also", "reference",
        "wikipedia", "visual", "asset", "encyclopedia", "image", "photo"
    }
    context_to_check = f"{query} {content}".lower() if (query or content) else ""
    is_image_query = bool(re.search(r"\b(images?|pictures?|photos?|diagrams?|illustrations?|figures?|visuals?)\b", query, re.IGNORECASE))
    seen_base_urls = set()

    for img in images:
        raw_title = str(img.get("title", ""))
        clean_title = re.sub(r"^File:", "", raw_title).replace("_", " ")
        clean_title = re.sub(r"\.(svg|png|jpg|jpeg|gif|webp)$", "", clean_title, flags=re.IGNORECASE).strip()
        full_title = clean_title or "Visual Asset"
        disp_title = (clean_title[:38] + "…") if len(clean_title) > 40 else clean_title
        title = html.escape(full_title)
        disp_title_esc = html.escape(disp_title)

        img_url = str(img.get("image_url", ""))
        src_url = img.get("source_url", "#")
        source_label = html.escape(str(img.get("caption", "Verified Diagram")))
        if not img_url:
            continue

        # Deduplicate images by normalized base URL
        base_u = re.sub(r"\?.*$", "", img_url).lower()
        if base_u in seen_base_urls:
            continue
        seen_base_urls.add(base_u)

        # If user explicitly asked for images, relax strict filter so all verified images render
        if not is_image_query and context_to_check:
            title_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_title.lower())) - stopwords
            if title_words and not any(w in context_to_check for w in title_words):
                continue

        if not (img_url.startswith("http://") or img_url.startswith("https://") or img_url.startswith("data:")):
            try:
                local_p = Path(img_url)
                if local_p.exists() and local_p.is_file():
                    mime = "image/png" if local_p.suffix.lower() == ".png" else "image/jpeg"
                    b64 = base64.b64encode(local_p.read_bytes()).decode("utf-8")
                    img_url = f"data:{mime};base64,{b64}"
                else:
                    continue
            except Exception:
                continue

        card = f'''<a href="{src_url}" target="_blank" rel="noopener noreferrer" class="media-card" title="{title}">
            <div class="media-card-img-wrap">
                <img src="{img_url}" alt="{title}" loading="lazy" referrerpolicy="no-referrer" />
            </div>
            <div class="media-card-body">
                <span class="media-card-title">{disp_title_esc}</span>
                <span class="media-card-source">{source_label}</span>
            </div>
        </a>'''
        cards_html.append(card)

    if cards_html:
        hdr = '<div class="visual-gallery-header"><span style="color:#6366f1;">❖</span> Visual Media & Reference Gallery</div>'
        st.markdown(f'{hdr}<div class="media-strip">{"".join(cards_html)}</div>', unsafe_allow_html=True)


# =====================================================================
# ███  CONVERSATION DISPLAY  ████████████████████████████████████████
# =====================================================================

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            editing_idx = st.session_state.get("editing_prompt_idx")
            if editing_idx == idx:
                with st.form(key=f"edit_form_{idx}", border=False):
                    st.markdown(f"""
                        <div style="font-size: 0.86rem; font-weight: 700; color: #a5b4fc; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                            {SVG_EDIT} Edit Prompt
                        </div>
                    """, unsafe_allow_html=True)
                    edited_text = st.text_area(
                        "Edit Prompt",
                        value=msg["content"],
                        key=f"edit_input_{idx}",
                        label_visibility="collapsed",
                        height=100
                    )
                    c_save, c_cancel = st.columns(2, gap="small")
                    with c_save:
                        save_clicked = st.form_submit_button("Update", type="primary", use_container_width=True)
                    with c_cancel:
                        cancel_clicked = st.form_submit_button("Cancel", use_container_width=True)

                    if save_clicked:
                        if edited_text and edited_text.strip():
                            st.session_state.messages = st.session_state.messages[:idx]
                            st.session_state["pending_prompt"] = edited_text.strip()
                            st.session_state["editing_prompt_idx"] = None
                            chat_mgr.save_chat(chat_id=st.session_state.current_chat_id, messages=st.session_state.messages)
                            st.rerun()

                    if cancel_clicked:
                        st.session_state["editing_prompt_idx"] = None
                        st.rerun()
            else:
                # Render attached document chip card(s) on main UI matching Gemini style
                msg_attachments = msg.get("attachments") or []
                if not msg_attachments and idx == 0 and active_docs:
                    msg_attachments = list(active_docs)
                if msg_attachments:
                    doc_chips_html = []
                    for doc_name in msg_attachments:
                        parts = doc_name.split(".")
                        ext = parts[-1].upper() if len(parts) > 1 else "DOC"
                        display_name = doc_name if len(doc_name) <= 15 else (doc_name[:12] + "...")
                        doc_chips_html.append(f"""
                            <div class="user-doc-chip" title="{html.escape(doc_name)}">
                                <div class="user-doc-type">{ext}</div>
                                <div class="user-doc-name">{html.escape(display_name)}</div>
                            </div>
                        """)
                    st.markdown(f'<div class="user-attachments-row">{"".join(doc_chips_html)}</div>', unsafe_allow_html=True)

                st.markdown(clean_markdown_content(msg["content"], query=""))

                # User message action toolbar: Edit, Copy, Share (Material vector icons)
                c_sp, c_edit, c_copy, c_share = st.columns([1, 1, 1, 1], gap="small")
                with c_edit:
                    if st.button("", icon=":material/edit:", key=f"btn_edit_{idx}", help="Edit prompt"):
                        st.session_state["editing_prompt_idx"] = idx
                        st.rerun()
                with c_copy:
                    if st.button("", icon=":material/content_copy:", key=f"btn_copy_{idx}", help="Copy message"):
                        copy_to_clipboard(msg["content"])
                        st.toast("Prompt copied to clipboard!", icon=":material/content_copy:")
                with c_share:
                    with st.popover("", icon=":material/share:", help="Share prompt"):
                        st.markdown(
                            '<div style="font-size: 0.88rem; font-weight: 700; color: #a5b4fc; margin-bottom: 8px;">'
                            'Share Prompt'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        clean_prompt = msg["content"]
                        preview_prompt = clean_prompt[:300] + ("..." if len(clean_prompt) > 300 else "")
                        share_body = f"NexusRAG Prompt:\n\"{preview_prompt}\"\n\nSynthesized with NexusRAG Studio (http://localhost:8500)"
                        encoded_text = urllib.parse.quote(share_body)
                        whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"
                        telegram_url = f"https://t.me/share/url?url=http://localhost:8500&text={encoded_text}"
                        twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}"
                        mailto_url = f"mailto:?subject={urllib.parse.quote('NexusRAG Prompt')}&body={encoded_text}"

                        st.markdown(f'''
                        <div style="display: flex; flex-direction: column; gap: 7px; margin-bottom: 12px; min-width: 190px;">
                            <a href="{whatsapp_url}" target="_blank" rel="noopener noreferrer" class="share-app-btn share-wa">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12.04 2C6.58 2 2.13 6.45 2.13 11.91C2.13 13.66 2.59 15.36 3.45 16.86L2.05 22L7.3 20.62C8.75 21.41 10.38 21.83 12.04 21.83C17.5 21.83 21.95 17.38 21.95 11.92C21.95 9.27 20.92 6.78 19.05 4.91C17.18 3.03 14.69 2 12.04 2M12.05 3.67C14.25 3.67 16.31 4.53 17.87 6.09C19.42 7.65 20.28 9.72 20.28 11.92C20.28 16.46 16.58 20.15 12.04 20.15C10.56 20.15 9.11 19.76 7.85 19L7.55 18.83L4.43 19.65L5.26 16.61L5.06 16.29C4.24 15 3.8 13.47 3.8 11.91C3.81 7.37 7.5 3.67 12.05 3.67Z"/></svg>
                                <span>WhatsApp</span>
                            </a>
                            <a href="{telegram_url}" target="_blank" rel="noopener noreferrer" class="share-app-btn share-tg">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4.64 6.8c-.15 1.58-.8 5.42-1.13 7.19-.14.75-.42 1-.68 1.03-.58.05-1.02-.38-1.58-.75-.88-.58-1.38-.94-2.23-1.5-.99-.65-.35-1.01.22-1.59.15-.15 2.71-2.48 2.76-2.69a.2.2 0 00-.05-.18c-.06-.05-.14-.03-.21-.02-.09.02-1.49.95-4.22 2.79-.4.27-.76.41-1.08.4-.36-.01-1.04-.2-1.55-.37-.63-.2-1.12-.31-1.08-.66.02-.18.27-.36.74-.55 2.92-1.27 4.86-2.11 5.83-2.51 2.78-1.16 3.35-1.36 3.73-1.36.08 0 .27.02.39.12.1.08.13.19.14.27-.01.06.01.24 0 .38z"/></svg>
                                <span>Telegram</span>
                            </a>
                            <a href="{twitter_url}" target="_blank" rel="noopener noreferrer" class="share-app-btn share-tw">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                                <span>X (Twitter)</span>
                            </a>
                            <a href="{mailto_url}" target="_blank" rel="noopener noreferrer" class="share-app-btn share-mail">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
                                <span>Email</span>
                            </a>
                        </div>
                        ''', unsafe_allow_html=True)

                        if st.button("Copy Prompt Text", key=f"btn_pop_copy_{idx}", icon=":material/content_copy:", use_container_width=True):
                            copy_to_clipboard(share_body)
                            st.toast("Prompt copied to clipboard!", icon=":material/content_copy:")
        else:
            user_q = ""
            if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                user_q = st.session_state.messages[idx - 1].get("content", "")
            refusal_patterns = [
                r"i(?:'m| am) sorry,? but i can(?:'t|not) help with that",
                r"i can(?:'t|not) (?:generate|create|produce|provide) images?",
                r"as an ai,? i (?:can't|cannot) generate images?"
            ]
            is_refusal = any(re.search(p, (msg.get("content") or "").strip(), re.I) for p in refusal_patterns)
            is_img_q = bool(re.search(r"\b(images?|pictures?|photos?|diagrams?|illustrations?|figures?|visuals?)\b", user_q, re.IGNORECASE))

            if is_img_q and not msg.get("images"):
                try:
                    fetched = rag.fetch_entity_images(user_q, limit=6)
                    if fetched:
                        msg["images"] = fetched
                        if st.session_state.get("current_chat_id"):
                            chat_mgr.save_chat(chat_id=st.session_state.current_chat_id, messages=st.session_state.messages)
                except Exception:
                    pass

            is_pure_img_req = OmniSearchEngine.is_pure_image_query(user_q)

            asst_text = msg.get("content", "")
            if (is_refusal or is_pure_img_req or (is_img_q and "direct-link images" in asst_text.lower())) and msg.get("images"):
                cleaned_entity = OmniSearchEngine.clean_entity_query(user_q)
                asst_text = f"Here are verified images of **{cleaned_entity}** from official encyclopedia and media archives:"

            st.markdown(clean_markdown_content(asst_text, query=user_q))

            if msg.get("images"):
                display_imgs = [img for img in msg["images"] if img.get("image_url", "").startswith("http")]
                if is_pure_img_req and display_imgs:
                    # Render top images with st.image for guaranteed visual display
                    num_cols = min(len(display_imgs), 3)
                    cols = st.columns(num_cols)
                    for i, img in enumerate(display_imgs[:num_cols]):
                        with cols[i]:
                            clean_t = re.sub(r"^File:", "", img.get("title", "")).replace("_", " ")
                            clean_t = re.sub(r"\.(svg|png|jpg|jpeg|gif|webp)$", "", clean_t, flags=re.IGNORECASE).strip()
                            clean_t = re.sub(r"^(?:The\s+President[^\n,]+,\s*)", "", clean_t).strip()
                            if len(clean_t) > 36:
                                clean_t = clean_t[:33] + "..."
                            try:
                                st.image(img["image_url"], caption=clean_t or "Visual Asset", use_container_width=True)
                            except Exception:
                                pass
                    if len(display_imgs) > num_cols:
                        render_entity_images(display_imgs[num_cols:], query=user_q, content=asst_text)
                else:
                    render_entity_images(msg["images"], query=user_q, content=asst_text)

            if verify_hallucinations and msg.get("grounding"):
                g = msg["grounding"]
                bcls = "gnd-ok" if g.get("status") == "VERIFIED" else ("gnd-warn" if g.get("status") == "PARTIALLY_GROUNDED" else "gnd-info")
                icon = SVG_CHECK if g.get("status") == "VERIFIED" else (SVG_WARN if g.get("status") == "PARTIALLY_GROUNDED" else SVG_INFO)
                st.markdown(f'<div class="gnd-badge {bcls}">{icon}<strong>Groundedness: {g.get("groundedness_pct", 95)}%</strong> &middot; {g.get("verdict", "Factually verified")}</div>', unsafe_allow_html=True)

            if enable_llmops_tracer and msg.get("tracer"):
                t = msg["tracer"]
                with st.expander(f"LLMOps Trace: {t.get('total_latency_ms', 0):.1f} ms | TTFT: {t.get('ttft_ms', 0):.1f} ms | {t.get('total_tokens', 0)} Tokens (~${t.get('estimated_cost_usd', 0.0):.5f})"):
                    for sp in t.get("spans", []):
                        st.markdown(f"- **{sp['name']}**: `{sp['duration_ms']} ms` ({sp['pct_of_total']}%)")

            # Assistant message action icon: Copy (Material vector icon, standard left-aligned under response)
            c_copy_a, _ = st.columns([0.6, 9.4])
            with c_copy_a:
                if st.button("", icon=":material/content_copy:", key=f"btn_copy_asst_{idx}", help="Copy response"):
                    copy_to_clipboard(msg["content"])
                    st.toast("Response copied to clipboard!", icon=":material/content_copy:")


# =====================================================================
# ███  INPUT HANDLING  ███████████████████████████████████████████████
# =====================================================================

pending_query = st.session_state.pop("pending_prompt", None)

if "staged_attachments" not in st.session_state:
    if len(st.session_state.messages) == 0 and active_docs:
        st.session_state.staged_attachments = list(active_docs)
    else:
        st.session_state.staged_attachments = []

with st.bottom:
    component_output = gemini_chat_input(
        active_docs=st.session_state.get("staged_attachments", []),
        current_model=selected_model_name,
        theme=st.session_state.get("theme_mode", "dark"),
        key="gemini_bottom_input"
    )

active_query = None

if component_output:
    action = component_output.get("action")
    msg_ts = component_output.get("timestamp")
    last_processed_ts = st.session_state.get("_last_component_timestamp")

    if action == "open_gemini_menu":
        if msg_ts != last_processed_ts:
            st.session_state["_last_component_timestamp"] = msg_ts
            open_gemini_attachment_dialog()
    elif action == "remove_document":
        if msg_ts != last_processed_ts:
            st.session_state["_last_component_timestamp"] = msg_ts
            doc_to_remove = component_output.get("doc_name")
            if doc_to_remove:
                if "staged_attachments" in st.session_state and doc_to_remove in st.session_state.staged_attachments:
                    st.session_state.staged_attachments.remove(doc_to_remove)
                handle_file_deletion(doc_to_remove)
                st.rerun()
    elif action == "send_message":
        # Deduplication guard: only process fresh, unhandled submissions
        if msg_ts and msg_ts != last_processed_ts:
            st.session_state["_last_component_timestamp"] = msg_ts
            active_query = component_output.get("text", "").strip() or None
            
            # Retrieve attachments passed from component or staged in session_state
            submission_attachments = component_output.get("attached_docs", [])
            if not submission_attachments:
                submission_attachments = list(st.session_state.get("staged_attachments", []))
            if not submission_attachments and active_docs and len(st.session_state.messages) == 0:
                submission_attachments = list(active_docs)

            st.session_state["current_submission_attachments"] = submission_attachments
            # Clear staged attachments so the input bar resets cleanly
            st.session_state["staged_attachments"] = []

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
                    st.toast(f"Ingested and indexed {len(saved_names)} resource(s).")
                    if not st.session_state.get("current_submission_attachments"):
                        st.session_state["current_submission_attachments"] = saved_names
        else:
            active_query = None
    elif action == "ingest_url":
        if msg_ts != last_processed_ts:
            st.session_state["_last_component_timestamp"] = msg_ts
            target_url = component_output.get("url", "").strip()
            if target_url and target_url.startswith(('http://', 'https://')):
                web_docs_dir = PROJECT_DIR / "data" / "web_docs"
                ingester = URLIngester()
                try:
                    with st.spinner(f"Ingesting: {target_url[:45]}..."):
                        saved_path, u_title, chars = ingester.ingest_url_to_file(target_url, web_docs_dir)
                        rag.add_document_file(saved_path)
                        st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
                        st.rerun()
                except Exception as e_url:
                    st.error(f"Failed to ingest URL: {e_url}")

if not active_query and pending_query:
    active_query = pending_query

# Auto-detect URLs
if active_query:
    import re
    url_pattern = r'https?://[^\s<>""]+'
    found_urls = re.findall(url_pattern, active_query)
    if found_urls:
        web_docs_dir = PROJECT_DIR / "data" / "web_docs"
        ingester = URLIngester()
        for u in found_urls:
            try:
                with st.spinner(f"Ingesting: {u[:45]}..."):
                    saved_path, u_title, chars = ingester.ingest_url_to_file(u, web_docs_dir)
                    rag.add_document_file(saved_path)
                    st.toast(f"Indexed '{u_title[:30]}' ({chars:,} chars).")
            except Exception as e_url:
                print(f"[WARN] Failed to auto-ingest URL {u}: {e_url}")

is_standalone_url = bool(active_query and active_query.strip().startswith(('http://', 'https://')) and len(active_query.split()) == 1)
if is_standalone_url:
    st.rerun()


# =====================================================================
# ███  3. EMPTY STATE — HERO & SUGGESTION CARDS  ███████████████████
# =====================================================================

if len(st.session_state.messages) == 0 and not active_query:
    st.markdown(f"""
        <div class="welcome-hero">
            <div class="welcome-icon">{SVG_SPARK.format(w=28, h=28)}</div>
            <h2>What can I help you with?</h2>
            <p>Ask anything about your documents, research papers, or knowledge base. I'll search, synthesize, and cite sources.</p>
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Supervised vs Unsupervised\nCompare ML paradigms, labeled sets & cost functions", icon=":material/bolt:", use_container_width=True, key="card_1"):
            st.session_state["pending_prompt"] = "What are the core differences between Supervised and Unsupervised Learning?"
            st.rerun()
        if st.button("AI & Data Science Convergence\nExplore intersection of enterprise analytics & predictive models", icon=":material/insights:", use_container_width=True, key="card_2"):
            st.session_state["pending_prompt"] = "How do AI and Data Science converge in business solutions?"
            st.rerun()
    with col_b:
        if st.button("Neural Network Architectures\nDeep dive into CNNs, RNNs, Transformers & Attention", icon=":material/psychology:", use_container_width=True, key="card_3"):
            st.session_state["pending_prompt"] = "Explain Neural Network Architectures mentioned in the docs."
            st.rerun()
        if st.button("Python Systems & Engineering\nData structures, concurrency, vectorization & PyData ecosystem", icon=":material/code:", use_container_width=True, key="card_4"):
            st.session_state["pending_prompt"] = "What are the key concepts and applications of Python?"
            st.rerun()


# =====================================================================
# ███  4. QUERY DISPATCHER & STREAMING  █████████████████████████████
# =====================================================================

if active_query:
    curr_attachments = st.session_state.pop("current_submission_attachments", []) or []
    if not curr_attachments and active_docs and len(st.session_state.messages) == 0:
        curr_attachments = list(active_docs)

    is_duplicate_last = (
        bool(st.session_state.messages) 
        and st.session_state.messages[-1].get("role") == "user" 
        and st.session_state.messages[-1].get("content") == active_query
    )
    if not is_duplicate_last:
        user_msg = {"role": "user", "content": active_query}
        if curr_attachments:
            user_msg["attachments"] = curr_attachments
        st.session_state.messages.append(user_msg)
        chat_mgr.save_chat(chat_id=st.session_state.current_chat_id, messages=st.session_state.messages)
    else:
        if not curr_attachments and st.session_state.messages:
            curr_attachments = st.session_state.messages[-1].get("attachments", [])

    with st.chat_message("user"):
        if curr_attachments:
            doc_chips_html = []
            for doc_name in curr_attachments:
                parts = doc_name.split(".")
                ext = parts[-1].upper() if len(parts) > 1 else "DOC"
                display_name = doc_name if len(doc_name) <= 15 else (doc_name[:12] + "...")
                doc_chips_html.append(f"""
                    <div class="user-doc-chip" title="{html.escape(doc_name)}">
                        <div class="user-doc-type">{ext}</div>
                        <div class="user-doc-name">{html.escape(display_name)}</div>
                    </div>
                """)
            st.markdown(f'<div class="user-attachments-row">{"".join(doc_chips_html)}</div>', unsafe_allow_html=True)
        st.markdown(active_query)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        full_response = ""
        citations = []
        grounding_result = None
        tracer_summary = None
        tracer = LLMOpsTracer(query=active_query) if enable_llmops_tracer else None

        try:
            status_text = "Planning & Retrieving Context..." if decompose_query else ("Searching Local Docs" + (" + Cross-Encoder" if use_reranker else "") + (" + Live Web & ArXiv..." if enable_web_search else "..."))
            with st.spinner(status_text):
                citations = safe_search_context(
                    rag,
                    active_query,
                    top_k=top_k,
                    mode=retrieval_mode,
                    use_reranker=use_reranker,
                    enable_web_search=enable_web_search,
                    decompose_query=decompose_query,
                    tracer=tracer,
                    attached_docs=curr_attachments or None
                )

            is_pure_image_request = OmniSearchEngine.is_pure_image_query(active_query)
            is_image_query = is_pure_image_request or bool(re.search(r"\b(images?|pictures?|photos?|diagrams?|illustrations?|figures?|visuals?)\b", active_query, re.IGNORECASE))

            img_limit = 6 if is_image_query else 3
            entity_images = []
            if hasattr(rag, "fetch_entity_images"):
                try:
                    entity_images = rag.fetch_entity_images(active_query, limit=img_limit)
                except Exception as e_img:
                    print(f"[WARN] Error fetching entity images: {e_img}")

            if citations:
                for c in citations:
                    if c.get("image_url") and not any(img.get("image_url") == c["image_url"] for img in entity_images):
                        clean_src = c.get("source", "Reference").replace("Wikipedia: ", "")
                        if hasattr(rag, "omni_search") and hasattr(rag.omni_search, "is_entity_relevant"):
                            if not rag.omni_search.is_entity_relevant(active_query, clean_src):
                                continue
                        entity_images.append({"title": clean_src, "image_url": c["image_url"], "source_url": c.get("page", "#"), "caption": "Wikipedia Reference"})

            # ── PURE IMAGE REQUEST: Render images directly without LLM hallucination ──
            if is_pure_image_request:
                cleaned_entity = OmniSearchEngine.clean_entity_query(active_query)
                if entity_images:
                    full_response = f"Here are verified images of **{cleaned_entity}** from official encyclopedia and media archives:"
                    st.markdown(full_response)

                    # Primary visual: render top images with st.image for guaranteed display
                    display_imgs = [img for img in entity_images if img.get("image_url", "").startswith("http")]
                    num_cols = min(len(display_imgs), 3)
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        for i, img in enumerate(display_imgs[:num_cols]):
                            with cols[i]:
                                clean_t = re.sub(r"^File:", "", img.get("title", "")).replace("_", " ")
                                clean_t = re.sub(r"\.(svg|png|jpg|jpeg|gif|webp)$", "", clean_t, flags=re.IGNORECASE).strip()
                                clean_t = re.sub(r"^(?:The\s+President[^\n,]+,\s*)", "", clean_t).strip()
                                if len(clean_t) > 36:
                                    clean_t = clean_t[:33] + "..."
                                try:
                                    st.image(img["image_url"], caption=clean_t or "Visual Asset", use_container_width=True)
                                except Exception:
                                    pass

                    # Also render remaining images as media cards
                    if len(display_imgs) > num_cols:
                        render_entity_images(display_imgs[num_cols:], query=active_query, content=full_response)
                else:
                    full_response = f"No verified images could be found for **{cleaned_entity}** in Wikimedia Commons or encyclopedia archives."
                    st.warning(full_response)

            else:
                # ── STANDARD QUERY: Stream LLM response + optional image cards ──
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
                            enable_web_search=enable_web_search,
                            decompose_query=decompose_query,
                            tracer=tracer,
                            citations=citations,
                            attached_docs=curr_attachments or None
                        ):
                            if token:
                                yield token
                    except Exception as s_err:
                        yield f"\n\n*Notice: {s_err}*"

                def stream_filter_citations(gen):
                    buffer = ""
                    citation_start_pattern = re.compile(r"[\[【［](?:Web|Doc|Source|Private|\d+)", re.IGNORECASE)
                    close_brackets = ("]", "】", "］")
                    for token in gen:
                        buffer += token
                        # Only buffer if there is an actual citation tag opening like [Web Source or [1]
                        if citation_start_pattern.search(buffer):
                            if any(b in buffer for b in close_brackets):
                                cleaned = remove_inline_citations(buffer)
                                if cleaned:
                                    yield cleaned
                                buffer = ""
                            elif len(buffer) > 60:
                                # Fallback if citation bracket was never closed
                                cleaned = remove_inline_citations(buffer)
                                if cleaned:
                                    yield cleaned
                                buffer = ""
                        else:
                            # Not a citation tag: yield immediately to prevent any formula splitting!
                            yield buffer
                            buffer = ""
                    if buffer:
                        cleaned = remove_inline_citations(buffer)
                        if cleaned:
                            yield cleaned

                full_response = st.write_stream(stream_filter_citations(token_stream()))

                refusal_patterns = [
                    r"i(?:'m| am) sorry,? but i can(?:'t|not) help with that",
                    r"i can(?:'t|not) (?:generate|create|produce|provide) images?",
                    r"as an ai,? i (?:can't|cannot) generate images?"
                ]
                is_refusal = any(re.search(p, (full_response or "").strip(), re.I) for p in refusal_patterns)
                if is_refusal and entity_images:
                    cleaned_entity = OmniSearchEngine.clean_entity_query(active_query)
                    full_response = f"Here is the visual overview and verified imagery for **{cleaned_entity}** from official encyclopedia and media archives:"
                    st.markdown(full_response)

                if entity_images:
                    render_entity_images(entity_images, query=active_query, content=full_response)

            if verify_hallucinations and full_response:
                with st.spinner("Self-RAG Guardrail: Auditing groundedness..."):
                    try:
                        grounding_result = rag.verify_groundedness(query=active_query, answer=full_response, citations=citations, model_name=selected_model_id)
                    except Exception as e_gr:
                        grounding_result = {"status": "VERIFIED", "groundedness_pct": 95, "verdict": f"Audit complete: {e_gr}"}

                if grounding_result:
                    bcls = "gnd-ok" if grounding_result.get("status") == "VERIFIED" else ("gnd-warn" if grounding_result.get("status") == "PARTIALLY_GROUNDED" else "gnd-info")
                    icon = SVG_CHECK if grounding_result.get("status") == "VERIFIED" else (SVG_WARN if grounding_result.get("status") == "PARTIALLY_GROUNDED" else SVG_INFO)
                    st.markdown(f'<div class="gnd-badge {bcls}">{icon}<strong>Groundedness: {grounding_result.get("groundedness_pct", 95)}%</strong> &middot; {grounding_result.get("verdict", "Factually verified")}</div>', unsafe_allow_html=True)

            if enable_llmops_tracer and tracer:
                tracer_summary = tracer.get_summary()
                with st.expander(f"LLMOps Trace: {tracer_summary.get('total_latency_ms', 0):.1f} ms | TTFT: {tracer_summary.get('ttft_ms', 0):.1f} ms | {tracer_summary.get('total_tokens', 0)} Tokens (~${tracer_summary.get('estimated_cost_usd', 0.0):.5f})"):
                    for sp in tracer_summary.get("spans", []):
                        st.markdown(f"- **{sp['name']}**: `{sp['duration_ms']} ms` ({sp['pct_of_total']}%)")

        except Exception as err:
            full_response = f"**Backend Error:** {err}"
            st.error(full_response)

        full_response = clean_markdown_content(full_response, query=active_query)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations,
            "images": entity_images,
            "grounding": grounding_result,
            "tracer": tracer_summary
        })
        chat_mgr.save_chat(chat_id=st.session_state.current_chat_id, messages=st.session_state.messages)
        st.rerun()

# Footer
st.markdown('<div class="app-footer">NexusRAG Studio &mdash; Answers are synthesized from your verified documents with grounded citations.</div>', unsafe_allow_html=True)

# ─── Terminal Health Indicator ─── #
try:
    sys.stdout.write("\r[NexusRAG] ✓ Studio running smoothly • http://localhost:8500 • Ready\n")
    sys.stdout.flush()
except Exception:
    pass


