import os
import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project and src are on sys.path
PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag.search import RAGSearch, AVAILABLE_MODELS, DEFAULT_GROQ_MODEL

app = FastAPI(
    title="NexusRAG Studio",
    description="Enterprise-grade Retrieval-Augmented Generation (RAG) platform with FAISS and Groq LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG Search Instance
rag_engine: Optional[RAGSearch] = None

def get_rag() -> RAGSearch:
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGSearch(
            persist_dir=str(PROJECT_DIR / "faiss_store"),
            data_dir=str(PROJECT_DIR / "data"),
            llm_model=DEFAULT_GROQ_MODEL
        )
    return rag_engine

# Static directory for frontend assets
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request Models
class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatTurn]] = []
    top_k: Optional[int] = 4
    model_name: Optional[str] = None

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>NexusRAG Server Running</h1><p>Frontend static files not found.</p>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/status")
async def get_status():
    rag = get_rag()
    stats = rag.vectorstore.get_stats()
    return {
        "status": "online",
        "models": AVAILABLE_MODELS,
        "default_model": rag.default_model,
        "embedding_model": rag.vectorstore.embedding_model,
        "stats": stats,
        "has_groq_key": bool(os.getenv("GROQ_API_KEY"))
    }

@app.get("/api/documents")
async def list_documents():
    data_dir = PROJECT_DIR / "data"
    documents = []
    supported_exts = {".pdf", ".txt", ".csv", ".docx", ".doc", ".xlsx", ".json"}
    
    if data_dir.exists():
        for f in data_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                if "vector_store" in f.parts or "faiss_store" in f.parts:
                    continue
                documents.append({
                    "name": f.name,
                    "relative_path": str(f.relative_to(data_dir)),
                    "size_bytes": f.stat().st_size,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "extension": f.suffix.lower().replace(".", ""),
                    "last_modified": f.stat().st_mtime
                })
    return {"documents": sorted(documents, key=lambda d: d["name"].lower())}

@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    rag = get_rag()
    data_dir = PROJECT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        safe_filename = Path(file.filename).name
        # Categorize by extension into folder if helpful
        ext = Path(safe_filename).suffix.lower()
        if ext == ".pdf":
            target_folder = data_dir / "pdf"
        elif ext in [".txt", ".csv", ".json"]:
            target_folder = data_dir / "text_files"
        else:
            target_folder = data_dir / "uploads"
        
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = target_folder / safe_filename
        
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Incrementally add to vectorstore
        rag.add_document_file(target_path)
        uploaded_files.append(safe_filename)

    stats = rag.vectorstore.get_stats()
    return {
        "message": f"Successfully uploaded and indexed {len(uploaded_files)} document(s).",
        "uploaded_files": uploaded_files,
        "stats": stats
    }

@app.post("/api/documents/reindex")
async def reindex_documents():
    rag = get_rag()
    stats = rag.reindex_all()
    return {
        "message": "Full re-indexing completed successfully.",
        "stats": stats
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    rag = get_rag()
    history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
    try:
        result = rag.search_and_summarize(
            query=request.query,
            chat_history=history,
            top_k=request.top_k or 4,
            model_name=request.model_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    rag = get_rag()
    history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
    citations = rag.search_context(request.query, top_k=request.top_k or 4)

    def event_generator():
        # First event sends the citations payload
        citations_payload = json.dumps({"type": "citations", "citations": citations})
        yield f"data: {citations_payload}\n\n"

        # Stream LLM tokens
        try:
            for token in rag.stream_query(
                query=request.query,
                chat_history=history,
                top_k=request.top_k or 4,
                model_name=request.model_name
            ):
                token_payload = json.dumps({"type": "token", "content": token})
                yield f"data: {token_payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
