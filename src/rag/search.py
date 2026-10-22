import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
from dotenv import load_dotenv

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env")
load_dotenv()  # Fallback to local .env

try:
    from .data_loader import load_all_documents, load_single_document
    from .vectorstore import FaissVectorStore
    from .hybrid_retriever import HybridRetriever
    from .reranker import DocumentReranker
    from .multi_search import OmniSearchEngine
    from .tracer import LLMOpsTracer
    from .guardrails import AgenticGuardrail
except ImportError:
    from rag.data_loader import load_all_documents, load_single_document
    from rag.vectorstore import FaissVectorStore
    from rag.hybrid_retriever import HybridRetriever
    from rag.reranker import DocumentReranker
    from rag.multi_search import OmniSearchEngine
    from rag.tracer import LLMOpsTracer
    from rag.guardrails import AgenticGuardrail

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = [
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B (High Quality)",
        "badge": "120B",
        "desc": "Deep reasoning & grounded synthesis"
    },
    {
        "id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B (Ultra Fast)",
        "badge": "20B",
        "desc": "Lightning fast responses for quick queries"
    },
    {
        "id": "qwen/qwen3.8-27b",
        "name": "Qwen 3.8 27B (Coder & Logic)",
        "badge": "27B",
        "desc": "Strong coding, mathematics & technical analysis"
    },
    {
        "id": "allam-2-7b",
        "name": "Allam 2 7B (Lightweight)",
        "badge": "7B",
        "desc": "Fast lightweight conversational model"
    },
]

def sanitize_text(text: str) -> str:
    """Sanitize non-standard unicode characters that can break terminal or client encodings."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return (
        text.replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "--")
            .replace("\u202f", " ")
            .replace("\u00a0", " ")
    )

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = DEFAULT_GROQ_MODEL,
        data_dir: str = "data",
    ):
        self.persist_path = Path(persist_dir)
        if not self.persist_path.is_absolute():
            self.persist_path = PROJECT_DIR / self.persist_path

        self.data_path = Path(data_dir)
        if not self.data_path.is_absolute():
            self.data_path = PROJECT_DIR / self.data_path

        self.default_model = llm_model
        self.vectorstore = FaissVectorStore(str(self.persist_path), embedding_model)
        
        # Load or initialize vectorstore
        if not self.vectorstore.is_initialized():
            print(f"[INFO] Initializing vector store from documents in {self.data_path}...")
            docs = load_all_documents(self.data_path)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Initialize Hybrid Retriever (Dense + BM25)
        self.hybrid = HybridRetriever(self.vectorstore)
        self._reranker: Optional[DocumentReranker] = None
        self._omni_search: Optional[OmniSearchEngine] = None

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            print("[WARN] GROQ_API_KEY is not set in environment or .env file.")

        # Initialize embedded ACID-compliant SQLite Document & Chunk Database
        from rag.database import SQLiteDocumentDB
        self.sqlite_db = SQLiteDocumentDB(self.data_path / "nexusrag.db")

        # Auto-sync existing indexed documents to SQLite if needed
        try:
            existing_db_docs = self.sqlite_db.get_all_documents()
            if not existing_db_docs and hasattr(self.vectorstore, "metadata") and self.vectorstore.metadata:
                by_source = {}
                for m in self.vectorstore.metadata:
                    src = m.get("source_name") or Path(m.get("source", "doc")).name
                    if src not in by_source:
                        by_source[src] = []
                    by_source[src].append(m)
                for src_name, chunk_metas in by_source.items():
                    p = None
                    for candidate in self.data_path.rglob(src_name):
                        if candidate.is_file():
                            p = candidate
                            break
                    raw_bytes = p.read_bytes() if p and p.is_file() else b""
                    ftype = p.suffix.lstrip(".").lower() if p else "pdf"
                    preview = chunk_metas[0].get("text", "") if chunk_metas else ""
                    self.sqlite_db.store_document(
                        filename=src_name,
                        file_type=ftype,
                        raw_bytes=raw_bytes,
                        chunks=chunk_metas,
                        text_preview=preview
                    )
        except Exception as e_sql_sync:
            print(f"[INFO] SQLite auto-sync notice: {e_sql_sync}")

        # Initialize PostgreSQL Vector Database connection only if configured (avoids connection timeout delay)
        keys_to_check = ["DATABASE_URL", "POSTGRES_URL", "POSTGRES_HOST", "database_url", "postgres_url", "postgres_host"]
        has_pg_config = any(os.getenv(k) for k in keys_to_check) or os.getenv("ENABLE_POSTGRES") == "true"
        if not has_pg_config:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    has_pg_config = any(k in st.secrets for k in keys_to_check + ["postgres", "supabase"])
            except Exception:
                pass
        if has_pg_config:
            try:
                from rag.database import PostgresVectorDB
                self.pg_db = PostgresVectorDB()
                if not self.pg_db.is_connected:
                    self.pg_db = None
                else:
                    # Auto-sync existing local indexed documents to Supabase if Supabase is empty
                    try:
                        existing_pg_docs = self.pg_db.list_documents()
                        if not existing_pg_docs:
                            raw_docs = load_all_documents(self.data_path)
                            if raw_docs:
                                print(f"[INFO] Auto-syncing {len(raw_docs)} documents to Supabase...")
                                chunks = self.vectorstore.emb_pipe.chunk_documents(raw_docs)
                            if chunks:
                                embeddings = self.vectorstore.emb_pipe.embed_chunks(chunks)
                                by_source = {}
                                for chunk, emb in zip(chunks, embeddings):
                                    src = chunk.metadata.get("source", "unknown")
                                    if src not in by_source:
                                        by_source[src] = {"chunks": [], "embeddings": []}
                                    by_source[src]["chunks"].append(chunk)
                                    by_source[src]["embeddings"].append(emb)

                                for src, data in by_source.items():
                                    p = Path(src)
                                    raw_bytes = p.read_bytes() if p.exists() and p.is_file() else b""
                                    self.pg_db.store_document(
                                        filename=p.name or src,
                                        file_type=p.suffix.lstrip(".").lower() or "txt",
                                        raw_bytes=raw_bytes,
                                        chunks=data["chunks"],
                                        embeddings=data["embeddings"]
                                    )
                                print("[INFO] Supabase auto-sync completed successfully.")
                    except Exception as e_pg_sync:
                        print(f"[WARN] Failed to auto-sync documents to Supabase: {e_pg_sync}")
            except Exception as e_db:
                print(f"[INFO] PostgreSQL client not initialized: {e_db}")
                self.pg_db = None
        else:
            self.pg_db = None

        self.guardrail = AgenticGuardrail()

    @property
    def reranker(self) -> DocumentReranker:
        """Lazy load Cross-Encoder model to conserve startup resources."""
        if self._reranker is None:
            self._reranker = DocumentReranker()
        return self._reranker

    @property
    def omni_search(self) -> OmniSearchEngine:
        """Lazy load Omni-Source Search Engine (Wikipedia, ArXiv, Google/Serper)."""
        if self._omni_search is None:
            self._omni_search = OmniSearchEngine()
        return self._omni_search

    def fetch_entity_images(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Fetch visual media, entity portraits, and concept diagrams."""
        try:
            return self.omni_search.fetch_relevant_images(query, limit=limit, data_dir=self.data_path)
        except Exception as e:
            print(f"[WARN] Failed to fetch entity images: {e}")
            return []

    def get_llm(self, model_name: Optional[str] = None, temperature: float = 0.2):
        api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file.")
        chosen_model = model_name or self.default_model
        return ChatGroq(groq_api_key=api_key, model_name=chosen_model, temperature=temperature)

    def reindex_all(self) -> Dict[str, Any]:
        """Reload all documents from data folder and rebuild the vector store and PostgreSQL."""
        docs = load_all_documents(self.data_path)
        self.vectorstore.build_from_documents(docs)
        self.hybrid.reload()

        # Mirror existing documents and chunks to PostgreSQL if connected
        if hasattr(self, "pg_db") and self.pg_db and self.pg_db.is_connected and docs:
            try:
                chunks = self.vectorstore.emb_pipe.chunk_documents(docs)
                if chunks:
                    embeddings = self.vectorstore.emb_pipe.embed_chunks(chunks)
                    by_source = {}
                    for chunk, emb in zip(chunks, embeddings):
                        src = chunk.metadata.get("source", "unknown")
                        if src not in by_source:
                            by_source[src] = {"chunks": [], "embeddings": []}
                        by_source[src]["chunks"].append(chunk)
                        by_source[src]["embeddings"].append(emb)

                    for src, data in by_source.items():
                        p = Path(src)
                        raw_bytes = p.read_bytes() if p.exists() and p.is_file() else b""
                        self.pg_db.store_document(
                            filename=p.name or src,
                            file_type=p.suffix.lstrip(".").lower() or "txt",
                            raw_bytes=raw_bytes,
                            chunks=data["chunks"],
                            embeddings=data["embeddings"]
                        )
            except Exception as e_sync:
                print(f"[WARN] Failed to sync to PostgreSQL during reindex_all: {e_sync}")

        # Mirror all documents to embedded SQLite Database
        if hasattr(self, "sqlite_db") and self.sqlite_db and docs:
            try:
                chunks = self.vectorstore.emb_pipe.chunk_documents(docs)
                by_source = {}
                for chunk in chunks:
                    src = chunk.metadata.get("source", "unknown")
                    if src not in by_source:
                        by_source[src] = []
                    by_source[src].append(chunk)

                for src, c_list in by_source.items():
                    p = Path(src)
                    raw_bytes = p.read_bytes() if p.exists() and p.is_file() else b""
                    prev = c_list[0].page_content if c_list else ""
                    self.sqlite_db.store_document(
                        filename=p.name or src,
                        file_type=p.suffix.lstrip(".").lower() or "txt",
                        raw_bytes=raw_bytes,
                        chunks=c_list,
                        text_preview=prev
                    )
            except Exception as e_sq:
                print(f"[WARN] Failed to sync to SQLite during reindex_all: {e_sq}")

        return self.vectorstore.get_stats()

    def add_document_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Incrementally load, index, and save a single uploaded document to database and FAISS."""
        docs = load_single_document(file_path)
        if docs:
            self.vectorstore.add_documents(docs)
            self.hybrid.reload()

            p = Path(file_path)
            raw_bytes = p.read_bytes() if p.is_file() else b""
            chunks = self.vectorstore.emb_pipe.chunk_documents(docs)
            prev = chunks[0].page_content if chunks else ""

            # 1. Permanently store into embedded SQLite Database
            if hasattr(self, "sqlite_db") and self.sqlite_db:
                try:
                    self.sqlite_db.store_document(
                        filename=p.name,
                        file_type=p.suffix.lstrip(".").lower() or "txt",
                        raw_bytes=raw_bytes,
                        chunks=chunks,
                        text_preview=prev
                    )
                except Exception as e_sdb:
                    print(f"[WARN] Failed to store {p.name} in SQLite: {e_sdb}")

            # 2. Mirror to secure PostgreSQL vectorstore if connected
            if hasattr(self, "pg_db") and self.pg_db and self.pg_db.is_connected:
                try:
                    embeddings = self.vectorstore.emb_pipe.embed_chunks(chunks)
                    self.pg_db.store_document(
                        filename=p.name,
                        file_type=p.suffix.lstrip(".").lower() or "txt",
                        raw_bytes=raw_bytes,
                        chunks=chunks,
                        embeddings=embeddings
                    )
                except Exception as e_pg:
                    print(f"[WARN] Failed to mirror document {p.name} to PostgreSQL: {e_pg}")

        return self.vectorstore.get_stats()

    def remove_document(self, doc_name: str) -> Dict[str, Any]:
        """
        Instantly remove a document from FAISS, BM25, SQLite database, and disk in milliseconds.
        Avoids expensive re-embedding.
        """
        # 1. Delete physical file from data directory
        for p in self.data_path.rglob(doc_name):
            if p.is_file():
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[WARN] Failed to unlink {p}: {e}")

        # 2. Instantly update FAISS vector store
        self.vectorstore.remove_document(doc_name)

        # 3. Fast reload BM25 lexical index in memory
        self.hybrid.reload()

        # 4. Remove from embedded SQLite database
        if hasattr(self, "sqlite_db") and self.sqlite_db:
            try:
                self.sqlite_db.remove_document(doc_name)
            except Exception as e_sdel:
                print(f"[WARN] Failed to delete {doc_name} from SQLite: {e_sdel}")

        # 5. Remove from PostgreSQL if connected
        if hasattr(self, "pg_db") and self.pg_db and self.pg_db.is_connected:
            try:
                from sqlalchemy import text
                with self.pg_db.engine.begin() as conn:
                    conn.execute(text("DELETE FROM documents WHERE filename = :fname"), {"fname": doc_name})
            except Exception as e_del:
                print(f"[WARN] Failed to delete from PostgreSQL: {e_del}")

        return self.vectorstore.get_stats()

    def search_context(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        use_reranker: bool = True,
        enable_web_search: bool = True,
        decompose_query: bool = False,
        tracer: Optional[LLMOpsTracer] = None,
        attached_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top matching document chunks with formatted metadata.
        Modes: 'hybrid' (BM25 + FAISS via RRF), 'dense' (FAISS only), 'sparse' (BM25 only).
        Supports Agentic Query Decomposition and LLMOps real-time tracing.
        """
        # Detect if query explicitly refers to uploaded documents, knowledge vault, or summarization of files
        doc_intent_pattern = re.compile(
            r"\b(uploaded|upload|updated document|updated file|this document|the document|the pdf|the file|"
            r"these documents|all documents|my file|my document|my pdf|what is in the document|summarize the document|"
            r"about the document|overview of the document|contents of the document|what did i upload|in the pdf|"
            r"in this file|document summary|summarize document|summarize the file|knowledge vault|knowledge base|"
            r"tell me about the document|explain the document|explain the file|tell me about the file|"
            r"what is this document about|what's in the document|what is in my document|tell me about the updated document|"
            r"give about this|explain this|tell me about this|summarize this|what is this|about this|"
            r"what does this say|simplify this|simple words|break this down|breakdown of this|details of this|"
            r"give me details|explain in detail|explain this in simple words)\b",
            re.IGNORECASE
        )
        is_doc_intent = bool(attached_docs) or bool(doc_intent_pattern.search(query))
        has_local_docs = bool(self.hybrid.corpus_chunks or (hasattr(self.vectorstore, "metadata") and self.vectorstore.metadata))

        # Optional Agentic Query Planning / Decomposition
        sub_queries = [query]
        if decompose_query and len(query.strip()) > 20 and not is_doc_intent:
            d_span = tracer.start_span("Agentic Query Planner (Decomposition)") if tracer else None
            try:
                sub_queries = self.guardrail.decompose_query(query, llm=self.get_llm())
            except Exception:
                sub_queries = [query]
            if d_span:
                d_span.finish({"sub_queries": sub_queries})

        # Stage 1: Candidate Generation
        stage1_k = max(top_k * 3, 12) if use_reranker else top_k
        s1_span = tracer.start_span(f"Stage 1 Retrieval ({mode.upper()})") if tracer else None

        raw_candidates_map = {}
        for q in sub_queries:
            if mode == "dense":
                sub_res = self.hybrid.search_dense(q, top_k=stage1_k)
            elif mode == "sparse":
                sub_res = self.hybrid.search_sparse(q, top_k=stage1_k)
            else:
                sub_res = self.hybrid.search_hybrid(q, top_k=stage1_k)

            for item in sub_res:
                key = item.get("text", "")[:100]
                if key not in raw_candidates_map:
                    raw_candidates_map[key] = item

        # If user explicitly attached documents or asks about their uploaded document,
        # prioritize and inject the indexed local documents from the knowledge vault
        if (is_doc_intent or attached_docs) and has_local_docs:
            all_local = self.hybrid.corpus_chunks or getattr(self.vectorstore, "metadata", []) or []
            for meta in all_local:
                src_name = meta.get("source_name") or Path(meta.get("source", "doc")).name
                # Check if attached_docs matches or include all local vault chunks
                is_match = True
                if attached_docs:
                    is_match = any(att.lower() in src_name.lower() or src_name.lower() in att.lower() for att in attached_docs)

                if is_match:
                    key = meta.get("text", "")[:100]
                    if key not in raw_candidates_map:
                        raw_candidates_map[key] = {
                            "index": 0,
                            "score": 0.98,
                            "metadata": meta,
                            "text": meta.get("text", ""),
                            "source": src_name,
                            "page": meta.get("page", 1),
                            "file_type": meta.get("file_type", ""),
                            "method": "vault_direct"
                        }

        raw_results = list(raw_candidates_map.values())
        if s1_span:
            s1_span.finish({"candidates_count": len(raw_results), "sub_queries_count": len(sub_queries)})

        # Stage 2: Cross-Encoder Re-Ranking
        s2_span = tracer.start_span("Stage 2 MS-MARCO Cross-Encoder") if tracer else None
        if use_reranker and raw_results and not is_doc_intent:
            raw_results = self.reranker.rerank(query=query, candidates=raw_results, top_k=top_k)
        else:
            raw_results = raw_results[:top_k]

        if s2_span:
            s2_span.finish({"reranked_count": len(raw_results)})

        citations = []
        for r in raw_results:
            score = round(float(r.get("score", 0.0)), 3)
            raw_logit = r.get("raw_rerank_score")

            # If doc intent is active, preserve all local vault chunks with high relevance
            if is_doc_intent and not r.get("is_web", False):
                score = max(score, 0.92)
                relevance_pct = int(score * 100)
            else:
                relevance_pct = max(0, min(100, int(score * 100)))
                # Discard completely unrelated chunks (0% relevance or negative cross-encoder logit)
                if raw_logit is not None and (raw_logit < -3.5 or relevance_pct <= 5):
                    if len(raw_results) > 1 and not is_doc_intent:
                        continue
                elif raw_logit is None and relevance_pct <= 5:
                    if len(raw_results) > 1 and not is_doc_intent:
                        continue

            snippet = r.get("text", "").strip()
            preview = (snippet[:200] + "...") if len(snippet) > 200 else snippet

            citations.append({
                "source": r.get("source", "Unknown"),
                "page": r.get("page", 1),
                "score": score,
                "relevance_pct": relevance_pct,
                "preview": preview,
                "full_text": snippet,
                "retrieval_mode": r.get("method", mode),
                "is_web": False
            })

        # Fallback: If filtering eliminated everything but we had Stage 1 local results, preserve top candidates
        if not citations and raw_results and not any(r.get("is_web") for r in raw_results):
            for top_fallback in raw_results[:top_k]:
                snippet = top_fallback.get("text", "").strip()
                citations.append({
                    "source": top_fallback.get("source", "Knowledge Vault Document"),
                    "page": top_fallback.get("page", 1),
                    "score": 0.88,
                    "relevance_pct": 88,
                    "preview": (snippet[:200] + "...") if len(snippet) > 200 else snippet,
                    "full_text": snippet,
                    "retrieval_mode": top_fallback.get("method", mode),
                    "is_web": False
                })

        # Stage 3: Live External Search Augmentation (Wikipedia, ArXiv, Google/Serper)
        # Suppress external web search when user is specifically querying their private uploaded documents
        if enable_web_search and query.strip() and not (is_doc_intent and has_local_docs):
            s3_span = tracer.start_span("Stage 3 Live External Search (OmniSearch)") if tracer else None
            try:
                external_results = self.omni_search.search_all(query, max_results=3)
                citations.extend(external_results)
            except Exception as e:
                print(f"[WARN] External search failed: {e}")
            if s3_span:
                s3_span.finish({"external_count": len(citations) - len(raw_results)})

        # Record audit trail in PostgreSQL if connected
        if hasattr(self, "pg_db") and self.pg_db and self.pg_db.is_connected:
            self.pg_db.log_audit(
                query_text=query,
                retrieval_mode=mode,
                top_chunks_count=len(citations)
            )

        return citations

    def _build_prompt_messages(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        attached_docs: Optional[List[str]] = None
    ) -> List[Any]:
        context_blocks = []
        for i, c in enumerate(citations, 1):
            source_name = c.get("source", "Reference")
            source_info = f"--- Context Passage: {source_name} ---"
            context_blocks.append(f"{source_info}\n{c['full_text']}")

        if context_blocks:
            context_str = "\n\n".join(context_blocks)
            doc_instructions = (
                "1. Synthesize accurate, comprehensive, and well-structured answers strictly grounded in the provided Context.\n"
                "2. FACTUAL GROUNDING ON DATES & CREDITS: Always prioritize the release dates, years, artists, singers, and producers stated in the Context.\n"
                "   - For example, if the context mentions a 2025 release or specific collaborators (e.g., Ali Brothers, Parvej Hussain, Nanku), use those exact facts. NEVER assume or invent an older year (such as 2023) or fictional singers.\n"
                "3. STRICT PROHIBITION ON CITATIONS & SOURCE NUMBERS: DO NOT write citation tags or source number labels (NEVER write '【Web Source 1】', '[Web Source 1]', 'Web Source 1, 2, 3', '[Doc 1]', '[1]', or any lenticular/bracketed numbers) in your response under ANY circumstances. Do NOT cite or refer to sources by index number. Write pure, natural, fluent narrative prose only.\n"
                "4. If private documents and web resources both touch on the topic, highlight how they complement each other.\n"
                "5. UPLOADED DOCUMENT SYNTHESIS: If the user asks about 'the document', 'the updated document', 'the uploaded file', or requests a summary/overview of their files, thoroughly synthesize and explain every major section, concept, pipeline, and method contained in the Context Passages.\n"
            )
            if attached_docs:
                att_names = ", ".join(attached_docs)
                doc_instructions += (
                    f"6. ATTACHED FILE GROUNDING: The user attached '{att_names}' directly to their prompt. "
                    "You MUST treat the provided Context as the contents of this attached document and address their prompt "
                    "(e.g., explain in simple words, give details, break it down) directly using this Context. "
                    "Never state or imply that no file was attached or visible.\n"
                )
        else:
            context_str = "No specific private documents or external sources matched this query."
            doc_instructions = (
                "1. Answer the user's question directly, clearly, and informatively using your broad technical and general knowledge.\n"
                "2. Do not fabricate or invent citations from unmentioned documents.\n"
                "3. If asking about recent events, releases, or songs and verified context is not available, explicitly state what is confirmed rather than guessing historical dates or collaborators.\n"
            )

        user_wants_table = bool(re.search(r"\b(table|tabular|grid|matrix)\b", query, re.IGNORECASE))
        if user_wants_table:
            format_instructions = (
                "3. RESPONSE FORMATTING:\n"
                "   - The user explicitly requested a table. Provide accurate, clean Markdown tables with headers.\n"
            )
        else:
            format_instructions = (
                "3. STRICT FORMATTING MANDATE (ABSOLUTELY NO TABLES):\n"
                "   - The user did NOT request a table. You MUST NOT output markdown tables (| ... | ... |).\n"
                "   - DO NOT format definitions, categories, paradigms, challenges, or comparisons as tables under ANY circumstances.\n"
                "   - Write all explanations, concepts, and answers in natural, fluent, narrative prose paragraphs and bullet points.\n"
                "   - Use clean subheadings (###) and bulleted lists for key takeaways, but maintain rich narrative prose.\n"
            )

        is_image_request = bool(re.search(r"\b(images?|pictures?|photos?|diagrams?|illustrations?|figures?|visuals?)\b", query, re.IGNORECASE))
        if is_image_request:
            visual_instructions = (
                "4. VISUAL MEDIA & IMAGE INQUIRIES (DO NOT REFUSE & DO NOT FABRICATE LINKS):\n"
                "   - The user is asking about images, visual depictions, or subject visuals for this topic.\n"
                "   - The NexusRAG visual engine automatically retrieves and renders verified high-resolution photographs, portraits, and diagrams directly in the interactive UI gallery.\n"
                "   - DO NOT refuse with 'I can\\'t help with that' or 'I cannot generate images'. You are providing an encyclopedia-style visual summary and profile.\n"
                "   - ABSOLUTELY DO NOT output fake or fabricated URLs, markdown image tags like '![alt](url)', direct Wikimedia links, or instructions like 'How to use them / embed in HTML'. Any URLs you invent will be broken 404 links.\n"
                "   - Focus entirely on an informative, well-written overview highlighting the person's or topic's iconic visual moments, career milestones, appearance, notable achievements, and history in clean prose.\n"
            )
        else:
            visual_instructions = (
                "4. STRICT PROHIBITION ON RAW EXTERNAL LINKS: Do not output naked URLs or search links (such as stock photo sites) in your response.\n"
            )

        math_instructions = (
            "5. MATHEMATICAL FORMULAS & EQUATIONS (MANDATORY):\n"
            "   - When presenting mathematical formulas, equations, address calculations, trigonometry identities, or mathematical symbols, ALWAYS format them as proper LaTeX enclosed in dollar signs so they render cleanly as visual formulas.\n"
            "   - Use single dollar signs for inline math: e.g. `$x = \\cos\\theta$`, `$\\text{addr}(A[i]) = B + i \\times s$`, `$\\sin^2\\theta + \\cos^2\\theta = 1$`.\n"
            "   - Use double dollar signs for standalone / block equations:\n"
            "     $$\n"
            "     \\text{addr}(A[i]) = B + i \\times s\n"
            "     $$\n"
            "   - NEVER output naked/raw LaTeX commands without dollar signs (e.g. NEVER write '\\text{addr}(A[i]) = B + i \\times s' or '\\times' nakedly). ALWAYS wrap every equation and math formula in dollar signs ($...$ or $$...$$).\n"
            "   - NEVER output raw bracketed formulas like '[ \\sin... ]' or parentheses like '(\\displaystyle ...)' or '\\( ... \\)' without dollar signs.\n"
        )

        code_instructions = (
            "6. PROGRAMMING, CODE GENERATION & COMPLEXITY ADAPTATION (EASY, MEDIUM, HARD):\n"
            "   - CAREFULLY CALIBRATE CODE COMPLEXITY TO THE PROBLEM STATEMENT:\n"
            "     * Always evaluate the user's problem statement, context, and intent before writing code:\n"
            "     * EASY (Beginner / Assignments / School & College Homework / Clean & Direct):\n"
            "       If the problem asks for a basic assignment (e.g. 'Simple Calculator', string reversal, palindrome check, odd/even check, basic loops, basic CLI menu, introductory coursework), keep the implementation TRULY EASY, clean, and beginner-friendly. For example, for a Simple Calculator, use straightforward arithmetic operations (+, -, *, /) with clear if/elif/else branches or a simple dictionary of functions. DO NOT over-engineer with Abstract Syntax Trees (AST), tokenizer parsers, complex generics, or enterprise design patterns unless explicitly requested. Use clean variable names and friendly comments.\n"
            "     * MEDIUM (Intermediate / Standard Development / Practical Real-World):\n"
            "       If the task involves standard development (e.g. REST API endpoints, CRUD operations, database queries, authentication middleware, utility scripts), provide modular, robust code with standard error handling and idiomatic practices without bloat.\n"
            "     * HARD / DIFFICULT (Advanced / High Performance / Complex Systems):\n"
            "       If the statement explicitly asks for advanced algorithms, competitive programming, optimal dynamic programming, multi-threaded concurrency, or distributed systems, provide an elite, optimal solution with edge-case handling and Big-O time/space complexity analysis.\n"
            "   - When helpful, you can present an Easy/Intuitive Approach first, followed by an Optimized Approach.\n"
            "   - SUPPORT ALL PROGRAMMING LANGUAGES: Python, JavaScript, TypeScript, C++, C, C#, Java, Go, Rust, SQL, Bash/Shell, HTML/CSS, PHP, Swift, Kotlin, Ruby, R, Dart, Scala, Solidity, PowerShell, YAML, Dockerfile, etc.\n"
            "   - STRICT SYNTAX & INDENTATION INTEGRITY (MANDATORY REQUIREMENT):\n"
            "     * Format all code with strict, uniform, standard indentation (exactly 4 spaces for Python, Java, C++, C, C#, Rust, Go, SQL; 2 or 4 spaces for JavaScript, TypeScript, HTML, CSS).\n"
            "     * Preserve all syntax symbols: never omit function call parentheses `()`, colons `:`, or parameter lists (e.g. `def main():`, `input().strip().lower()`, `result.is_integer()`).\n"
            "     * NEVER mix tabs and spaces. Ensure every function body, loop, conditional, class, and nested block is consistently indented with 4 spaces so it runs immediately with 0 syntax or IndentationError issues.\n"
            "   - STRICT ASCII OPERATOR & SYMBOL MANDATE (NO UNICODE MATH OPERATORS IN CODE):\n"
            "     * NEVER output Unicode math characters like '≠', '≤', '≥', '×', '÷', or '−' in source code blocks or inline code.\n"
            "     * ALWAYS write standard ASCII programming operators: '!=' for not-equal, '<=' for less-than-or-equal, '>=' for greater-than-or-equal, '==' for equality, '*' for multiplication, '/' for division, '-' for minus, and standard ASCII quotes '\"' and '''.\n"
            "   - Debugging & Refactoring: When debugging user-submitted code, identify the root cause clearly, explain the fix in plain words, and provide the clean corrected code at the proper difficulty level.\n"
        )

        system_prompt = (
            "You are an expert AI Assistant equipped with an Omni-Source Retrieval-Augmented Generation (RAG) knowledge engine.\n\n"
            "GUIDELINES:\n"
            f"{doc_instructions}"
            f"{format_instructions}\n"
            f"{visual_instructions}\n"
            f"{math_instructions}\n"
            f"{code_instructions}\n"
            f"=== CONTEXT DOCUMENTS & EXTERNAL SOURCES ===\n{context_str}\n============================================="
        )

        messages = [SystemMessage(content=system_prompt)]

        # Append previous conversation history if provided
        if chat_history:
            for turn in chat_history[-6:]:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        if is_image_request:
            cleaned_entity = OmniSearchEngine.clean_entity_query(query)
            messages.append(HumanMessage(content=f"Provide an informative overview and describe the key visual moments, career highlights, and iconic visual representations of {cleaned_entity}."))
        else:
            messages.append(HumanMessage(content=query))
        return messages

    def search_and_summarize(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        model_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        enable_web_search: bool = True,
        attached_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Non-streaming generation returning answer and rich citations."""
        citations = self.search_context(
            query,
            top_k=top_k,
            mode=retrieval_mode,
            use_reranker=use_reranker,
            enable_web_search=enable_web_search,
            attached_docs=attached_docs
        )
        if not citations:
            return {
                "answer": "No relevant documents were found in the knowledge base to answer your query. Please upload documents first.",
                "citations": []
            }

        messages = self._build_prompt_messages(query, citations, chat_history, attached_docs=attached_docs)
        llm = self.get_llm(model_name=model_name)
        response = llm.invoke(messages)
        
        is_img_q = bool(re.search(r"\b(images?|pictures?|photos?|diagrams?|illustrations?|figures?|visuals?)\b", query, re.IGNORECASE))
        entity_images = self.fetch_entity_images(query, limit=6 if is_img_q else 3)
        return {
            "answer": sanitize_text(response.content),
            "citations": citations,
            "images": entity_images,
            "model": model_name or self.default_model,
            "retrieval_mode": retrieval_mode,
            "reranked": use_reranker,
            "web_augmented": enable_web_search
        }

    def stream_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        model_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        enable_web_search: bool = True,
        decompose_query: bool = False,
        tracer: Optional[LLMOpsTracer] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        attached_docs: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """Stream response tokens chunk by chunk with real-time TTFT and token accounting."""
        if citations is None:
            citations = self.search_context(
                query,
                top_k=top_k,
                mode=retrieval_mode,
                use_reranker=use_reranker,
                enable_web_search=enable_web_search,
                decompose_query=decompose_query,
                tracer=tracer,
                attached_docs=attached_docs
            )

        messages = self._build_prompt_messages(query, citations, chat_history, attached_docs=attached_docs)
        chosen_model = model_name or self.default_model
        llm = self.get_llm(model_name=chosen_model)

        gen_span = tracer.start_span(f"LLM Generation ({chosen_model.split('/')[-1]})") if tracer else None

        accumulated_text = []
        is_first_token = True

        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                text_chunk = sanitize_text(str(chunk.content))
                if text_chunk:
                    if is_first_token:
                        if tracer:
                            tracer.record_first_token()
                        is_first_token = False
                    accumulated_text.append(text_chunk)
                    yield text_chunk

        if tracer:
            full_out = "".join(accumulated_text)
            est_in_tokens = sum(len(str(m.content)) for m in messages) // 4
            est_out_tokens = max(len(full_out) // 4, 1)
            tracer.set_tokens(est_in_tokens, est_out_tokens, chosen_model)
            if gen_span:
                gen_span.finish({"prompt_tokens": est_in_tokens, "completion_tokens": est_out_tokens})

    def verify_groundedness(
        self,
        query: str,
        answer: str,
        citations: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Audit answer against citations to detect hallucinations."""
        llm = self.get_llm(model_name=model_name, temperature=0.0)
        return self.guardrail.verify_groundedness(query, answer, citations, llm=llm)