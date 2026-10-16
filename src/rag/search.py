import os
import sys
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
except ImportError:
    from rag.data_loader import load_all_documents, load_single_document
    from rag.vectorstore import FaissVectorStore
    from rag.hybrid_retriever import HybridRetriever
    from rag.reranker import DocumentReranker
    from rag.multi_search import OmniSearchEngine

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = [
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (High Quality)"},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B (Ultra Fast)"},
    {"id": "qwen/qwen3.8-27b", "name": "Qwen 3.8 27B"},
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

        # Initialize PostgreSQL Vector Database connection (with graceful fallback)
        try:
            from rag.database import PostgresVectorDB
            self.pg_db = PostgresVectorDB()
        except Exception as e_db:
            print(f"[INFO] PostgreSQL client not initialized: {e_db}")
            self.pg_db = None

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

    def get_llm(self, model_name: Optional[str] = None, temperature: float = 0.2):
        api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file.")
        chosen_model = model_name or self.default_model
        return ChatGroq(groq_api_key=api_key, model_name=chosen_model, temperature=temperature)

    def reindex_all(self) -> Dict[str, Any]:
        """Reload all documents from data folder and rebuild the vector store."""
        docs = load_all_documents(self.data_path)
        self.vectorstore.build_from_documents(docs)
        self.hybrid.reload()
        return self.vectorstore.get_stats()

    def add_document_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Incrementally load and index a single uploaded document."""
        docs = load_single_document(file_path)
        if docs:
            self.vectorstore.add_documents(docs)
            self.hybrid.reload()

            # Mirror to secure PostgreSQL vectorstore if connected
            if hasattr(self, "pg_db") and self.pg_db and self.pg_db.is_connected:
                p = Path(file_path)
                try:
                    raw_bytes = p.read_bytes()
                    chunks = self.vectorstore.embedding_manager.split_documents(docs)
                    embeddings = self.vectorstore.embedding_manager.embed_documents(chunks)
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
        Instantly remove a document from FAISS, BM25, and disk in milliseconds.
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

        # 4. Remove from PostgreSQL if connected
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
        enable_web_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top matching document chunks with formatted metadata.
        Modes: 'hybrid' (BM25 + FAISS via RRF), 'dense' (FAISS only), 'sparse' (BM25 only).
        When use_reranker=True, performs Two-Stage Retrieval (Cross-Encoder scoring on candidate pool).
        When enable_web_search=True, augments with live multi-source web, Wikipedia, and ArXiv findings.
        """
        # Stage 1: Candidate Generation (retrieve wider pool if re-ranking is active)
        stage1_k = max(top_k * 3, 12) if use_reranker else top_k

        if mode == "dense":
            raw_results = self.hybrid.search_dense(query, top_k=stage1_k)
        elif mode == "sparse":
            raw_results = self.hybrid.search_sparse(query, top_k=stage1_k)
        else:
            raw_results = self.hybrid.search_hybrid(query, top_k=stage1_k)

        # Stage 2: Cross-Encoder Re-Ranking
        if use_reranker and raw_results:
            raw_results = self.reranker.rerank(query=query, candidates=raw_results, top_k=top_k)
        else:
            raw_results = raw_results[:top_k]

        citations = []
        for r in raw_results:
            score = round(float(r.get("score", 0.0)), 3)
            relevance_pct = max(0, min(100, int(score * 100)))
            
            # Discard completely unrelated chunks (0% relevance or negative cross-encoder logit)
            raw_logit = r.get("raw_rerank_score")
            if raw_logit is not None and (raw_logit < -3.0 or relevance_pct <= 5):
                continue
            elif raw_logit is None and relevance_pct <= 5:
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

        # Stage 3: Live External Search Augmentation (Wikipedia, ArXiv, Google/Serper)
        if enable_web_search and query.strip():
            try:
                external_results = self.omni_search.search_all(query, max_results=2)
                citations.extend(external_results)
            except Exception as e:
                print(f"[WARN] External search failed: {e}")

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
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Any]:
        context_blocks = []
        for i, c in enumerate(citations, 1):
            if c.get("is_web"):
                source_info = f"[Web/External Source {i}: {c['source']} | URL: {c['page']}]"
            else:
                source_info = f"[Private Document {i}: {c['source']}, Page {c['page']}]"
            context_blocks.append(f"{source_info}\n{c['full_text']}")

        if context_blocks:
            context_str = "\n\n".join(context_blocks)
            doc_instructions = (
                "1. Synthesize accurate, comprehensive, and well-structured answers using the provided Context.\n"
                "2. Whenever you state facts or insights from the context, include in-text citations referencing the document or web source (e.g., '[fundamentals_of_ai.pdf, Page 3]' or '[Wikipedia: Deep Learning]').\n"
                "3. If private documents and web resources both touch on the topic, highlight how they complement each other.\n"
            )
        else:
            context_str = "No specific private documents or external sources matched this query."
            doc_instructions = (
                "1. Answer the user's question directly, clearly, and informatively using your broad technical and general knowledge.\n"
                "2. Do not fabricate or invent citations from unmentioned documents.\n"
            )

        system_prompt = (
            "You are an expert AI Assistant equipped with an Omni-Source Retrieval-Augmented Generation (RAG) knowledge engine.\n\n"
            "GUIDELINES:\n"
            f"{doc_instructions}"
            "3. Format your answer using clear Markdown with headings, bullet points, and code blocks where applicable.\n\n"
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
        enable_web_search: bool = True
    ) -> Dict[str, Any]:
        """Non-streaming generation returning answer and rich citations."""
        citations = self.search_context(
            query,
            top_k=top_k,
            mode=retrieval_mode,
            use_reranker=use_reranker,
            enable_web_search=enable_web_search
        )
        if not citations:
            return {
                "answer": "No relevant documents were found in the knowledge base to answer your query. Please upload documents first.",
                "citations": []
            }

        messages = self._build_prompt_messages(query, citations, chat_history)
        llm = self.get_llm(model_name=model_name)
        response = llm.invoke(messages)
        
        return {
            "answer": sanitize_text(response.content),
            "citations": citations,
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
        enable_web_search: bool = True
    ) -> Generator[str, None, None]:
        """Stream response tokens chunk by chunk."""
        citations = self.search_context(
            query,
            top_k=top_k,
            mode=retrieval_mode,
            use_reranker=use_reranker,
            enable_web_search=enable_web_search
        )
        messages = self._build_prompt_messages(query, citations, chat_history)
        llm = self.get_llm(model_name=model_name)
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield sanitize_text(str(chunk.content))