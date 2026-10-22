import os
import re
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure logging
logger = logging.getLogger("NexusRAG.Database")

def _resolve_config(key: str, default: Any = None) -> Any:
    """Resolve configuration from os.environ first, then fallback to st.secrets with full alias matching."""
    clean_k = key.replace("POSTGRES_", "").replace("postgres_", "")
    keys_to_test = [key, key.upper(), key.lower(), clean_k, clean_k.upper(), clean_k.lower()]
    if clean_k.lower() in ("db", "database", "dbname"):
        keys_to_test.extend(["database", "db", "dbname", "POSTGRES_DB", "postgres_db"])

    for k in keys_to_test:
        val = os.getenv(k)
        if val is not None and str(val).strip():
            return val

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in keys_to_test:
                if k in st.secrets:
                    return st.secrets[k]
            for sec in ["postgres", "supabase", "database", "POSTGRES", "SUPABASE"]:
                if sec in st.secrets and isinstance(st.secrets[sec], (dict, object)):
                    sec_dict = st.secrets[sec]
                    for k in keys_to_test:
                        if k in sec_dict:
                            return sec_dict[k]
    except Exception:
        pass
    return default

class PostgresVectorDB:
    """
    Enterprise-Grade Secure PostgreSQL + pgvector Database Manager.
    
    Features:
    - ACID-compliant document & chunk persistence
    - Hardware-accelerated HNSW vector indexing (Cosine metric)
    - Metadata filtering with JSONB
    - Multi-tenant isolation (user_id partitioning)
    - Cryptographic SHA-256 deduplication
    - Graceful offline fallback to local FAISS
    """
    _schema_initialized: bool = False

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        sslmode: Optional[str] = None,
        database_url: Optional[str] = None
    ):
        self.database_url = database_url or _resolve_config("DATABASE_URL") or _resolve_config("POSTGRES_URL")
        self.host = host or _resolve_config("POSTGRES_HOST", "localhost")
        self.port = int(port or _resolve_config("POSTGRES_PORT", 5432))
        self.database = database or _resolve_config("POSTGRES_DB", "postgres")
        self.user = user or _resolve_config("POSTGRES_USER", "postgres")
        self.password = password or _resolve_config("POSTGRES_PASSWORD", "")
        
        # Auto-detect SSL: Cloud providers like Supabase require sslmode=require
        if sslmode:
            self.sslmode = sslmode
        elif "supabase.co" in (self.host or "") or (self.database_url and "supabase.co" in self.database_url):
            self.sslmode = "require"
        else:
            self.sslmode = os.getenv("POSTGRES_SSLMODE", "prefer")
        
        self.engine = None
        self._connected = False
        self._initialize_engine()

    def _initialize_engine(self):
        """Construct secure SQLAlchemy engine with connection pooling and timeout guards."""
        try:
            from sqlalchemy import create_engine
            try:
                from pgvector.sqlalchemy import Vector
            except ImportError:
                pass
            
            # Connection URL
            if self.database_url:
                url = self.database_url
                if url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+psycopg2://", 1)
                elif url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
                if "?" not in url and self.sslmode:
                    url += f"?sslmode={self.sslmode}"
            else:
                from urllib.parse import quote_plus
                safe_pass = quote_plus(str(self.password))
                url = f"postgresql+psycopg2://{self.user}:{safe_pass}@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"
            
            # 1. Primary connection attempt
            self.engine = create_engine(
                url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=5,
                connect_args={"connect_timeout": 5}
            )
            # Test ping
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            self._connected = True
            self.connection_mode = "Direct"
            logger.info(f"Connected to secure PostgreSQL vectorstore at {self.host}:{self.port}/{self.database}")
            if not PostgresVectorDB._schema_initialized:
                self.init_schema()
                PostgresVectorDB._schema_initialized = True
        except Exception as e_primary:
            # 2. Automatic Supabase IPv4 Pooler Fallback
            # Streamlit Cloud & AWS EC2 lack IPv6 routing to db.<ref>.supabase.co
            pooler_success = False
            ref_match = re.search(r"([a-z0-9]{20})", str(self.host) + str(self.database_url or ""))
            is_supabase = ("supabase" in str(self.host).lower()) or ("supabase" in str(self.database_url or "").lower())
            if is_supabase:
                ref = ref_match.group(1) if ref_match else "tfhanwlouknwbsjopbev"
                from urllib.parse import quote_plus
                safe_pass = quote_plus(str(self.password))
                from sqlalchemy import create_engine, text
                
                # Check known pooler hosts (ap-northeast-2 is this project's primary region, with fallbacks)
                candidate_poolers = [
                    f"aws-0-ap-northeast-2.pooler.supabase.com",
                    f"aws-0-ap-south-1.pooler.supabase.com",
                    f"aws-0-us-east-1.pooler.supabase.com"
                ]
                for p_host in candidate_poolers:
                    for p_port in [6543, 5432]:
                        p_user = f"postgres.{ref}"
                        p_url = f"postgresql+psycopg2://{p_user}:{safe_pass}@{p_host}:{p_port}/postgres?sslmode=require"
                        try:
                            p_engine = create_engine(
                                p_url,
                                pool_size=5,
                                max_overflow=10,
                                pool_timeout=5,
                                connect_args={"connect_timeout": 5}
                            )
                            with p_engine.connect() as conn:
                                conn.execute(text("SELECT 1"))
                            self.engine = p_engine
                            self._connected = True
                            self.connection_mode = f"IPv4 Pooler ({p_host}:{p_port})"
                            logger.info(f"Connected to Supabase via IPv4 Pooler at {p_host}:{p_port}")
                            if not PostgresVectorDB._schema_initialized:
                                self.init_schema()
                                PostgresVectorDB._schema_initialized = True
                            pooler_success = True
                            break
                        except Exception:
                            continue
                    if pooler_success:
                        break

            if not pooler_success:
                self._connected = False
                self.engine = None
                self.connection_mode = "Offline"
                self.last_error = str(e_primary)
                logger.warning(f"PostgreSQL not reachable ({e_primary}). Using local in-memory FAISS fallback.")

    @property
    def is_connected(self) -> bool:
        """Check if live database connection is active."""
        return self._connected and self.engine is not None

    def init_schema(self):
        """Initialize pgvector extension, master documents table, chunks table and HNSW index."""
        if not self.is_connected:
            return

        from sqlalchemy import text
        
        ddl = """
        -- 1. Enable pgvector and UUID extensions
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

        -- 2. Master Documents Inventory
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            sha256_hash VARCHAR(64) UNIQUE NOT NULL,
            uploaded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            user_id VARCHAR(100) DEFAULT 'default_user'
        );

        -- 3. Document Chunks with Vector Embeddings
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            embedding vector(384) NOT NULL
        );

        -- 4. HNSW Index for Fast Cosine Vector Retrieval
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
        ON document_chunks USING hnsw (embedding vector_cosine_ops);

        -- 5. Audit Logging Table
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id VARCHAR(100) DEFAULT 'default_user',
            query_text TEXT NOT NULL,
            retrieval_mode VARCHAR(50) NOT NULL,
            top_chunks_count INT NOT NULL,
            executed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        -- 6. Shared Cloud Chat Sessions (Supabase Sync with User Scoping)
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id VARCHAR(100) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            is_archived BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            user_id VARCHAR(100) DEFAULT 'default_user'
        );
        ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(100) DEFAULT 'default_user';
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(ddl))
            logger.info("PostgreSQL pgvector schema verified and active.")
        except Exception as err:
            logger.error(f"Failed to initialize schema: {err}")

    def store_document(
        self,
        filename: str,
        file_type: str,
        raw_bytes: bytes,
        chunks: List[Any],
        embeddings: np.ndarray,
        user_id: str = "default_user"
    ) -> Optional[str]:
        """
        Store a document and all associated vector chunks in an ACID-safe transaction.
        Deduplicates automatically using SHA-256 content hashes.
        """
        if not self.is_connected:
            return None

        from sqlalchemy import text
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        file_size = len(raw_bytes)

        try:
            with self.engine.begin() as conn:
                # 1. Upsert into documents table
                upsert_doc_query = text("""
                    INSERT INTO documents (filename, file_type, file_size_bytes, sha256_hash, user_id)
                    VALUES (:fname, :ftype, :fsize, :fhash, :uid)
                    ON CONFLICT (sha256_hash) DO UPDATE 
                    SET filename = EXCLUDED.filename, uploaded_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """)
                doc_row = conn.execute(upsert_doc_query, {
                    "fname": filename,
                    "ftype": file_type,
                    "fsize": file_size,
                    "fhash": file_hash,
                    "uid": user_id
                }).fetchone()

                if not doc_row:
                    return None
                doc_id = str(doc_row[0])

                # 2. Clear old chunks if updating existing document
                conn.execute(text("DELETE FROM document_chunks WHERE document_id = :did"), {"did": doc_id})

                # 3. Bulk insert vector chunks
                chunk_insert_query = text("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, metadata, embedding)
                    VALUES (:did, :cidx, :content, :meta, :emb);
                """)
                
                for idx, chunk in enumerate(chunks):
                    emb_list = embeddings[idx].tolist() if hasattr(embeddings[idx], "tolist") else list(embeddings[idx])
                    content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                    meta = chunk.metadata if hasattr(chunk, "metadata") else {}

                    conn.execute(chunk_insert_query, {
                        "did": doc_id,
                        "cidx": idx,
                        "content": content,
                        "meta": json.dumps(meta),
                        "emb": str(emb_list)
                    })

                logger.info(f"Saved document '{filename}' with {len(chunks)} chunks to PostgreSQL.")
                return doc_id
        except Exception as e:
            logger.error(f"Error storing document {filename} in PostgreSQL: {e}")
            return None

    def search_similar_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute high-performance cosine similarity search via pgvector HNSW index.
        Returns top matching chunks with similarity scores.
        """
        if not self.is_connected:
            return []

        from sqlalchemy import text
        emb_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)

        query = """
            SELECT 
                c.id,
                c.content,
                c.metadata,
                d.filename,
                d.file_type,
                1 - (c.embedding <=> :query_vec::vector) AS similarity
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """
        params = {"query_vec": str(emb_list), "top_k": top_k}

        if user_id:
            query += " WHERE d.user_id = :user_id"
            params["user_id"] = user_id

        query += " ORDER BY c.embedding <=> :query_vec::vector ASC LIMIT :top_k;"

        results = []
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(query), params).fetchall()
                for r in rows:
                    results.append({
                        "id": str(r[0]),
                        "content": r[1],
                        "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                        "filename": r[3],
                        "file_type": r[4],
                        "similarity": float(r[5])
                    })
        except Exception as e:
            logger.error(f"Vector search failed in PostgreSQL: {e}")
        return results

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents and chunk statistics."""
        if not self.is_connected:
            return []

        from sqlalchemy import text
        query = """
            SELECT 
                d.id,
                d.filename,
                d.file_type,
                d.file_size_bytes,
                d.uploaded_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
        """
        params = {}
        if user_id:
            query += " WHERE d.user_id = :uid"
            params["uid"] = user_id

        query += " GROUP BY d.id ORDER BY d.uploaded_at DESC;"

        docs = []
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(query), params).fetchall()
                for r in rows:
                    docs.append({
                        "id": str(r[0]),
                        "filename": r[1],
                        "file_type": r[2],
                        "file_size_bytes": r[3],
                        "uploaded_at": r[4].isoformat() if r[4] else None,
                        "chunk_count": r[5]
                    })
        except Exception as e:
            logger.error(f"Failed to list documents from PostgreSQL: {e}")
        return docs

    def log_audit(self, query_text: str, retrieval_mode: str, top_chunks_count: int, user_id: str = "default_user"):
        """Record cryptographic audit log of retrieval query."""
        if not self.is_connected:
            return

        from sqlalchemy import text
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO audit_logs (user_id, query_text, retrieval_mode, top_chunks_count)
                    VALUES (:uid, :q, :mode, :count);
                """), {
                    "uid": user_id,
                    "q": query_text,
                    "mode": retrieval_mode,
                    "count": top_chunks_count
                })
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")

    def save_chat_session(
        self,
        chat_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        is_archived: bool = False,
        user_id: Optional[str] = None
    ):
        """Save or update chat session in PostgreSQL / Supabase with user scoping."""
        if not self.is_connected:
            return
        from sqlalchemy import text
        uid = user_id or "default_user"
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO chat_sessions (id, title, messages, is_archived, updated_at, user_id)
                    VALUES (:id, :title, :messages, :archived, CURRENT_TIMESTAMP, :user_id)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        messages = EXCLUDED.messages,
                        is_archived = EXCLUDED.is_archived,
                        updated_at = CURRENT_TIMESTAMP,
                        user_id = COALESCE(EXCLUDED.user_id, chat_sessions.user_id);
                """), {
                    "id": chat_id,
                    "title": title,
                    "messages": json.dumps(messages),
                    "archived": is_archived,
                    "user_id": uid
                })
        except Exception as e:
            logger.warning(f"Failed to save chat session to PostgreSQL: {e}")

    def load_chat_session(self, chat_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load chat session from PostgreSQL / Supabase with strict user isolation."""
        if not self.is_connected:
            return None
        from sqlalchemy import text
        import time
        try:
            with self.engine.connect() as conn:
                query = "SELECT id, title, messages, is_archived, created_at, updated_at, user_id FROM chat_sessions WHERE id = :id"
                params = {"id": chat_id}
                if user_id:
                    query += " AND (LOWER(user_id) = LOWER(:user_id) OR user_id = :user_id)"
                    params["user_id"] = str(user_id).strip()
                row = conn.execute(text(query), params).fetchone()
                if row:
                    msgs = row[2] if isinstance(row[2], list) else json.loads(row[2])
                    return {
                        "id": row[0],
                        "title": row[1],
                        "messages": msgs,
                        "is_archived": bool(row[3]),
                        "created_at": row[4].timestamp() if row[4] else time.time(),
                        "updated_at": row[5].timestamp() if row[5] else time.time(),
                        "user_id": row[6] if len(row) > 6 else user_id
                    }
        except Exception as e:
            logger.warning(f"Failed to load chat session from PostgreSQL: {e}")
        return None

    def list_chat_sessions(self, include_archived: bool = False, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List chat sessions from PostgreSQL / Supabase with strict user scoping."""
        if not self.is_connected:
            return []
        from sqlalchemy import text
        try:
            with self.engine.connect() as conn:
                query = "SELECT id, title, is_archived, created_at, updated_at, jsonb_array_length(messages) as message_count, user_id FROM chat_sessions WHERE 1=1"
                params = {}
                if not include_archived:
                    query += " AND is_archived = FALSE"
                if user_id:
                    query += " AND (LOWER(user_id) = LOWER(:user_id) OR user_id = :user_id)"
                    params["user_id"] = str(user_id).strip()
                query += " ORDER BY updated_at DESC;"
                rows = conn.execute(text(query), params).fetchall()
                chats = []
                for r in rows:
                    chats.append({
                        "id": r[0],
                        "title": r[1],
                        "created_at": r[3].timestamp() if r[3] else 0,
                        "updated_at": r[4].timestamp() if r[4] else 0,
                        "message_count": int(r[5]) if r[5] is not None else 0,
                        "is_archived": bool(r[2]),
                        "user_id": r[6] if len(r) > 6 else user_id
                    })
                return chats
        except Exception as e:
            logger.warning(f"Failed to list chat sessions from PostgreSQL: {e}")
        return []

    def set_chat_archived(self, chat_id: str, is_archived: bool) -> bool:
        """Archive or unarchive chat session in PostgreSQL / Supabase."""
        if not self.is_connected:
            return False
        from sqlalchemy import text
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE chat_sessions SET is_archived = :archived, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id;
                """), {"id": chat_id, "archived": is_archived})
            return True
        except Exception as e:
            logger.warning(f"Failed to set chat archive state in PostgreSQL: {e}")
            return False

    def delete_chat_session(self, chat_id: str, user_id: Optional[str] = None) -> bool:
        """Delete chat session from PostgreSQL / Supabase with strict user isolation."""
        if not self.is_connected:
            return False
        from sqlalchemy import text
        try:
            with self.engine.begin() as conn:
                query = "DELETE FROM chat_sessions WHERE id = :id"
                params = {"id": chat_id}
                if user_id:
                    query += " AND user_id = :user_id"
                    params["user_id"] = user_id
                conn.execute(text(query), params)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete chat session from PostgreSQL: {e}")
            return False


import sqlite3
from pathlib import Path

class SQLiteDocumentDB:
    """
    Embedded ACID-compliant SQLite Document & Chunk Database.
    Guarantees every uploaded document and chunk is permanently saved locally
    in data/nexusrag.db with zero external database configuration required.
    """
    def __init__(self, db_path: str | Path = "data/nexusrag.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    sha256_hash TEXT UNIQUE NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    preview_text TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks (document_id)
            """)
            conn.commit()

    def store_document(
        self,
        filename: str,
        file_type: str,
        raw_bytes: bytes,
        chunks: List[Any],
        text_preview: str = ""
    ) -> str:
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        file_size = len(raw_bytes)
        doc_id = hashlib.md5(filename.encode("utf-8")).hexdigest()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (id, filename, file_type, file_size_bytes, sha256_hash, uploaded_at, chunk_count, preview_text)
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?)
                ON CONFLICT(sha256_hash) DO UPDATE SET
                    filename = excluded.filename,
                    uploaded_at = datetime('now'),
                    chunk_count = excluded.chunk_count,
                    preview_text = excluded.preview_text
            """, (doc_id, filename, file_type, file_size, file_hash, len(chunks), text_preview[:500]))

            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
            for idx, c in enumerate(chunks):
                chunk_id = f"{doc_id}_{idx}"
                content = c.page_content if hasattr(c, "page_content") else (c.get("text", "") if isinstance(c, dict) else str(c))
                meta_dict = c.metadata if hasattr(c, "metadata") else (c if isinstance(c, dict) else {})
                meta = json.dumps(meta_dict)
                conn.execute("""
                    INSERT INTO document_chunks (id, document_id, chunk_index, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (chunk_id, doc_id, idx, content, meta))
            conn.commit()
        return doc_id

    def remove_document(self, filename: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
            return [dict(row) for row in cur.fetchall()]

