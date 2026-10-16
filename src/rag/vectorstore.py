import os
from pathlib import Path
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
import sys
try:
    from .embedding import EmbeddingPipeline
except ImportError:
    from rag.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.persist_dir = str(Path(persist_dir).resolve())
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.emb_pipe = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.faiss_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

    def is_initialized(self) -> bool:
        return os.path.exists(self.faiss_path) and os.path.exists(self.meta_path)

    def build_from_documents(self, documents: List[Any]):
        """Rebuild index completely from a list of documents."""
        print(f"[INFO] Building vector store from {len(documents)} raw document sections...")
        if not documents:
            print("[WARN] No documents provided to build vector store.")
            self.metadata = []
            self.index = None
            return

        chunks = self.emb_pipe.chunk_documents(documents)
        embeddings = self.emb_pipe.embed_chunks(chunks)

        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["text"] = chunk.page_content
            meta["chunk_id"] = i
            meta["source_name"] = meta.get("source_name") or Path(meta.get("source", "Unknown")).name
            meta["page"] = meta.get("page", 1)
            metadatas.append(meta)

        self.index = None
        self.metadata = []
        self.add_embeddings(embeddings, metadatas)
        self.save()
        print(f"[INFO] Vector store built with {len(metadatas)} chunks and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Any]):
        """Incrementally add documents to existing index."""
        if not documents:
            return
        if self.index is None:
            if self.is_initialized():
                self.load()
            else:
                return self.build_from_documents(documents)

        chunks = self.emb_pipe.chunk_documents(documents)
        if not chunks:
            return

        embeddings = self.emb_pipe.embed_chunks(chunks)
        start_id = len(self.metadata)
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["text"] = chunk.page_content
            meta["chunk_id"] = start_id + i
            meta["source_name"] = meta.get("source_name") or Path(meta.get("source", "Unknown")).name
            meta["page"] = meta.get("page", 1)
            metadatas.append(meta)

        self.add_embeddings(embeddings, metadatas)
        self.save()
        print(f"[INFO] Incrementally added {len(metadatas)} chunks to vector store.")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if len(embeddings) == 0:
            return
        dim = embeddings.shape[1]
        if self.index is None:
            # Using IndexFlatIP for cosine similarity on normalized vectors
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index ({len(self.metadata)} records) to {self.persist_dir}")

    def load(self):
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded Faiss index ({len(self.metadata)} chunks) from {self.persist_dir}")
        else:
            print(f"[WARN] No index found at {self.persist_dir}")

    def clear(self):
        """Reset the vector store index and metadata."""
        self.index = None
        self.metadata = []
        if os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        print(f"[INFO] Vector store cleared at {self.persist_dir}")

    def remove_document(self, doc_name: str) -> int:
        """
        Instantly remove all chunks for a document in milliseconds without re-embedding.
        Reconstructs the remaining vector matrix directly from FAISS memory.
        """
        if self.index is None or not self.metadata:
            return 0

        target_base = Path(doc_name).name.lower()
        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if Path(m.get("source_name", "")).name.lower() != target_base
            and Path(m.get("source", "")).name.lower() != target_base
        ]

        removed_count = len(self.metadata) - len(keep_indices)
        if removed_count == 0:
            return 0

        dim = self.index.d
        if keep_indices:
            remaining_vectors = np.empty((len(keep_indices), dim), dtype=np.float32)
            for new_i, old_i in enumerate(keep_indices):
                self.index.reconstruct(old_i, remaining_vectors[new_i])

            new_metadata = [self.metadata[i] for i in keep_indices]
            for i, m in enumerate(new_metadata):
                m["chunk_id"] = i

            self.index = faiss.IndexFlatIP(dim)
            self.index.add(remaining_vectors)
            self.metadata = new_metadata
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

        self.save()
        print(f"[INFO] Instantly removed {removed_count} chunks for '{doc_name}' ({len(self.metadata)} remaining).")
        return removed_count

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype("float32"), k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "index": int(idx),
                "score": float(score),
                "metadata": meta,
                "text": meta.get("text", ""),
                "source": meta.get("source_name", "Unknown"),
                "page": meta.get("page", 1),
                "file_type": meta.get("file_type", "")
            })
        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.index and self.is_initialized():
            self.load()
        if not self.index or self.index.ntotal == 0:
            return []
        query_emb = self.emb_pipe.embed_query(query_text)
        return self.search(query_emb, top_k=top_k)

    def get_stats(self) -> Dict[str, Any]:
        doc_names = set(m.get("source_name", "Unknown") for m in self.metadata)
        return {
            "total_chunks": len(self.metadata),
            "total_documents": len(doc_names),
            "documents": sorted(list(doc_names)),
            "index_dimension": self.index.d if self.index else 384,
            "persist_dir": self.persist_dir
        }