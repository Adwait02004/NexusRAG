import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    """
    Combines Dense Vector Retrieval (FAISS) and Sparse Lexical Retrieval (BM25)
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all chunks currently stored in vectorstore metadata."""
        if not self.vectorstore.metadata:
            self.bm25 = None
            self.corpus_chunks = []
            return

        self.corpus_chunks = self.vectorstore.metadata
        tokenized_corpus = [tokenize(c.get("text", "")) for c in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[INFO] Initialized BM25 index over {len(self.corpus_chunks)} chunks.")

    def reload(self):
        """Refresh BM25 index after document additions or re-indexing."""
        self._build_bm25_index()

    def search_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using FAISS dense embeddings."""
        return self.vectorstore.query(query, top_k=top_k)

    def search_sparse(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using BM25 keyword matching."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = max(scores) if max(scores) > 0 else 1.0

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.corpus_chunks[idx]
            norm_score = float(scores[idx] / max_score)
            results.append({
                "index": int(idx),
                "score": norm_score,
                "metadata": meta,
                "text": meta.get("text", ""),
                "source": meta.get("source_name", "Unknown"),
                "page": meta.get("page", 1),
                "file_type": meta.get("file_type", ""),
                "method": "sparse_bm25"
            })
        return results

    def search_hybrid(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Merge Dense and Sparse rankings using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score(d) = sum(1 / (rrf_k + rank(d)))
        """
        candidate_k = max(top_k * 2, 10)
        dense_results = self.search_dense(query, top_k=candidate_k)
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        rrf_scores: Dict[int, float] = {}
        doc_map: Dict[int, Dict[str, Any]] = {}

        # Dense rank accumulation
        for rank, item in enumerate(dense_results, 1):
            idx = item["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))
            doc_map[idx] = item

        # Sparse rank accumulation
        for rank, item in enumerate(sparse_results, 1):
            idx = item["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))
            if idx not in doc_map:
                doc_map[idx] = item

        # Sort by merged RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        final_results = []

        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        for idx in sorted_indices[:top_k]:
            item = doc_map[idx].copy()
            item["rrf_score"] = rrf_scores[idx]
            item["score"] = round(float(rrf_scores[idx] / max_rrf), 4)
            item["method"] = "hybrid_rrf"
            final_results.append(item)

        return final_results
