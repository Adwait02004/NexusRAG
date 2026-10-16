import math
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class DocumentReranker:
    """
    Two-Stage Information Retrieval Cross-Encoder Re-ranker.
    Scores joint (Query, Document) token pairs with cross-attention to
    substantially improve precision over bi-encoder vector cosine similarity.
    """
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        print(f"[INFO] Initializing Cross-Encoder Re-Ranker: {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Cross-Encoder Re-Ranker ready.")

    def _sigmoid(self, x: float) -> float:
        """Convert raw logits to a calibrated [0.0, 1.0] probability."""
        return 1.0 / (1.0 + math.exp(-max(min(x, 15.0), -15.0)))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Re-score candidate chunks using full cross-attention.
        
        Args:
            query: The user prompt or retrieval query.
            candidates: Initial candidate chunks retrieved by hybrid or vector search.
            top_k: Number of highest-confidence passages to retain.
            
        Returns:
            List of chunks re-ordered by cross-encoder relevance.
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        # Prepare (query, text) pairs
        pairs = []
        for c in candidates:
            # Truncate chunk text safely to fit cross-encoder max sequence length
            text = c.get("text", "")[:1200]
            pairs.append([query, text])

        try:
            raw_scores = self.model.predict(pairs)
        except Exception as e:
            print(f"[WARN] Cross-Encoder prediction failed: {e}. Falling back to candidate order.")
            return candidates[:top_k]

        scored_candidates = []
        for item, score in zip(candidates, raw_scores):
            entry = item.copy()
            entry["raw_rerank_score"] = float(score)
            calibrated_prob = self._sigmoid(float(score))
            entry["rerank_score"] = round(calibrated_prob, 4)
            entry["score"] = entry["rerank_score"]
            entry["method"] = f"{item.get('method', 'retrieval')} + cross_encoder"
            scored_candidates.append(entry)

        # Sort descending by re-ranking score
        scored_candidates.sort(key=lambda x: x["raw_rerank_score"], reverse=True)
        return scored_candidates[:top_k]
