import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project and src are on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag.search import RAGSearch

# Ground-truth test dataset for evaluating retrieval effectiveness
EVALUATION_DATASET = [
    {
        "query": "What are the common types of neural network architectures?",
        "expected_doc": "fundamentals_of_ai.pdf"
    },
    {
        "query": "What are the key concepts and applications of Python?",
        "expected_doc": "python_intro.txt"
    },
    {
        "query": "How is machine learning defined and what are supervised learning algorithms?",
        "expected_doc": "machine_learning.txt"
    },
    {
        "query": "How do AI and Data Science converge in business solutions?",
        "expected_doc": "ai_data_science_convergence.pdf"
    },
    {
        "query": "What are the fundamental techniques in data science practice?",
        "expected_doc": "essentials_of_data_science.pdf"
    }
]

def run_benchmark():
    print("=" * 70)
    print("   NEXUSRAG QUANTITATIVE RETRIEVAL BENCHMARK SUITE")
    print("=" * 70)
    
    rag = RAGSearch()
    methods = [
        {"id": "dense", "name": "Dense (FAISS)", "mode": "dense", "rerank": False},
        {"id": "sparse", "name": "Sparse (BM25)", "mode": "sparse", "rerank": False},
        {"id": "hybrid", "name": "Hybrid (RRF)", "mode": "hybrid", "rerank": False},
        {"id": "reranked", "name": "Two-Stage (RRF + Cross-Encoder)", "mode": "hybrid", "rerank": True},
    ]

    metrics: Dict[str, Dict[str, Any]] = {
        m["id"]: {
            "name": m["name"],
            "hits": 0,
            "total": len(EVALUATION_DATASET),
            "total_latency": 0.0,
            "scores": []
        }
        for m in methods
    }

    for item in EVALUATION_DATASET:
        query = item["query"]
        expected = item["expected_doc"].lower()
        print(f"\n[QUERY]: '{query}' (Target: {expected})")

        for m_cfg in methods:
            m_id = m_cfg["id"]
            t0 = time.perf_counter()
            citations = rag.search_context(
                query,
                top_k=3,
                mode=m_cfg["mode"],
                use_reranker=m_cfg["rerank"]
            )
            latency = (time.perf_counter() - t0) * 1000.0  # in ms

            retrieved_sources = [c["source"].lower() for c in citations]
            is_hit = any(expected in s for s in retrieved_sources)

            if is_hit:
                metrics[m_id]["hits"] += 1

            avg_score = sum(c["score"] for c in citations) / len(citations) if citations else 0.0
            metrics[m_id]["scores"].append(avg_score)
            metrics[m_id]["total_latency"] += latency

            status_label = "[PASS]" if is_hit else "[FAIL]"
            top_src = citations[0]['source'] if citations else 'None'
            print(f"  {status_label:<6} [{m_cfg['name']:<30}] Hit: {str(is_hit):<5} | Latency: {latency:5.1f}ms | Top Source: {top_src}")

    print("\n" + "=" * 75)
    print("   BENCHMARK RESULTS & RETRIEVAL COMPARISON")
    print("=" * 75)
    print(f"| {'Method':<32} | {'Hit Rate @ 3':<14} | {'Avg Latency':<13} |")
    print(f"|{'-'*34}|{'-'*16}|{'-'*15}|")

    for m_cfg in methods:
        m_id = m_cfg["id"]
        m = metrics[m_id]
        hit_rate = (m["hits"] / m["total"]) * 100.0
        avg_lat = m["total_latency"] / m["total"]
        print(f"| {m['name']:<32} | {hit_rate:>6.1f}%{'':<7} | {avg_lat:>6.1f} ms{'':<5} |")

    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_benchmark()
