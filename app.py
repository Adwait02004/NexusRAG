import os
import sys
import subprocess
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent
venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

# Automatically activate / re-launch with project .venv if executed from global python
if venv_python.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe != target_exe:
            print(f"[INFO] Switching to project virtual environment: {target_exe}")
            res = subprocess.run([str(target_exe), str(Path(__file__).resolve())] + sys.argv[1:])
            sys.exit(res.returncode)
    except Exception as e:
        print(f"[WARN] Auto-switch to .venv failed: {e}")

# Add src to sys.path
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse

if __name__ == "__main__":
    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="NexusRAG Studio • Enterprise AI & Hybrid Retrieval Platform")
    parser.add_argument("--cli", action="store_true", help="Run a quick one-off CLI query")
    parser.add_argument("--eval", action="store_true", help="Run the automated quantitative benchmark suite")
    parser.add_argument("--query", type=str, default="Explain Neural Network Architectures", help="Query string for CLI mode")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "dense", "sparse"], help="Retrieval method")
    parser.add_argument("--port", type=int, default=8500, help="Port for the Streamlit web studio")

    args = parser.parse_args()

    if args.eval:
        from evaluate import run_benchmark
        run_benchmark()
        sys.exit(0)

    if args.cli:
        from rag.search import RAGSearch
        print("\n" + "="*55)
        print("   ⚡ NEXUSRAG TERMINAL BENCHMARK MODE")
        print("="*55)
        rag_search = RAGSearch(
            persist_dir=str(PROJECT_DIR / "faiss_store"),
            data_dir=str(PROJECT_DIR / "data"),
        )
        print(f"[QUERY]: {args.query}")
        print(f"[MODE]:  {args.mode.upper()}")
        result = rag_search.search_and_summarize(args.query, top_k=3, retrieval_mode=args.mode)
        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== VERIFIED CITATIONS ===")
        for c in result.get("citations", []):
            print(f"- {c['source']} (Page {c['page']}) [Score: {c['score']}] [{c.get('retrieval_mode', 'hybrid')}]")
        print("="*55 + "\n")
    else:
        # Launch Streamlit 100% Pure Python Web Studio
        streamlit_script = PROJECT_DIR / "src" / "rag" / "streamlit_app.py"
        print("\n" + "="*65)
        print("   🚀 LAUNCHING NEXUSRAG STUDIO (100% PURE PYTHON)")
        print("="*65)
        print(f"   ► URL:           http://localhost:{args.port}")
        print(f"   ► Architecture:  Hybrid Search (BM25 + FAISS via RRF)")
        print(f"   ► Inference:     Groq Ultra-Fast API")
        print(f"   ► Evaluation:    python app.py --eval")
        print("="*65 + "\n")

        streamlit_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_script),
            f"--server.port={args.port}",
            "--server.headless=false",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false"
        ]
        subprocess.run(streamlit_cmd)