"""
NexusRAG Studio - Root Entrypoint for Streamlit Community Cloud and Web Deployments.
"""
import sys
import runpy
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Forward execution to src/rag/streamlit_app.py
target_script = SRC_DIR / "rag" / "streamlit_app.py"
runpy.run_path(str(target_script), run_name="__main__")
