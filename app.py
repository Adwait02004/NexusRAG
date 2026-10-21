import os
import sys
import subprocess
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent
venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

# 1. Ensure UTF-8 encoding across Windows terminal and subprocesses
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 2. Automatically re-launch with project .venv if executed from global python in terminal
if venv_python.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe != target_exe:
            env = os.environ.copy()
            res = subprocess.run([str(target_exe), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
            sys.exit(res.returncode)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        pass

def _ensure_streamlit_secrets():
    """Synthesize .streamlit/secrets.toml from environment variables if running in Hugging Face Spaces or container."""
    try:
        secrets_path = PROJECT_DIR / ".streamlit" / "secrets.toml"
        if secrets_path.exists() and secrets_path.stat().st_size > 50:
            return
        
        client_id = os.getenv("CLERK_CLIENT_ID")
        client_secret = os.getenv("CLERK_CLIENT_SECRET")
        if not (client_id and client_secret):
            return

        metadata_url = os.getenv(
            "CLERK_SERVER_METADATA_URL",
            "https://advanced-porpoise-7274.clerk.accounts.dev/.well-known/openid-configuration",
        )
        cookie_secret = os.getenv(
            "CLERK_COOKIE_SECRET",
            os.getenv("STREAMLIT_AUTH_COOKIE_SECRET", "8a76145ec7740c84fd18912b8b52129564bd63c71a683a3b74cd808f2b17a5a8"),
        )
        redirect_uri = os.getenv("CLERK_REDIRECT_URI", "")

        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Auto-generated from environment variables for Hugging Face Spaces / Cloud deployment",
            "[auth]",
            f'cookie_secret = "{cookie_secret}"',
            f'client_id = "{client_id}"',
            f'client_secret = "{client_secret}"',
            f'server_metadata_url = "{metadata_url}"',
        ]
        if redirect_uri:
            lines.append(f'redirect_uri = "{redirect_uri}"')
        lines.extend([
            "",
            "[auth.client_kwargs]",
            'scope = "openid profile email"',
            'prompt = "login"',
            "",
            "[auth.clerk]",
            f'client_id = "{client_id}"',
            f'client_secret = "{client_secret}"',
            f'server_metadata_url = "{metadata_url}"',
            "",
            "[auth.clerk.client_kwargs]",
            'scope = "openid profile email"',
            'prompt = "login"',
            ""
        ])
        secrets_path.write_text("\n".join(lines), encoding="utf-8")
        print("[INFO] Synthesized .streamlit/secrets.toml from environment variables.")
    except Exception as exc:
        print(f"[WARN] Could not synthesize .streamlit/secrets.toml: {exc}")

_ensure_streamlit_secrets()

# 3. Check if running inside Streamlit runtime (e.g. Hugging Face Spaces or Streamlit Cloud)
# Use streamlit.runtime.exists() which is completely silent and produces zero warnings
try:
    from streamlit.runtime import exists as _st_runtime_exists
    if _st_runtime_exists():
        import runpy
        import streamlit as st
        _studio_path = PROJECT_DIR / "src" / "rag" / "streamlit_app.py"
        runpy.run_path(str(_studio_path), run_name="__main__")
        st.stop()
except Exception:
    pass

# Add src to sys.path
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse

def disable_windows_quickedit():
    """Disable QuickEdit Mode in Windows console to prevent clicks from freezing execution."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # STD_INPUT_HANDLE = -10
            h_stdin = kernel32.GetStdHandle(-10)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
                # ENABLE_QUICK_EDIT_MODE = 0x0040, ENABLE_EXTENDED_FLAGS = 0x0080
                new_mode = (mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_stdin, new_mode)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        disable_windows_quickedit()
        os.environ["PYTHONUNBUFFERED"] = "1"

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
            # Launch Streamlit 100% Pure Python Web Studio with supervisor loop
            streamlit_script = PROJECT_DIR / "src" / "rag" / "streamlit_app.py"

            streamlit_cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_script),
                f"--server.port={args.port}",
                "--server.address=localhost",
                "--server.headless=false",
                "--server.fileWatcherType=auto",
                "--server.runOnSave=true",
                "--browser.gatherUsageStats=false"
            ]

            print("\n" + "=" * 58)
            print("  ⚡ NexusRAG Studio is launching...")
            print(f"  🌐 Local URL: http://localhost:{args.port}")
            print("=" * 58 + "\n")

            import time
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            while True:
                try:
                    res = subprocess.run(streamlit_cmd, env=env)
                    if res.returncode == 0:
                        break
                    print("\n" + "="*58)
                    print(f"  ⚠️  Studio process exited with code {res.returncode}.")
                    print("  🔄 Terminal Supervisor: Keeping server alive...")
                    print("  Waiting 2s before auto-restarting smoothly...")
                    print("="*58 + "\n")
                    time.sleep(2)
                except KeyboardInterrupt:
                    print("\n[INFO] Studio stopped cleanly by user.")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Studio stopped cleanly.")
        sys.exit(0)