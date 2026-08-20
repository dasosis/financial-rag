"""
start.py — Boot the Financial Report Q&A Engine from scratch.

Steps:
  1. Verify .env and OPENAI_API_KEY are present.
  2. Ingest PDFs (skipped when chroma_db already exists; use --reingest to force).
  3. Start the FastAPI server on port 8000.
  4. Start the Streamlit UI on port 8501.
  5. Wait until both ports are accepting connections, then open the browser.

Usage:
    python start.py            # skip ingestion if chroma_db exists
    python start.py --reingest # always wipe chroma_db and re-ingest
"""

from __future__ import annotations

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
ENV_FILE = ROOT / ".env"

API_HOST = "127.0.0.1"
API_PORT = 8000
UI_PORT = 8501

PYTHON = sys.executable          # same interpreter / venv that launched this script
READY_TIMEOUT = 60               # seconds to wait for each server to become ready
POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def _abort(msg: str) -> None:
    print(f"[start] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, label: str) -> None:
    _print(f"Waiting for {label} on {host}:{port} …")
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if _port_open(host, port):
            _print(f"{label} is ready.")
            return
        time.sleep(POLL_INTERVAL)
    _abort(f"{label} did not become ready within {READY_TIMEOUT}s.")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_env() -> None:
    if not ENV_FILE.exists():
        _abort(
            f".env not found at {ENV_FILE}.\n"
            "  Copy .env.example → .env and set OPENAI_API_KEY."
        )
    content = ENV_FILE.read_text()
    if "OPENAI_API_KEY" not in content or "sk-" not in content:
        _abort("OPENAI_API_KEY does not look set in .env.")
    _print(".env OK.")


def check_data() -> None:
    pdfs = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if not pdfs:
        _abort(
            f"No PDF files found in {DATA_DIR}.\n"
            "  Add your 10-K PDF files there and re-run."
        )
    _print(f"Found {len(pdfs)} PDF(s): {', '.join(p.name for p in pdfs)}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_ingest() -> None:
    _print("Running ingest.py …")
    result = subprocess.run([PYTHON, str(ROOT / "ingest.py")], cwd=ROOT)
    if result.returncode != 0:
        _abort("Ingestion failed. Check the output above.")
    _print("Ingestion complete.")


def start_api() -> subprocess.Popen:
    _print(f"Starting FastAPI server on port {API_PORT} …")
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "uvicorn",
            "api:app",
            "--host", API_HOST,
            "--port", str(API_PORT),
        ],
        cwd=ROOT,
    )
    return proc


def start_streamlit() -> subprocess.Popen:
    _print(f"Starting Streamlit on port {UI_PORT} …")
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "streamlit", "run",
            str(ROOT / "app.py"),
            "--server.port", str(UI_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
    )
    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Boot the Financial RAG Q&A engine.")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Wipe chroma_db and re-ingest all PDFs even if the store already exists.",
    )
    args = parser.parse_args()

    _print("=== Financial Report Q&A Engine — startup ===")

    check_env()
    check_data()

    # Ingestion
    if args.reingest:
        if CHROMA_DIR.exists():
            _print("--reingest: removing existing chroma_db …")
            shutil.rmtree(CHROMA_DIR)
        run_ingest()
    elif not CHROMA_DIR.exists():
        _print("chroma_db not found — running ingestion …")
        run_ingest()
    else:
        _print("chroma_db exists — skipping ingestion (pass --reingest to rebuild).")

    # Start servers
    api_proc = start_api()
    ui_proc = start_streamlit()

    # Register cleanup so Ctrl-C shuts both down cleanly
    procs = [api_proc, ui_proc]

    def _shutdown(sig, frame):
        _print("Shutting down servers …")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait until ready, then open browser
    _wait_for_port(API_HOST, API_PORT, "FastAPI")
    _wait_for_port("localhost", UI_PORT, "Streamlit")

    url = f"http://localhost:{UI_PORT}"
    _print(f"Opening browser at {url}")
    webbrowser.open(url)

    _print("All services running. Press Ctrl-C to stop.")
    try:
        # Keep the process alive so Ctrl-C works
        while True:
            # Restart a crashed server automatically
            for i, proc in enumerate(procs):
                if proc.poll() is not None:
                    label = "FastAPI" if i == 0 else "Streamlit"
                    _print(f"{label} exited unexpectedly (code {proc.returncode}) — restarting …")
                    procs[i] = start_api() if i == 0 else start_streamlit()
            time.sleep(2)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
