#!/usr/bin/env python3
"""
One-command web launcher for SoundID.

What it does:
1) Starts the FastAPI backend in Web/api.py
2) Waits for /health to respond
3) Opens the web page in the default browser

Cross-platform: Windows, macOS, Linux.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SoundID web UI + API")
    parser.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="API port (default: 5000)")
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for API health before opening browser (default: 20)",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install missing web dependencies",
    )
    return parser.parse_args()


def ensure_dependencies(requirements_file: Path, no_install: bool) -> None:
    missing = []
    for module in ("fastapi", "uvicorn"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if not missing:
        return

    if no_install:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing dependencies: {joined}. "
            f"Install them with: {sys.executable} -m pip install -r {requirements_file}"
        )

    print(f"[setup] Installing missing dependencies: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("Dependency installation failed.")


def wait_for_health(base_url: str, timeout_s: float) -> bool:
    health_url = f"{base_url}/health"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.3)

    return False


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parent
    web_dir = root / "Web"
    html_file = web_dir / "esiea_instrument_detector.html"
    requirements_file = web_dir / "requirements.txt"

    if not web_dir.exists():
        print(f"[error] Web directory not found: {web_dir}")
        return 1
    if not html_file.exists():
        print(f"[error] HTML file not found: {html_file}")
        return 1
    if not requirements_file.exists():
        print(f"[error] requirements.txt not found: {requirements_file}")
        return 1

    try:
        ensure_dependencies(requirements_file, no_install=args.no_install)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

    api_base = f"http://{args.host}:{args.port}"
    html_url = f"{html_file.as_uri()}?api={api_base}&autoping=1"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print(f"[run] Starting API on {api_base}")
    print(f"[run] Command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=str(web_dir))

    try:
        healthy = wait_for_health(api_base, args.health_timeout)

        if healthy:
            print("[ok] API is healthy.")
            if not args.no_browser:
                print(f"[open] {html_url}")
                webbrowser.open(html_url)
        else:
            print("[warn] API health check timed out.")
            print(f"[hint] You can still open this page manually: {html_url}")

        print("[info] Press Ctrl+C to stop the API server.")
        return process.wait()

    except KeyboardInterrupt:
        print("\n[stop] Shutting down API server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
