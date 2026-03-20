"""
api.py — SoundID relay
========================
Reçoit le POST de la page, loggue, appelle eval.py, loggue la réponse, renvoie.

Lancer :
    uvicorn api:app --reload --port 5000

Dépendances :
    pip install fastapi uvicorn python-multipart
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

EVAL_SCRIPT = Path(__file__).parent / "eval.py"
LOG_FILE    = Path(__file__).parent / "logs.txt"

app = FastAPI(title="SoundID API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Logging ────────────────────────────────────────────────────────────────

def log(section: str, data: dict | str):
    """Écrit un bloc dans logs.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line      = f"\n{'═'*60}\n[{timestamp}] {section}\n{'─'*60}\n"
    if isinstance(data, dict):
        line += json.dumps(data, ensure_ascii=False, indent=2)
    else:
        line += str(data)
    line += "\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    audio:       UploadFile = File(...),
    model:       str        = Form("LilDrill"),
    binary_mode: str        = Form("false"),
):
    raw = await audio.read()

    # ── Log du POST entrant ────────────────────────────────────────────────
    post_info = {
        "filename":    audio.filename,
        "size_bytes":  len(raw),
        "model":       model,
        "binary_mode": binary_mode,
        "content_type": audio.content_type,
    }
    log("POST /predict — requête reçue", post_info)

    # ── Fichier temporaire ─────────────────────────────────────────────────
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    # ── Appel eval.py ──────────────────────────────────────────────────────
    cmd = [sys.executable, str(EVAL_SCRIPT), tmp_path, model, binary_mode]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        log("ERREUR", {"reason": "eval.py timeout > 120s"})
        raise HTTPException(status_code=504, detail="eval.py timeout (> 120s)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        log("ERREUR eval.py", {"returncode": proc.returncode, "stderr": proc.stderr.strip()})
        raise HTTPException(
            status_code=500,
            detail=f"eval.py error (code {proc.returncode}):\n{proc.stderr.strip()}",
        )

    # ── Parser et logger la réponse ────────────────────────────────────────
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        log("ERREUR JSON", {"raw": proc.stdout[:400], "error": str(e)})
        raise HTTPException(
            status_code=500,
            detail=f"eval.py returned invalid JSON: {e}\n{proc.stdout[:400]}",
        )

    log("RÉPONSE eval.py — JSON envoyé à la page", result)
    return result
