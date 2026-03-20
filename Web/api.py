"""
api.py — SoundID relay
========================
Reçoit le POST de la page, pipe les bytes audio sur stdin d'eval.py, renvoie le JSON.

Lancer :
    uvicorn api:app --reload --port 5000

Dépendances :
    pip install fastapi uvicorn python-multipart
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

EVAL_SCRIPT = Path(__file__).parent / "eval.py"
LOG_FILE    = Path(__file__).parent / "logs.txt"

app = FastAPI(title="SoundID API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Logging ────────────────────────────────────────────────────────────────

def log(section: str, data):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    body = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n{'═'*60}\n[{ts}] {section}\n{'─'*60}\n{body}\n")


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
    # ── Lire les bytes audio depuis le POST ────────────────────────────────
    raw = await audio.read()

    log("POST /predict", {
        "filename":     audio.filename,
        "size_bytes":   len(raw),
        "model":        model,
        "binary_mode":  binary_mode,
        "content_type": audio.content_type,
    })

    # ── Appel eval.py — les bytes audio arrivent via stdin ─────────────────
    # Signature : python eval.py <model_name> <binary_mode>
    # stdin     : bytes bruts du fichier audio
    # stdout    : JSON résultat
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        model,
        binary_mode,
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=raw,                  # bytes audio → stdin d'eval.py
            capture_output=True,
            timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        log("ERREUR", "eval.py timeout > 120s")
        raise HTTPException(status_code=504, detail="eval.py timeout (> 120s)")

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        combined = stderr or stdout or "(aucune sortie — torch installe dans le venv ?)"
        log("ERREUR eval.py", {"returncode": proc.returncode, "stderr": stderr, "stdout": stdout})
        raise HTTPException(
            status_code=500,
            detail=f"eval.py error (code {proc.returncode}):\n{combined}",
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as e:
        log("ERREUR JSON", {"raw": stdout[:400], "error": str(e)})
        raise HTTPException(status_code=500, detail=f"JSON invalide: {e}\n{stdout[:400]}")

    log("OK — réponse envoyée", result)
    return result