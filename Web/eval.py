"""
eval.py — SoundID inference engine
====================================
Appelé par api.py via subprocess.

Usage :
    python eval.py <audio_path> <model_name> <binary_mode>

    audio_path   : chemin absolu vers le fichier audio temporaire
    model_name   : LilDrill | Brrrevet | Silver | MonkI
    binary_mode  : true | false

Sortie : JSON sur stdout, une seule ligne, capturé par api.py.

JSON de sortie :
{
  "model_name":   "Silver",
  "binary_mode":  false,
  "duration_s":   19.2,
  "n_windows":    7,
  "window_s":     3.0,

  "summary": {
    "gac": { "presence_ratio": 0.43, "presence_count": 3, "dominant": false },
    "org": { "presence_ratio": 0.14, "presence_count": 1, "dominant": false },
    "pia": { "presence_ratio": 0.71, "presence_count": 5, "dominant": true  },
    "voi": { "presence_ratio": 0.57, "presence_count": 4, "dominant": false }
  },

  "timeline": [
    {
      "window_index": 0,
      "start_s":  0.0,
      "end_s":    3.0,
      "winner":   "pia",
      "scores":   { "gac": 0.12, "org": 0.05, "pia": 0.78, "voi": 0.05 }
    },
    ...
  ],

  "processed_in_ms": 312.4
}

Notes :
- En mode binaire, chaque fenêtre ne produit qu'un score de 1.0 pour le gagnant,
  0.0 pour les autres. La présence ratio reste calculée de la même façon.
- Le "dominant" dans summary indique l'instrument le plus souvent gagnant.
- Architecture CNN attendue : ConvBlock(1→32→64→128) + AdaptiveAvgPool2d(4,4)
  + Linear(2048→256→4). À adapter si ton train.py diffère.
"""

import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════

INSTRUMENTS = ["gac", "org", "pia", "voi"]
SAMPLE_RATE = 16_000
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
WINDOW_S    = 3.0       # durée d'une fenêtre = durée train
# Pas de chevauchement : chaque fenêtre est indépendante, pas de vote redondant
# Si tu veux du 50% overlap, change STEP_S = 1.5

STEP_S      = 3.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR  = PROJECT_ROOT / "Models"

# ══════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE — doit correspondre à ton train.py
# ══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.block(x)


class InstrumentCNN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_model(name: str) -> InstrumentCNN:
    # Le front/API est verrouille sur best_model; on force ce checkpoint ici aussi.
    name = "best_model"
    # Supporte .pth et .pt
    path = None
    for ext in (".pth", ".pt"):
        candidate = (MODELS_DIR / f"{name}{ext}").resolve()
        if candidate.exists():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(f"Modele introuvable : {MODELS_DIR / name}(.pth|.pt)")

    model = InstrumentCNN(n_classes=len(INSTRUMENTS))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Accepte soit un state_dict brut, soit un checkpoint complet de train.py.
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_audio_from_stdin() -> torch.Tensor:
    """
    Lit les bytes audio depuis sys.stdin.
    Attend du PCM WAV (encodé par le navigateur via AudioContext avant envoi).
    Décodage avec le module 'wave' de la stdlib — zéro dépendance externe.
    Retourne waveform float32 (1, N) resampleé à SAMPLE_RATE.
    """
    import wave
    import numpy as np

    raw = sys.stdin.buffer.read()
    buf = io.BytesIO(raw)

    with wave.open(buf) as wf:
        n_channels   = wf.getnchannels()
        sample_width = wf.getsampwidth()   # bytes par sample
        sr           = wf.getframerate()
        n_frames     = wf.getnframes()
        raw_pcm      = wf.readframes(n_frames)

    # Décode selon la profondeur de bits
    if sample_width == 2:       # 16-bit PCM — le plus courant
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:     # 32-bit int
        samples = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:     # 8-bit unsigned
        samples = (np.frombuffer(raw_pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"sample_width={sample_width} non supporté")

    # Reshape en (channels, samples) puis moyenne → mono
    samples  = samples.reshape(-1, n_channels).T       # (channels, samples)
    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform[:1]

    # Resample si nécessaire
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform


def slice_windows(waveform: torch.Tensor) -> list[dict]:
    """
    Découpe le waveform en fenêtres de WINDOW_S secondes, pas de STEP_S.
    La dernière fenêtre est paddée avec du silence si elle est incomplète.
    Retourne une liste de dicts : { tensor, start_s, end_s }
    """
    win_n  = int(WINDOW_S * SAMPLE_RATE)
    step_n = int(STEP_S   * SAMPLE_RATE)
    total  = waveform.shape[1]

    # Si le fichier est plus court qu'une fenêtre, on padde et on a 1 fenêtre
    if total < win_n:
        pad      = torch.zeros(1, win_n - total)
        waveform = torch.cat([waveform, pad], dim=1)
        total    = win_n

    windows = []
    start   = 0
    while start < total:
        end    = start + win_n
        chunk  = waveform[:, start:end]

        # Dernière fenêtre potentiellement trop courte → pad
        if chunk.shape[1] < win_n:
            pad   = torch.zeros(1, win_n - chunk.shape[1])
            chunk = torch.cat([chunk, pad], dim=1)

        windows.append({
            "tensor":  chunk,
            "start_s": round(start / SAMPLE_RATE, 3),
            "end_s":   round(min(end, total) / SAMPLE_RATE, 3),
        })
        start += step_n

    return windows


def mel_transform_fn():
    mel = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    db  = T.AmplitudeToDB()
    return lambda w: db(mel(w))


def normalize(mel: torch.Tensor) -> torch.Tensor:
    return (mel - mel.mean()) / (mel.std() + 1e-6)


# ══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def run_inference(
    model:       InstrumentCNN,
    windows:     list[dict],
    binary_mode: bool,
) -> tuple[list[dict], dict]:
    """
    Appelle le modèle UNE FOIS PAR FENÊTRE.
    Retourne (timeline, summary).
    """
    to_mel = mel_transform_fn()

    # ── Compteurs pour le summary ──────────────────────────────────────────
    presence_count = {inst: 0 for inst in INSTRUMENTS}
    n = len(windows)

    timeline = []

    with torch.no_grad():
        for idx, win in enumerate(windows):
            mel    = to_mel(win["tensor"])             # (1, N_MELS, T)
            mel    = normalize(mel)
            inp    = mel.unsqueeze(0)                  # (1, 1, N_MELS, T)
            logits = model(inp).squeeze(0)             # (4,)
            probs  = torch.sigmoid(logits)             # scores bruts [0,1]

            winner_idx = int(probs.argmax())
            winner     = INSTRUMENTS[winner_idx]

            if binary_mode:
                scores = {inst: (1.0 if inst == winner else 0.0)
                          for inst in INSTRUMENTS}
            else:
                scores = {inst: round(float(probs[i]), 4)
                          for i, inst in enumerate(INSTRUMENTS)}

            presence_count[winner] += 1

            timeline.append({
                "window_index": idx,
                "start_s":      win["start_s"],
                "end_s":        win["end_s"],
                "winner":       winner,
                "scores":       scores,
            })

    # ── Summary ───────────────────────────────────────────────────────────
    dominant = max(presence_count, key=presence_count.get)
    summary  = {
        inst: {
            "presence_ratio": round(presence_count[inst] / n, 4),
            "presence_count": presence_count[inst],
            "dominant":       (inst == dominant),
            # liste des indices de fenêtres où cet instrument est winner
            "window_indices": [
                e["window_index"] for e in timeline if e["winner"] == inst
            ],
        }
        for inst in INSTRUMENTS
    }

    return timeline, summary


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 3:
        _die("Usage: eval.py <model_name> <binary_mode>  (audio via stdin)")

    model_name   = sys.argv[1]
    binary_mode  = sys.argv[2].lower() == "true"

    t0 = time.perf_counter()

    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        _die(str(e))

    try:
        waveform = load_audio_from_stdin()
    except Exception as e:
        _die(f"Impossible de lire l'audio depuis stdin : {e}")

    duration_s = waveform.shape[1] / SAMPLE_RATE
    windows    = slice_windows(waveform)
    timeline, summary = run_inference(model, windows, binary_mode)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    output = {
        "model_name":      model_name,
        "binary_mode":     binary_mode,
        "duration_s":      round(duration_s, 3),
        "n_windows":       len(windows),
        "window_s":        WINDOW_S,
        "summary":         summary,
        "timeline":        timeline,
        "processed_in_ms": elapsed_ms,
    }

    print(json.dumps(output), flush=True)


def _die(msg: str):
    print(json.dumps({"error": msg}), flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()