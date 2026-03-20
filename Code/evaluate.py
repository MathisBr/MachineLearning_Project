"""
evaluate.py — Évaluation du modèle sur l'ensemble de test.

Métrique spécifique du sujet (section 2) :
    Pour chaque instance de test :
        score = 1 si le modèle prédit un instrument présent dans l'annotation,
        score = 0 sinon.
    Score final = moyenne sur toutes les instances.

Cette métrique est plus souple qu'une accuracy classique : le modèle n'a pas
besoin de trouver l'instrument DOMINANT, mais UN instrument parmi ceux présents.

Usage:
    python evaluate.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import config
from dataset import build_test_loader
from model import build_model, InstrumentCNN


@torch.no_grad()
def predict_with_sliding_window(
    log_mel: torch.Tensor,
    model: nn.Module,
    window_frames: int = 130,
    stride_frames: int = 65,
) -> int:
    """
    Decoupe un spectrogramme en fenetres fixes et moyenne les logits.

    Args:
        log_mel:        Spectrogramme (1, N_MELS, T).
        model:          Modele de classification.
        window_frames:  Taille de fenetre en frames.
        stride_frames:  Pas de deplacement entre fenetres.

    Returns:
        Index de classe majoritaire (via moyenne de logits).
    """
    total_frames = log_mel.shape[2]

    if total_frames <= window_frames:
        padded = F.pad(log_mel, (0, window_frames - total_frames))
        logits = model(padded.unsqueeze(0).to(config.DEVICE))
        return logits[0].argmax().item()
    else:
        starts = range(0, total_frames - window_frames + 1, stride_frames)
        all_logits = []
        for start in starts:
            window = log_mel[:, :, start : start + window_frames]
            logits = model(window.unsqueeze(0).to(config.DEVICE))[0]
            all_logits.append(logits)
        
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits.argmax().item()


def load_best_model() -> InstrumentCNN:
    """Charge le meilleur modèle depuis le checkpoint sauvegardé."""
    model = build_model()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print(f"✓ Modèle chargé depuis : {config.MODEL_PATH}")
    return model


@torch.no_grad()
def evaluate_test_set(model_path: str = None, metadata_path: str = None) -> float:
    """
    Évalue le modèle sur l'ensemble de test avec la métrique du sujet.

    Args:
        model_path:    Chemin vers le checkpoint du modèle.
        metadata_path: Chemin vers test_metadata.json.

    Returns:
        Score moyen (float entre 0 et 1).
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    if metadata_path is None:
        metadata_path = str(Path(config.CACHE_DIR) / "test_metadata.json")

    print("=" * 55)
    print("    ÉVALUATION — Ensemble de test")
    print("=" * 55)

    model       = build_model()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    test_loader = build_test_loader(metadata_path)

    scores           = []
    per_class_hits   = {cls: 0 for cls in config.INSTRUMENT_CLASSES}
    per_class_total  = {cls: 0 for cls in config.INSTRUMENT_CLASSES}
    predictions_log  = []

    for spectrogram, instruments_present in test_loader:
        if spectrogram.ndim == 4:
            spectrogram = spectrogram[0]

        pred_idx    = predict_with_sliding_window(spectrogram, model)
        pred_label  = config.IDX_TO_CLASS[pred_idx]
        pred_prob   = 0.0

        # Fix PyTorch collate_fn returning tuples instead of strings for batch_size=1
        if len(instruments_present) > 0 and isinstance(instruments_present[0], tuple):
            instruments_present = [inst[0] for inst in instruments_present]

        # Filtrage : on ne garde que les instruments qui nous intéressent
        relevant_instruments = [
            inst for inst in instruments_present
            if inst in config.INSTRUMENT_CLASSES
        ]

        # Calcul du score selon la métrique du sujet
        hit   = 1 if pred_label in relevant_instruments else 0
        scores.append(hit)

        # Stats par classe (pour le rapport détaillé)
        for inst in relevant_instruments:
            per_class_total[inst] += 1
            if inst == pred_label:
                per_class_hits[inst] += 1

        predictions_log.append({
            "predicted":  pred_label,
            "confidence": pred_prob,
            "ground_truth": list(relevant_instruments),
            "hit": hit,
        })

    # ── Résultats ─────────────────────────────────────────────────────────────
    final_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'─' * 45}")
    print(f"  Score global : {final_score:.2%}  ({sum(scores)}/{len(scores)})")
    print(f"  Objectif     : ≥ 70%  → {'✅ ATTEINT' if final_score >= 0.70 else '❌ NON ATTEINT'}")
    print(f"{'─' * 45}")

    print("\n  Détail par instrument :")
    print(f"  {'Instrument':>12} | {'Vrais positifs':>14} | {'Total':>7} | {'Recall':>7}")
    print(f"  {'─'*12}─|─{'─'*14}─|─{'─'*7}─|─{'─'*7}")
    for cls in config.INSTRUMENT_CLASSES:
        total  = per_class_total[cls]
        hits   = per_class_hits[cls]
        recall = (hits / total) if total > 0 else 0.0
        print(f"  {cls:>12} | {hits:>14} | {total:>7} | {recall:>7.2%}")

    return final_score


def predict_single_file(wav_path: str, model_path: str = None) -> dict:
    """
    Prédit l'instrument dominant dans un fichier audio unique.
    Utile pour le bonus (application).

    Args:
        wav_path:   Chemin vers un fichier .wav.
        model_path: Chemin vers le checkpoint du modèle.

    Returns:
        Dictionnaire avec 'instrument', 'confidence' et 'all_probs'.
    """
    import torchaudio
    import torchaudio.transforms as T
    from preprocess import load_and_resample, waveform_to_log_mel, build_mel_transform

    if model_path is None:
        model_path = config.MODEL_PATH

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    mel_transform = build_mel_transform()
    waveform      = load_and_resample(wav_path)
    log_mel       = waveform_to_log_mel(waveform, mel_transform)
    log_mel       = log_mel.unsqueeze(0).to(config.DEVICE)  # (1, 1, N_MELS, T)

    with torch.no_grad():
        logits = model(log_mel)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_label  = config.IDX_TO_CLASS[pred_idx]
    confidence  = probs[pred_idx].item()
    all_probs   = {config.IDX_TO_CLASS[i]: probs[i].item() for i in range(config.NUM_CLASSES)}

    return {
        "instrument": pred_label,
        "confidence": confidence,
        "all_probs":  all_probs,
    }


if __name__ == "__main__":
    evaluate_test_set()
