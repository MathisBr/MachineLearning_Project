"""
preprocess.py — Transformation des fichiers audio en Log-Mel spectrogrammes.

Stratégie choisie : pré-calculer les spectrogrammes AVANT l'entraînement
et les sauvegarder sur disque (.pt). Cela évite de répéter la transformation
à chaque batch (recommandation du sujet, section 3.4).

Représentation retenue : Log-Mel Spectrogram 2D
  → Supérieur aux MFCC pour les CNN selon Chen et al. (2024) et Castel-Branco
    et al. (2020) sur IRMAS.
  → Plus riche en information que les MFCC (pas de compression DCT).
  → Exploitable directement par des Conv2d comme une image.

Usage:
    python preprocess.py
"""

import json
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

import config


def build_mel_transform() -> T.MelSpectrogram:
    """Construit la transformation Mel Spectrogram avec les paramètres de config."""
    return T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.F_MIN,
        f_max=config.F_MAX,
    )


def load_and_resample(file_path: str) -> torch.Tensor:
    """
    Charge un fichier audio, le convertit en mono et le rééchantillonne (via librosa).

    Returns:
        Tenseur de forme (1, N_samples).
    """
    import librosa
    
    waveform_np, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(waveform_np)
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        
    return waveform


def load_waveform_from_npy(file_path: str) -> torch.Tensor:
    """
    Charge un signal audio pré-converti en .npy/.npz et le remet au format attendu.

    Returns:
        Tenseur de forme (1, N_samples), dtype float32, plage approx. [-1, 1].
    """
    if file_path.endswith(".npz"):
        npz = np.load(file_path)
        waveform_np = npz["audio"]
    else:
        waveform_np = np.load(file_path)

    waveform_np = np.asarray(waveform_np)

    if waveform_np.ndim == 2:
        # Sécurise les rares cas multi-canaux en moyennant les canaux.
        if waveform_np.shape[0] <= 4:
            waveform_np = waveform_np.mean(axis=0)
        else:
            waveform_np = waveform_np.mean(axis=1)

    if np.issubdtype(waveform_np.dtype, np.integer):
        waveform_np = waveform_np.astype(np.float32) / 32767.0
    else:
        waveform_np = waveform_np.astype(np.float32, copy=False)

    waveform = torch.from_numpy(waveform_np)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return waveform


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Normalise la longueur du signal audio.
    - Trop court → padding de zéros à droite.
    - Trop long   → troncature.

    Note : valable UNIQUEMENT pour l'ensemble d'entraînement (longueur fixe = 3s).
    Pour le test, on utilise AdaptiveAvgPool2d dans le modèle.
    """
    current_length = waveform.shape[1]

    if current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_length > target_length:
        waveform = waveform[:, :target_length]

    return waveform


def waveform_to_log_mel(waveform: torch.Tensor, mel_transform: T.MelSpectrogram) -> torch.Tensor:
    """
    Convertit un signal audio en Log-Mel Spectrogram normalisé.

    La conversion logarithmique compresse la dynamique du signal et améliore
    la lisibilité des patterns fréquentiels pour le CNN.

    Returns:
        Tenseur de forme (1, N_MELS, T) — compatible avec Conv2d.
    """
    mel_spec = mel_transform(waveform)              # (1, N_MELS, T)
    log_mel_spec = torch.log1p(mel_spec)            # log(1 + x) — évite log(0)
    return log_mel_spec


def preprocess_train_set(mel_transform: T.MelSpectrogram) -> list[dict]:
    """
    Traite tous les fichiers d'entraînement et les sauvegarde dans CACHE_DIR.

    Returns:
        Liste de métadonnées : [{path_cached, label_idx}, ...]
    """
    train_dir = Path(config.TRAIN_DIR)
    cache_dir = Path(config.CACHE_DIR) / "train"
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    npy_root = Path(config.NPY_DIR)

    for instrument_name, label_idx in config.CLASS_TO_IDX.items():
        instrument_dir = train_dir / instrument_name
        npy_instrument_dir = npy_root / instrument_name

        if not instrument_dir.exists():
            print(f"[WARN] Dossier introuvable : {instrument_dir}")
            continue

        npy_files = sorted(npy_instrument_dir.glob("*.npy")) if npy_instrument_dir.exists() else []
        source_files = npy_files if npy_files else sorted(instrument_dir.glob("*.wav"))
        source_kind = "npy" if npy_files else "wav"

        print(f"  [{instrument_name}] {len(source_files)} fichiers ({source_kind})...")

        for src_path in source_files:
            cache_filename = f"{instrument_name}_{src_path.stem}.pt"
            cache_path = cache_dir / cache_filename

            if not cache_path.exists():
                if src_path.suffix.lower() in {".npy", ".npz"}:
                    waveform = load_waveform_from_npy(str(src_path))
                else:
                    waveform = load_and_resample(str(src_path))
                waveform = pad_or_truncate(waveform, config.TRAIN_SAMPLES)
                log_mel  = waveform_to_log_mel(waveform, mel_transform)
                torch.save(log_mel, cache_path)

            metadata.append({
                "cache_path": str(cache_path),
                "label_idx":  label_idx,
            })

    return metadata


def preprocess_test_set(mel_transform: T.MelSpectrogram) -> list[dict]:
    """
    Traite tous les fichiers de test.

    Note : les fichiers test ont des longueurs variables → PAS de padding/troncature.
    Le modèle utilise AdaptiveAvgPool2d pour absorber cette variabilité.

    Returns:
        Liste de métadonnées : [{path_cached, annotation_file}, ...]
    """
    test_dir  = Path(config.TEST_DIR)
    npy_test_dir = Path(config.NPY_DIR) / "Test"
    cache_dir = Path(config.CACHE_DIR) / "test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    npy_files = sorted(npy_test_dir.glob("*.npy")) if npy_test_dir.exists() else []
    source_files = npy_files if npy_files else sorted(test_dir.glob("*.wav"))

    for src_path in source_files:
        cache_path = cache_dir / f"{src_path.stem}.pt"

        if not cache_path.exists():
            if src_path.suffix.lower() in {".npy", ".npz"}:
                waveform = load_waveform_from_npy(str(src_path))
            else:
                waveform = load_and_resample(str(src_path))
            log_mel  = waveform_to_log_mel(waveform, mel_transform)
            torch.save(log_mel, cache_path)

        # Cherche le fichier d'annotation associé
        annotation_path = test_dir / f"{src_path.stem}.txt"

        metadata.append({
            "cache_path":      str(cache_path),
            "annotation_path": str(annotation_path) if annotation_path.exists() else None,
        })

    return metadata


def main():
    print("=== Pré-traitement des données ===")
    print(f"Device        : {config.DEVICE}")
    print(f"Sample rate   : {config.SAMPLE_RATE} Hz")
    print(f"N_MELS        : {config.N_MELS}")
    print(f"N_FFT         : {config.N_FFT}")
    print(f"Hop length    : {config.HOP_LENGTH}")
    print()

    mel_transform = build_mel_transform()

    # ── Ensemble d'entraînement ──────────────────────────────────────────────
    print(">>> Traitement de l'ensemble d'entraînement...")
    train_metadata = preprocess_train_set(mel_transform)
    train_meta_path = Path(config.CACHE_DIR) / "train_metadata.json"
    with open(train_meta_path, "w") as f:
        json.dump(train_metadata, f, indent=2)
    print(f"    → {len(train_metadata)} fichiers traités. Métadonnées : {train_meta_path}")

    # ── Ensemble de test ─────────────────────────────────────────────────────
    print(">>> Traitement de l'ensemble de test...")
    test_metadata = preprocess_test_set(mel_transform)
    test_meta_path = Path(config.CACHE_DIR) / "test_metadata.json"
    with open(test_meta_path, "w") as f:
        json.dump(test_metadata, f, indent=2)
    print(f"    → {len(test_metadata)} fichiers traités. Métadonnées : {test_meta_path}")

    print("\n✓ Pré-traitement terminé.")


if __name__ == "__main__":
    main()
