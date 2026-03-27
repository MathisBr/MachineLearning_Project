"""
preprocess.py — Transformation audio → Log-Mel Spectrogram + cache sur disque.

Les spectrogrammes sont pré-calculés avant l'entraînement pour ne pas
pénaliser la vitesse d'entraînement. Utilise soundfile + torchaudio transforms.
"""

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

import config


def load_audio(wav_path):
    """
    Charge un fichier audio en mono au sample rate cible.
    Utilise soundfile pour éviter la dépendance à torchcodec.
    
    Returns:
        Tensor (1, num_samples)
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    
    # Convertir en tensor
    waveform = torch.from_numpy(data)
    
    # Si stéréo, prendre la moyenne
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    
    # Ajouter la dimension channel
    waveform = waveform.unsqueeze(0)  # (1, num_samples)
    
    # Resample si nécessaire
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        waveform = resampler(waveform)
    
    return waveform


def compute_log_mel_spectrogram(waveform, sr=config.SAMPLE_RATE):
    """
    Convertit un waveform en Log-Mel Spectrogram.

    Args:
        waveform: Tensor audio (1, num_samples)
        sr: Sample rate

    Returns:
        Tensor (1, n_mels, time_frames) — Log-Mel normalisé
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.F_MIN,
        f_max=config.F_MAX,
        power=2.0,
    )

    mel_spec = mel_transform(waveform)

    # Log-scale (ajouter epsilon pour éviter log(0))
    log_mel = torch.log(mel_spec + 1e-9)

    # Normalisation par fichier (mean=0, std=1)
    mean = log_mel.mean()
    std = log_mel.std()
    if std > 0:
        log_mel = (log_mel - mean) / std

    return log_mel


def process_single_file(args):
    """Traite un seul fichier audio et sauvegarde le spectrogramme."""
    wav_path, cache_path = args

    if cache_path.exists():
        return True

    try:
        waveform = load_audio(wav_path)

        # Calcul du spectrogramme
        log_mel = compute_log_mel_spectrogram(waveform)

        # Sauvegarde
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(log_mel, str(cache_path))
        return True

    except Exception as e:
        print(f"[Erreur] {wav_path.name}: {e}")
        return False


def preprocess_train():
    """
    Pré-calcule les spectrogrammes pour toutes les données d'entraînement.
    Sauvegarde dans cache/train/{class}/{filename}.pt
    """
    print("\n[Preprocess] Calcul des spectrogrammes d'entraînement...")

    tasks = []
    for class_name, class_dir in config.TRAIN_DIRS.items():
        cache_class_dir = config.CACHE_DIR / "train" / class_name
        cache_class_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(class_dir.glob("*.wav"))
        for wav_path in wav_files:
            cache_path = cache_class_dir / (wav_path.stem + ".pt")
            tasks.append((wav_path, cache_path))

    # Traitement (séquentiel pour la stabilité sur Windows)
    success = 0
    for task in tqdm(tasks, desc="Preprocessing train"):
        if process_single_file(task):
            success += 1

    print(f"[Preprocess] {success}/{len(tasks)} fichiers train traités.")
    return success


def preprocess_test():
    """
    Pré-calcule les spectrogrammes pour les données de test.
    Sauvegarde dans cache/test/{filename}.pt
    
    Note: Les fichiers test ont des longueurs variables, donc les
    spectrogrammes auront des dimensions temporelles différentes.
    """
    print("\n[Preprocess] Calcul des spectrogrammes de test...")

    cache_test_dir = config.CACHE_DIR / "test"
    cache_test_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(config.TEST_DIR.glob("*.wav"))
    tasks = []
    for wav_path in wav_files:
        cache_path = cache_test_dir / (wav_path.stem + ".pt")
        tasks.append((wav_path, cache_path))

    success = 0
    for task in tqdm(tasks, desc="Preprocessing test"):
        if process_single_file(task):
            success += 1

    print(f"[Preprocess] {success}/{len(tasks)} fichiers test traités.")
    return success


def preprocess_all():
    """Lance le preprocessing complet (train + test)."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    preprocess_train()
    preprocess_test()
    print("\n[Preprocess] Terminé !")


if __name__ == "__main__":
    preprocess_all()
