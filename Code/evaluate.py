"""
evaluate.py — Évaluation sur le test set.

Métrique du sujet :
  score_i = 1 si le modèle prédit un instrument effectivement présent
  dans l'annotation, 0 sinon.
  Score global = moyenne des scores sur tous les fichiers test.

Fournit aussi :
  - Recall par classe (instrument)
  - Matrice de confusion simplifiée
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from collections import defaultdict
from tqdm import tqdm

import config
from model import InstrumentCNN
from dataset import get_test_loader


@torch.no_grad()
def evaluate_model(model, test_loader=None, verbose=True):
    """
    Évalue le modèle sur le test set avec la métrique IRMAS.

    Args:
        model: modèle entraîné
        test_loader: DataLoader de test (créé si None)
        verbose: afficher les détails

    Returns:
        dict avec score global, détails par classe, etc.
    """
    device = config.DEVICE
    model = model.to(device)
    model.eval()

    if test_loader is None:
        test_loader = get_test_loader()

    # Métriques
    total_files = 0
    correct_files = 0
    class_correct = defaultdict(int)   # Nombre de fois où la classe est correctement prédite
    class_total = defaultdict(int)     # Nombre de fois où la classe apparaît dans les annotations
    predictions_dist = defaultdict(int)  # Distribution des prédictions

    results = []

    for spec, annotations, filename in tqdm(test_loader, desc="Evaluation", disable=not verbose):
        spec = spec.to(device, non_blocking=True)
        annotations = annotations[0]   # Dé-batcher (batch_size=1)
        filename = filename[0]

        with autocast(device_type=device.type, enabled=config.USE_AMP):
            output = model(spec)

        # Prédiction : classe avec la plus haute probabilité
        probas = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probas, dim=1).item()
        predicted_class = config.IDX_TO_CLASS[predicted_idx]
        confidence = probas[0, predicted_idx].item()

        predictions_dist[predicted_class] += 1

        # Score IRMAS : 1 si la prédiction est dans les annotations
        score = 1 if predicted_class in annotations else 0
        correct_files += score
        total_files += 1

        # Recall par classe
        for ann_class in annotations:
            class_total[ann_class] += 1
            if predicted_class == ann_class:
                class_correct[ann_class] += 1

        results.append({
            "filename": filename,
            "predicted": predicted_class,
            "confidence": confidence,
            "annotations": annotations,
            "score": score,
        })

    # Score global
    global_score = correct_files / total_files if total_files > 0 else 0.0

    if verbose:
        print("\n" + "=" * 60)
        print("  Résultats d'évaluation IRMAS")
        print("=" * 60)
        print(f"\n  Score global : {global_score:.4f} ({correct_files}/{total_files})")

        # Recall par classe
        print(f"\n  {'Classe':<10} {'Recall':<12} {'Correct/Total'}")
        print("  " + "-" * 40)
        for cls in config.CLASSES:
            total = class_total.get(cls, 0)
            correct = class_correct.get(cls, 0)
            recall = correct / total if total > 0 else 0.0
            print(f"  {cls:<10} {recall:.4f}       {correct}/{total}")

        # Distribution des prédictions
        print(f"\n  Distribution des prédictions :")
        for cls in config.CLASSES:
            count = predictions_dist.get(cls, 0)
            pct = 100.0 * count / total_files if total_files > 0 else 0.0
            print(f"  {cls:<10} {count:4d} ({pct:.1f}%)")

        # Exemples d'erreurs
        errors = [r for r in results if r["score"] == 0]
        if errors and verbose:
            print(f"\n  Exemples d'erreurs ({len(errors)} total) :")
            for err in errors[:10]:
                print(
                    f"    {err['filename'][:50]:<50} "
                    f"pred={err['predicted']} (conf={err['confidence']:.2f}) "
                    f"actual={err['annotations']}"
                )

        print("=" * 60)

    return {
        "global_score": global_score,
        "correct": correct_files,
        "total": total_files,
        "class_recall": {
            cls: class_correct.get(cls, 0) / class_total.get(cls, 0)
            if class_total.get(cls, 0) > 0 else 0.0
            for cls in config.CLASSES
        },
        "predictions_distribution": dict(predictions_dist),
        "detailed_results": results,
    }


def evaluate_from_checkpoint(checkpoint_path=None):
    """
    Charge un modèle depuis un checkpoint et l'évalue.

    Args:
        checkpoint_path: chemin vers le checkpoint .pt
                         Si None, utilise le meilleur modèle sauvegardé.
    """
    if checkpoint_path is None:
        checkpoint_path = config.MODEL_DIR / "best_model.pt"

    print(f"\n[Eval] Chargement du modèle depuis {checkpoint_path}")

    model = InstrumentCNN(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[Eval] Modèle chargé (epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?'):.4f})")

    return evaluate_model(model)


if __name__ == "__main__":
    evaluate_from_checkpoint()
