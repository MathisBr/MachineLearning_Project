"""
main.py — Point d'entrée principal du pipeline IRMAS.

Enchaîne les 3 étapes :
  1. Preprocessing (audio → spectrogrammes cachés)
  2. Training (CNN avec optimisations modernes)
  3. Evaluation (métrique IRMAS sur le test set)

Usage :
  py main.py                    # Pipeline complète
  py main.py --skip-preprocess  # Skip si les caches existent
  py main.py --eval-only        # Évaluation uniquement
"""

import argparse
import sys
import os

# Ajouter le dossier Code au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from preprocess import preprocess_all
from train import train_model
from evaluate import evaluate_model, evaluate_from_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline IRMAS — Reconnaissance d'instruments de musique"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Sauter le preprocessing si les caches sont déjà calculés",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Évaluation uniquement (le modèle doit être déjà entraîné)",
    )
    args = parser.parse_args()

    # Afficher la configuration
    config.print_config()

    if args.eval_only:
        # Évaluation seule
        print("\n" + "=" * 60)
        print("  MODE : Évaluation uniquement")
        print("=" * 60)

        best_model_path = config.MODEL_DIR / "best_model.pt"
        if not best_model_path.exists():
            print(f"[Erreur] Modèle non trouvé : {best_model_path}")
            print("         Lancez d'abord un entraînement complet.")
            sys.exit(1)

        results = evaluate_from_checkpoint(best_model_path)
        return results

    # Etape 1 - Preprocessing
    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("  ÉTAPE 1/3 : Preprocessing")
        print("=" * 60)
        preprocess_all()
    else:
        print("\n[Info] Preprocessing ignoré (--skip-preprocess)")
        # Vérifier que le cache existe
        if not config.CACHE_DIR.exists():
            print("[Erreur] Dossier cache non trouvé. Relancez sans --skip-preprocess.")
            sys.exit(1)

    # Etape 2 - Training
    print("\n" + "=" * 60)
    print("  ÉTAPE 2/3 : Entraînement")
    print("=" * 60)
    model = train_model()

    # Etape 3 - Evaluation
    print("\n" + "=" * 60)
    print("  ÉTAPE 3/3 : Évaluation")
    print("=" * 60)
    results = evaluate_model(model)

    print(f"\n{'='*60}")
    print(f"  SCORE FINAL : {results['global_score']:.4f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
