"""
main.py — Point d'entrée principal de la pipeline complète.

Exécute dans l'ordre :
    1. Pré-traitement (preprocess.py)
    2. Entraînement  (train.py)
    3. Évaluation    (evaluate.py)

Usage:
    python main.py
    python main.py --skip-preprocess   # Si les caches existent déjà
    python main.py --eval-only         # Évaluation uniquement
"""

import argparse
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Reconnaissance d'instruments")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Ignore l'étape de pré-traitement (si cache existe déjà)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Lance uniquement l'évaluation sur le test set")
    args = parser.parse_args()

    train_meta_path = Path(config.CACHE_DIR) / "train_metadata.json"
    test_meta_path  = Path(config.CACHE_DIR) / "test_metadata.json"

    if not args.eval_only:
        # ── Étape 1 : Pré-traitement ─────────────────────────────────────────
        if not args.skip_preprocess or not train_meta_path.exists():
            from preprocess import main as run_preprocess
            print("\n[1/3] Pré-traitement des données...")
            run_preprocess()
        else:
            print("\n[1/3] Pré-traitement ignoré (cache existant détecté).")

        # ── Étape 2 : Entraînement ───────────────────────────────────────────
        from train import train
        print("\n[2/3] Entraînement du modèle...")
        train(metadata_path=str(train_meta_path))

    # ── Étape 3 : Évaluation ─────────────────────────────────────────────────
    from evaluate import evaluate_test_set
    print("\n[3/3] Évaluation sur l'ensemble de test...")
    score = evaluate_test_set(metadata_path=str(test_meta_path))

    print(f"\n{'═' * 45}")
    print(f"  Score final : {score:.2%}")
    print(f"{'═' * 45}\n")


if __name__ == "__main__":
    main()
