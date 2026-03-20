================================================================================
  INF3043 — Projet ML : Reconnaissance d'instruments de musique
  Cours : Machine Learning — Enseignant : Matthieu Le Berre
================================================================================

Participants
------------
  [NOM Prénom 1]
  [NOM Prénom 2]
  [NOM Prénom 3]
  [NOM Prénom 4]


Description du projet
---------------------

Pipeline de reconnaissance d'instruments de musique sur le dataset IRMAS
simplifié (4 classes : gac, org, pia, voi).

─── Structure des fichiers ───────────────────────────────────────────────────

  config.py       Hyperparamètres centralisés (chemins, taux d'apprentissage,
                  paramètres du spectrogramme, etc.)

  preprocess.py   Transformation audio → Log-Mel Spectrogram + cache sur disque.
                  Les spectrogrammes sont pré-calculés avant l'entraînement
                  pour ne pas pénaliser la vitesse d'entraînement.

  dataset.py      Datasets PyTorch (train/val/test) + DataLoaders.
                  Inclut l'augmentation SpecAugment (FrequencyMasking + TimeMasking)
                  appliquée uniquement sur les données d'entraînement.

  model.py        Architecture CNN :
                    - 3 blocs Conv2d + BatchNorm2d + ReLU + MaxPool2d
                    - AdaptiveAvgPool2d(4,4) pour absorber les longueurs variables
                    - Dropout (0.4) dans la partie dense
                    - ~600 000 paramètres (modèle compact pour éviter le surapprentissage)

  train.py        Boucle d'entraînement avec :
                    - Optimiseur Adam (lr=1e-3, weight_decay=1e-4)
                    - ReduceLROnPlateau (patience=5, factor=0.5)
                    - Early stopping (patience=10)
                    - Sauvegarde du meilleur modèle (val_loss minimale)

  evaluate.py     Évaluation sur le test set selon la métrique du sujet :
                    score_i = 1 si le modèle prédit un instrument présent, 0 sinon.
                    Fournit aussi un détail par classe (recall par instrument).

  main.py         Point d'entrée principal : enchaîne les 3 étapes.


─── Choix techniques ────────────────────────────────────────────────────────

  Représentation sonore :
    Log-Mel Spectrogram 2D (64 bandes Mel, n_fft=1024, hop=512, sr=22050 Hz).
    Choix justifié par la littérature : Chen et al. (2024) et Castel-Branco
    et al. (2020) montrent que le Log-Mel surpasse les MFCC sur IRMAS pour
    les CNN 2D.

  Gestion de la taille variable (test set) :
    Solution 2 du sujet — AdaptiveAvgPool2d(4,4) dans le modèle. Cela permet
    de traiter les fichiers test de longueurs quelconques sans découpage.
    Le batch_size est fixé à 1 pour le test loader conformément au sujet.

  Lutte contre le sur-apprentissage :
    - Modèle compact (~600k paramètres)
    - BatchNorm2d entre Conv2d et ReLU
    - Dropout(0.4) dans la partie dense
    - SpecAugment (FrequencyMasking + TimeMasking) pendant l'entraînement
    - Early stopping avec patience=10
    - ReduceLROnPlateau


─── Utilisation ─────────────────────────────────────────────────────────────

  Prérequis :
    pip install torch torchaudio

  Structure des données attendue :
    data/
    ├── train/
    │   ├── gac/   (fichiers .wav 3 secondes)
    │   ├── org/
    │   ├── pia/
    │   └── voi/
    └── test/
        ├── Song_name.wav
        └── Song_name.txt  (annotations : liste d'instruments)

  Exécution :
    python main.py                  # Pipeline complète
    python main.py --skip-preprocess  # Si les caches sont déjà calculés
    python main.py --eval-only        # Évaluation uniquement


─── Utilisation de l'IA générative ─────────────────────────────────────────

  L'IA générative (Claude, Anthropic) a été utilisée pour :
    - La recherche bibliographique sur les meilleures représentations sonores
      pour la classification d'instruments sur IRMAS.
    - La structuration du code selon les principes SOLID et la séparation
      des responsabilités (config / dataset / model / train / evaluate).
    - La rédaction des docstrings et commentaires explicatifs.

  Toute la logique métier, les choix d'architecture et les hyperparamètres
  ont été validés et ajustés manuellement par l'équipe.

================================================================================
