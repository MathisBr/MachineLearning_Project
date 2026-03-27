PROJET MACHINE LEARNING - RENDU

1) Participants
- Alystan GOUGEON--BRIHAULT
- Mathis BREVET
- Alexandra CULLMANN
- Lucas HARDY

2) Description du travail réalisé
Ce projet implémente une pipeline de classification d'instruments de musique sur IRMAS (classes: gac, org, pia, voi).

Contenu principal:
- Preprocessing audio vers spectrogrammes Mel log (dossier Code/preprocess.py)
- Entraînement d'un CNN PyTorch avec validations (Code/train.py, Code/model.py, Code/dataset.py)
- Évaluation du modèle sur le jeu de test (Code/evaluate.py)
- Point d'entrée pipeline (Code/main.py)
- API web de prédiction et script d'inférence (Web/api.py, Web/eval.py)

Organisation des données:
- Dataset/ : données source (train/test)
- cache/   : spectrogrammes pré-calculés (.pt)
- Models/  : checkpoints (ex: best_model.pt)

Exécution (exemple):
1. Installer les dépendances:
   pip install -r requirements.txt
2. Lancer la page web en une commande (ouvre le navigateur automatiquement):
   py launch_web.py
3. Lancer la pipeline complète:
   py Code/main.py
4. Variante évaluation seule:
   py Code/main.py --eval-only

3) Utilisation de l'IA générative
- Utilisée: Oui
- But:
  - Recherche documentaire pour se renseigner sur l'état de l'art (classification audio, CNN sur spectrogrammes Mel, bonnes pratiques d'évaluation).
  - Aide méthodologique pour identifier la meilleure façon de répondre au problème (choix pipeline preprocess/train/eval, métriques, validation).
  - Production assistée d'un script de recherche d'hyperparamètres pour proposer des valeurs optimales dans Code/config.py.
  - la rédaction du code de esiea_instrument_detector.html a été entièrement produite par l'IA à partir des spécifications fournies.

4) Le script
Objectif:
Trouver automatiquement de meilleures valeurs pour les hyperparamètres définis dans Code/config.py (ex: LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MIXUP_ALPHA, LABEL_SMOOTHING, COSINE_T0, COSINE_T_MULT).

Principe général:
1. Un espace de recherche a été défini pour les hyperparamètres principaux.
2. Des essais automatiques ont été exécutés (plusieurs configurations testées).
3. Pour chaque essai, le modèle a été entraîné puis évalué sur validation.
4. Les résultats ont été comparés à partir des métriques obtenues.
5. Les meilleurs réglages identifiés ont servi à fixer les valeurs retenues dans Code/config.py.

Trace de l'utilisation IA sur cette partie:
L'IA a été utilisée pour aider à structurer ce script de recherche (logique d'exploration, suivi des essais, et comparaison des métriques) afin d'identifier plus rapidement des paramètres performants.
