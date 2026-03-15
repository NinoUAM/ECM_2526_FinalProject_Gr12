# 1. EDA & Data Cleaning

Exploratory Data Analysis : script d'exploration des données

- Affiche df.head(), df.describe() et d'autres statistiques descriptives
- Visualisations initiales pour comprendre la structure des données

`ANALYSE.ipynb` : notebook d'exploration détaillée, graphiques et premières observations

`CLEANING.ipynb` : notebook de nettoyage et prétraitement
- Renommage des colonnes
- Filtrage sur pharma/biotech
- Conversion et nettoyage de funding_total_usd
- Gestion des valeurs manquantes

Ces fichiers permettent de préparer un dataset propre et exploitable pour la modélisation.

# 2. Machine Learning

Scripts et notebooks pour la modélisation ML :

- Feature engineering
- Entraînement des modèles (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- StackingClassifier
- Evaluation des performances (F1, Recall, matrices de confusion)

Ces scripts permettent de reproduire les prédictions financières et de générer le score ML des startups.

# 3. Analyse Éthique — Pipeline NLP

- Pipeline de collecte automatique de données textuelles depuis ClinicalTrials.gov, PubMed et les sites web des startups
- Analyse par ESG-BERT (26 catégories ESG) couplée à une analyse de sentiment via DistilBERT
- Génération d'un score éthique ajusté et d'un data_quality_score selon le nombre de sources collectées
- Fusion avec le score financier pour produire le classement final

# 4. API / Industrialisation

- Déploiement des modèles dans une API REST via FastAPI
- Endpoints disponibles : score financier seul, score complet financier + éthique, batch scoring jusqu'à 100 startups
- Dashboard interactif pour tester les startups directement depuis un navigateur
