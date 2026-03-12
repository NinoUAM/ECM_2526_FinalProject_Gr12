"""
API Biopharma ML — Équipe 12
Nino Tissot · Audrey Nourry · Mailis Briens · Hajar Belgroun

Expose deux modèles :
  - StackingClassifier (LR + RF + HGB → méta LR)  →  score financier
  - Score éthique NLP (ESG-BERT + DistilBERT sentiment)  →  score éthique ajusté
  - Score Final = 0.7 × score_stacking + 0.3 × adjusted_ethical_score × data_quality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer



try:
    with open("stack_model.pkl",     "rb") as f: stack            = pickle.load(f)
    with open("scaler_pipeline.pkl", "rb") as f: scaler_pipeline  = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f: feature_columns  = pickle.load(f)
    MODELS_LOADED = True
except FileNotFoundError:
    MODELS_LOADED = False
    print("⚠️  Artefacts ML non trouvés — /predict renverra une erreur jusqu'au chargement.")

try:
    with open("stack_enriched.pkl",            "rb") as f: stack_enriched   = pickle.load(f)
    with open("pipe_enrich.pkl",               "rb") as f: pipe_enrich      = pickle.load(f)
    with open("feature_columns_enriched.pkl",  "rb") as f: feature_cols_enrich = pickle.load(f)
    ENRICHED_LOADED = True
except FileNotFoundError:
    ENRICHED_LOADED = False


app = FastAPI(
    title="Biopharma ML API — Équipe 12",
    description=(
        "Prédit la probabilité de succès (acquisition / IPO) d'une startup biopharma "
        "à partir de ses données financières et de son score éthique ESG-BERT."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class StartupFeatures(BaseModel):
    """Features financières — correspondent exactement à FEATURES dans le notebook."""

    age:                   float = Field(...,  description="Âge de la startup en années à la date de coupure (2012-01-01)")
    funding_total_usd:     float = Field(...,  description="Financement total levé en USD")
    funding_rounds:        int   = Field(...,  description="Nombre de tours de financement")
    days_to_first_funding: float = Field(...,  description="Délai (jours) entre fondation et premier financement")
    funding_duration:      float = Field(...,  description="Durée totale d'activité de levée (jours)")
    funding_velocity:      float = Field(...,  description="funding_total_usd / age — vitesse de levée")
    round_intensity:       float = Field(...,  description="funding_rounds / age — régularité d'attraction des investisseurs")

    has_seed:    int = Field(0, ge=0, le=1, description="A levé un tour seed (0/1)")
    has_venture: int = Field(0, ge=0, le=1, description="A levé un tour venture (0/1)")
    has_grant:   int = Field(0, ge=0, le=1, description="A reçu un grant (0/1)")
    has_round_A: int = Field(0, ge=0, le=1, description="A levé un round A (0/1)")
    has_round_B: int = Field(0, ge=0, le=1, description="A levé un round B (0/1)")

    country_code: int = Field(0, description="Pays encodé (0 = OTHER/UNKNOWN)")

    adjusted_ethical_score: float = Field(0.0, ge=0.0, le=1.0, description="Score éthique ajusté par l'analyse de sentiment DistilBERT")
    sentiment_score:        float = Field(0.5, ge=0.0, le=1.0, description="Score de sentiment (0 = négatif, 1 = positif)")
    data_quality_score:     float = Field(0.0, ge=0.0, le=1.0, description="Fiabilité du score éthique (nb sources / 30, plafonné à 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 6.5,
                "funding_total_usd": 45000000,
                "funding_rounds": 3,
                "days_to_first_funding": 180,
                "funding_duration": 1200,
                "funding_velocity": 6923076,
                "round_intensity": 0.46,
                "has_seed": 1,
                "has_venture": 1,
                "has_grant": 0,
                "has_round_A": 1,
                "has_round_B": 0,
                "country_code": 1,
                "adjusted_ethical_score": 0.73,
                "sentiment_score": 0.82,
                "data_quality_score": 0.6,
            }
        }


FINANCIAL_FEATURES = [
    "age", "funding_total_usd", "funding_rounds",
    "days_to_first_funding", "funding_duration",
    "funding_velocity", "round_intensity",
    "has_seed", "has_venture", "has_grant", "has_round_A", "has_round_B",
    "country_code",
]

def build_input_df(data: StartupFeatures, columns: list) -> np.ndarray:
    """Reconstruit un DataFrame aligné sur les colonnes d'entraînement, puis scale."""
    row = {k: getattr(data, k) for k in FINANCIAL_FEATURES}
    df  = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    return df



@app.get("/", tags=["Statut"])
def home():
    return {
        "status":       "ok",
        "projet":       "Prédiction de Succès en Biopharma & Analyse Éthique",
        "equipe":       "Équipe 12 — Nino Tissot · Audrey Nourry · Mailis Briens · Hajar Belgroun",
        "modeles":      ["StackingClassifier (LR + RF + HGB)", "ESG-BERT + DistilBERT sentiment"],
        "models_ready": MODELS_LOADED,
        "docs":         "/docs",
    }


@app.post("/predict/financier", tags=["Prédiction"])
def predict_financier(data: StartupFeatures):
    """
    Score financier uniquement — StackingClassifier entraîné sur les features
    funding_velocity, round_intensity, funding_total_usd, etc.
    """
    if not MODELS_LOADED:
        raise HTTPException(503, detail="Artefacts ML non chargés. Lancez d'abord le notebook.")

    X = build_input_df(data, feature_columns)
    X_scaled = scaler_pipeline.transform(X)

    proba = float(stack.predict_proba(X_scaled)[0][1])
    label = "Succès probable (acquisition / IPO)" if proba >= 0.5 else "Échec probable (fermeture)"

    return {
        "prediction":          label,
        "probabilite_succes":  round(proba, 4),
        "probabilite_echec":   round(1 - proba, 4),
    }


@app.post("/predict/complet", tags=["Prédiction"])
def predict_complet(data: StartupFeatures):
    """
    Score final combiné = 0.7 × score_stacking + 0.3 × adjusted_ethical_score × data_quality_score.
    Correspond exactement à la formule de la cellule 29 du notebook (Section 8 — Fusion).
    Utilise le StackingClassifier enrichi si disponible (score éthique injecté comme feature ML).
    """
    if not MODELS_LOADED:
        raise HTTPException(503, detail="Artefacts ML non chargés. Lancez d'abord le notebook.")

    X_fin    = build_input_df(data, feature_columns)
    X_scaled = scaler_pipeline.transform(X_fin)
    score_stacking = float(stack.predict_proba(X_scaled)[0][1])

    score_stacking_enriched = None
    if ENRICHED_LOADED:
        X_enr = build_input_df(data, feature_cols_enrich)
        X_enr["adjusted_ethical_score"] = data.adjusted_ethical_score
        X_enr["sentiment_score"]        = data.sentiment_score
        X_enr = X_enr.reindex(columns=feature_cols_enrich, fill_value=0)
        X_enr_scaled = pipe_enrich.transform(X_enr)
        score_stacking_enriched = float(stack_enriched.predict_proba(X_enr_scaled)[0][1])

    score_final = (
        0.7 * score_stacking
        + 0.3 * data.adjusted_ethical_score * data.data_quality_score
    )

    label = "Succès probable (acquisition / IPO)" if score_final >= 0.5 else "Échec probable (fermeture)"

    return {
        "prediction":               label,
        "score_final":              round(score_final, 4),
        "detail": {
            "score_stacking":              round(score_stacking, 4),
            "score_stacking_enriched":     round(score_stacking_enriched, 4) if score_stacking_enriched else "N/A (artefact non chargé)",
            "adjusted_ethical_score":      round(data.adjusted_ethical_score, 4),
            "sentiment_score":             round(data.sentiment_score, 4),
            "data_quality_score":          round(data.data_quality_score, 4),
            "composante_financiere":       round(0.7 * score_stacking, 4),
            "composante_ethique":          round(0.3 * data.adjusted_ethical_score * data.data_quality_score, 4),
        },
    }


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(startups: list[StartupFeatures]):
    """
    Score final pour une liste de startups en une seule requête.
    Retourne le classement trié par score_final décroissant.
    """
    if not MODELS_LOADED:
        raise HTTPException(503, detail="Artefacts ML non chargés. Lancez d'abord le notebook.")
    if len(startups) > 100:
        raise HTTPException(400, detail="Maximum 100 startups par requête batch.")

    rows = []
    for i, data in enumerate(startups):
        X_fin    = build_input_df(data, feature_columns)
        X_scaled = scaler_pipeline.transform(X_fin)
        score_stacking = float(stack.predict_proba(X_scaled)[0][1])
        score_final    = 0.7 * score_stacking + 0.3 * data.adjusted_ethical_score * data.data_quality_score
        rows.append({
            "rank":                    None,  
            "startup_index":           i,
            "score_final":             round(score_final, 4),
            "score_stacking":          round(score_stacking, 4),
            "adjusted_ethical_score":  round(data.adjusted_ethical_score, 4),
            "sentiment_score":         round(data.sentiment_score, 4),
            "data_quality_score":      round(data.data_quality_score, 4),
            "prediction":              "Succès probable" if score_final >= 0.5 else "Échec probable",
        })

    rows.sort(key=lambda x: x["score_final"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    return {"nb_startups": len(rows), "classement": rows}