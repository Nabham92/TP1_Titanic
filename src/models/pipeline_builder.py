# src/models/pipeline_builder.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

# Variables globales pour la pipeline
NUMERICAL_VARIABLES: List[str] = ["age", "fare", "sibsp", "parch"]
CATEGORICAL_VARIABLES: List[str] = ["pclass", "sex", "embarked", "title", "cabin"]

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé qui extrait la première lettre de la variable 'cabin'.
    """
    def __init__(self, variable: str) -> None:
        self.variable = variable

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.variable in X.columns:
            X[self.variable] = X[self.variable].apply(lambda x: x[0] if pd.notnull(x) else x)
        return X

class CategoricalToObjectTransformer(BaseEstimator, TransformerMixin):
    """
    Convertit les colonnes catégoriques (de type category) en type object.
    """
    def __init__(self, variables: List[str]) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                X[var] = X[var].astype(str)
        return X

def build_pipeline() -> Pipeline:
    """
    Construit et retourne la pipeline complète de prétraitement et de modélisation.

    La pipeline effectue les opérations suivantes :
      - Extraction de la première lettre de la variable 'cabin'.
      - Conversion des variables catégoriques en type object.
      - Retrait des catégories rares avec RareLabelEncoder (en ignorant les valeurs manquantes).
      - One-hot encoding en k-1 variables.
      - Normalisation des variables.
      - Régression logistique (C=0.0005).
    
    Returns:
        Pipeline: Pipeline scikit-learn prête à l'entraînement.
    """
    try:
        logger.info("Construction de la pipeline")
        pipeline = Pipeline([
            ("extract_letter", ExtractLetterTransformer(variable="cabin")),
            ("cat_to_object", CategoricalToObjectTransformer(variables=CATEGORICAL_VARIABLES)),
            ("rare_label_encoder", RareLabelEncoder(
                tol=0.05,
                n_categories=1,
                variables=CATEGORICAL_VARIABLES,
                missing_values="ignore"
            )),
            ("categorical_encoder", OneHotEncoder(drop_last=True, variables=CATEGORICAL_VARIABLES)),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=0.0005, random_state=0))
        ])
        return pipeline
    except Exception as e:
        logger.error("Erreur lors de la construction de la pipeline: {}", e)
        raise
