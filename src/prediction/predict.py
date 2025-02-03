# src/prediction/predict.py
import joblib
import pandas as pd
from loguru import logger
from typing import Any
from src.data.data_cleaner import clean_data

def load_model(model_path: str) -> Any:
    """
    Charge le modèle sauvegardé.

    Parameters:
        model_path (str): Chemin vers le fichier du modèle.

    Returns:
        Any: Modèle chargé.
    """
    try:
        logger.info("Chargement du modèle depuis {}", model_path)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle: {}", e)
        raise

def predict(data: pd.DataFrame, model: Any) -> pd.Series:
    """
    Effectue des prédictions sur un DataFrame à l'aide du modèle chargé.

    Parameters:
        data (pd.DataFrame): Données sur lesquelles effectuer la prédiction.
        model (Any): Modèle entraîné.

    Returns:
        pd.Series: Prédictions (classes).
    """
    try:
        logger.info("Début de la prédiction")
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        logger.error("Erreur lors de la prédiction: {}", e)
        raise

def main() -> None:
    MODEL_PATH: str = "models/titanic_pipeline.joblib"
    DATA_PATH: str = "data/raw/train.csv"  # Exemple de données ; à adapter pour un nouveau jeu de données
    model = load_model(MODEL_PATH)
    df: pd.DataFrame = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    X = df.drop("survived", axis=1)
    predictions = predict(X, model)
    logger.info("Quelques prédictions: {}", predictions[:5].tolist())

if __name__ == "__main__":
    main()

