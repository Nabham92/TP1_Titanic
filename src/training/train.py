# src/training/train.py
import os
import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from typing import Any
import matplotlib.pyplot as plt

from src.data.data_loader import load_data
from src.data.data_cleaner import clean_data, get_overview
from src.models.pipeline_builder import build_pipeline
from src.models.evaluator import evaluate_model

def plot_numeric_distributions(data: pd.DataFrame, numeric_vars: list) -> None:
    """
    Trace les distributions des variables numériques.

    Parameters:
        data (pd.DataFrame): DataFrame contenant les données.
        numeric_vars (list): Liste des variables numériques.
    """
    try:
        num_vars = len(numeric_vars)
        fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(8, 4 * num_vars))
        if num_vars == 1:
            axes = [axes]
        for ax, var in zip(axes, numeric_vars):
            data[var].hist(ax=ax, bins=30)
            ax.set_title(f"Distribution de {var}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error("Erreur lors du tracé des distributions: {}", e)
        raise

def train_model(data_path: str, model_output_path: str) -> None:
    """
    Entraîne le modèle de régression logistique sur le jeu de données Titanic et sauvegarde la pipeline.

    Parameters:
        data_path (str): Chemin vers le fichier CSV contenant les données brutes.
        model_output_path (str): Chemin où sauvegarder le modèle entraîné.
    """
    try:
        logger.info("Début du processus d'entraînement")
        data: pd.DataFrame = load_data(data_path)
        data = clean_data(data)

        # Affichage des statistiques d'aperçu
        missing_numeric, missing_categorical, distinct_categorical = get_overview(data)
        logger.info("Proportion de valeurs manquantes pour les variables numériques:\n{}", missing_numeric)
        logger.info("Proportion de valeurs manquantes pour les variables catégoriques:\n{}", missing_categorical)
        logger.info("Nombre de valeurs différentes pour les variables catégoriques:\n{}", distinct_categorical)

        # Tracé des distributions des variables numériques (hors target 'survived')
        numeric_cols = data.select_dtypes(include=["number"]).columns.drop("survived", errors="ignore").tolist()
        plot_numeric_distributions(data, numeric_cols)

        # Définir la target et séparer train/test
        X = data.drop("survived", axis=1)
        y = data["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Construction et entraînement de la pipeline
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        logger.info("Entraînement terminé")

        # Évaluation du modèle
        evaluate_model(pipeline, X_train, X_test, y_train, y_test)

        # Sauvegarde de la pipeline
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(pipeline, model_output_path)
        logger.info("Modèle sauvegardé à {}", model_output_path)
    except Exception as e:
        logger.error("Erreur durant l'entraînement: {}", e)
        raise

def main() -> None:
    DATA_PATH: str = "data/raw/train.csv"
    MODEL_OUTPUT_PATH: str = "models/titanic_pipeline.joblib"
    train_model(DATA_PATH, MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()
