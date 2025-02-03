# src/data/data_loader.py
import pandas as pd
from loguru import logger
from typing import Any

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge le jeu de données depuis un fichier CSV.

    Parameters:
        file_path (str): Chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    try:
        logger.info("Chargement des données depuis {}", file_path)
        data: pd.DataFrame = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error("Erreur lors du chargement des données: {}", e)
        raise
