# src/data/data_cleaner.py
import pandas as pd
import numpy as np
import re
from loguru import logger
from typing import Tuple
from src.config import NUMERICAL_VARIABLES, CATEGORICAL_VARIABLES, COLUMNS_TO_DROP

# Colonnes à supprimer
COLUMNS_TO_DROP = ['name', 'ticket', 'boat', 'body', 'home.dest']

def get_title(name: str) -> str:
    """
    Extrait le titre honorifique depuis le nom du passager.
    
    Parameters:
        name (str): Nom complet du passager.
        
    Returns:
        str: Titre extrait ('Mr', 'Mrs', 'Miss', 'Master' ou 'Other').
    """
    if re.search("Mrs", name):
        return "Mrs"
    elif re.search("Mr", name):
        return "Mr"
    elif re.search("Miss", name):
        return "Miss"
    elif re.search("Master", name):
        return "Master"
    else:
        return "Other"

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le jeu de données Titanic en réalisant les actions suivantes :
      - Mettre les noms des colonnes en minuscules.
      - Remplacer "?" par np.nan.
      - Extraire le titre honorifique depuis la colonne 'name' et le stocker dans 'title'.
      - Imputer les valeurs manquantes pour les variables numériques avec la médiane.
      - Imputer les valeurs manquantes pour les variables catégoriques avec le string 'missing'.
      - Convertir les variables numériques en float.
      - Convertir les variables catégoriques en type 'category'.
      - Supprimer les colonnes inutiles.
    
    Parameters:
        data (pd.DataFrame): DataFrame brute.
    
    Returns:
        pd.DataFrame: DataFrame nettoyée.
    """
    try:
        logger.info("Début du nettoyage des données")
        # Mettre les noms des colonnes en minuscules
        data.columns = data.columns.str.lower()
        # Remplacer "?" par np.nan
        data = data.replace("?", np.nan)

        # Extraction du titre depuis 'name' (avant suppression)
        if "name" in data.columns:
            data["title"] = data["name"].apply(get_title)
        else:
            logger.warning("La colonne 'name' n'existe pas pour extraire le titre.")

        # Imputation des variables numériques par la médiane
        #numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        for col in NUMERICAL_VARIABLES:
            if data[col].isnull().any():
                median_value = data[col].median()
                logger.info("Imputation de la médiane pour la colonne '{}' : {}", col, median_value)
                data[col].fillna(median_value, inplace=True)

        # Imputation des variables catégoriques par "missing"
        #categorical_cols = ["pclass", "sex", "embarked", "title", "cabin"]
        for col in CATEGORICAL_VARIABLES:
            if col in data.columns:
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    logger.info("Imputation 'missing' pour la colonne '{}' ({} valeurs manquantes)", col, missing_count)
                    data[col].fillna("missing", inplace=True)
                # Conversion en type 'category'
                data[col] = data[col].astype("category")
            else:
                logger.warning("La colonne '{}' n'est pas présente dans le jeu de données.", col)

        # Conversion des variables numériques spécifiques en float
        for col in ["fare", "age"]:
            if col in data.columns:
                data[col] = data[col].astype("float")

        # Suppression des colonnes inutiles
        data.drop(labels=COLUMNS_TO_DROP, axis=1, inplace=True, errors="ignore")


        logger.info("Nettoyage terminé")
        data.to_csv(r"data\processed\cleaned_data.csv")
        
        return data

    except Exception as e:
        logger.error("Erreur dans le nettoyage des données: {}", e)
        raise

def get_overview(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcule et retourne les statistiques suivantes :
      - Proportion de valeurs manquantes pour les variables numériques.
      - Proportion de valeurs manquantes pour les variables catégoriques.
      - Nombre de valeurs différentes pour chaque variable catégorique.
    
    Parameters:
        data (pd.DataFrame): DataFrame nettoyée.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (missing_numeric, missing_categorical, distinct_categorical)
    """
    try:
        numeric_vars = data.select_dtypes(include=["number"]).columns.drop("survived", errors="ignore")
        categorical_vars = data.select_dtypes(include=["category"]).columns
        missing_numeric = data[numeric_vars].isnull().mean().to_frame("missing_percentage")
        missing_categorical = data[categorical_vars].isnull().mean().to_frame("missing_percentage")
        distinct_categorical = pd.DataFrame({col: data[col].nunique() for col in categorical_vars},
                                            index=["nunique"]).T
        return missing_numeric, missing_categorical, distinct_categorical
    except Exception as e:
        logger.error("Erreur dans le calcul des statistiques d'aperçu: {}", e)
        raise
