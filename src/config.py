# src/config.py
from typing import List

# Liste des variables numériques utilisées dans le projet
NUMERICAL_VARIABLES: List[str] = ["age", "fare", "sibsp", "parch"]

# Liste des variables catégoriques utilisées dans le projet
CATEGORICAL_VARIABLES: List[str] = ["pclass", "sex", "embarked", "title", "cabin"]

# Liste des colonnes à supprimer lors du nettoyage
COLUMNS_TO_DROP: List[str] = ["name", "ticket", "boat", "body", "home.dest"]
