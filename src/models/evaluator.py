# src/models/evaluator.py
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger
from sklearn.pipeline import Pipeline
from typing import Any

def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Évalue la performance du modèle en affichant le ROC-AUC et l'accuracy pour le train et le test.

    Parameters:
        pipeline (Pipeline): Pipeline entraînée.
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        y_train (pd.Series): Cible d'entraînement.
        y_test (pd.Series): Cible de test.
    """
    try:
        logger.info("Évaluation du modèle")
        y_train_pred_class = pipeline.predict(X_train)
        y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_accuracy = accuracy_score(y_train, y_train_pred_class)
        logger.info("Train ROC-AUC: {:.4f}", train_roc_auc)
        logger.info("Train Accuracy: {:.4f}", train_accuracy)

        y_test_pred_class = pipeline.predict(X_test)
        y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_accuracy = accuracy_score(y_test, y_test_pred_class)
        logger.info("Test ROC-AUC: {:.4f}", test_roc_auc)
        logger.info("Test Accuracy: {:.4f}", test_accuracy)
    except Exception as e:
        logger.error("Erreur lors de l'évaluation du modèle: {}", e)
        raise
