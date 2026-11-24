"""
modele_pipeline.py
Pipeline de classification binaire : prédire un séjour long (>= 7 jours)
sur diabetic_data.csv (XGBoost / LightGBM).

Fonctions principales :
    prepare_data()   : charge et prétraite les données, split train/test par patient
    train_model()    : entraîne un modèle (XGBoost ou LightGBM)
    evaluate_model() : calcule les métriques pour plusieurs seuils
    save_model()     : sauvegarde le modèle et ses métadonnées
    load_model()     : recharge un modèle sauvegardé
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LONG_STAY_DAYS = 7

# Colonnes exclues pour éviter les fuites de données (cible, identifiants, variables de sortie)
EXCLUDE_COLS = [
    "encounter_id",
    "patient_nbr",
    "time_in_hospital",
    "discharge_disposition_id",
    "readmitted",
    "weight",
]
NUMERIC_COLS = [
    "age_num",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


# --------------------------------------------------------------------------- #
# 1. Préparation des données
# --------------------------------------------------------------------------- #
def prepare_data(
    csv_path: str | Path,
    random_state: int = RANDOM_STATE,
    n_splits: int = 5,
    long_stay_days: int = LONG_STAY_DAYS,
) -> dict:
    """Charge le CSV, prétraite les variables et sépare train/test par patient.

    Retourne un dictionnaire : X_train, X_test, y_train, y_test, scale_pos_weight,
    feature_columns, numeric_columns, categorical_columns, train_patients, test_patients.
    """
    df = pd.read_csv(csv_path, na_values=["?"], keep_default_na=False, low_memory=False)

    # Cible binaire : séjour long
    stay_days = pd.to_numeric(df["time_in_hospital"], errors="raise").astype(int)
    y = (stay_days >= long_stay_days).astype(int)
    strat_labels = pd.Series(
        np.select([stay_days <= 3, stay_days <= 6], [0, 1], default=2), index=df.index
    ).astype(int)

    # Âge : tranche "[a-b)" -> milieu numérique
    work = df.copy()
    if "age" in work:
        bounds = work["age"].astype("string").str.extract(r"\[(\d+)-(\d+)\)")
        work["age_num"] = bounds.astype(float).mean(axis=1)
        work = work.drop(columns=["age"])

    X = work.drop(columns=[c for c in EXCLUDE_COLS if c in work.columns]).copy()
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)

    num_cols = [c for c in NUMERIC_COLS if c in X.columns]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = (
            X[c]
            .astype("string")
            .fillna("Inconnu")
            .replace({"?": "Inconnu"})
            .astype(str)
            .astype("category")
        )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    strat_labels = strat_labels.reset_index(drop=True)
    groups = df["patient_nbr"].reset_index(drop=True)

    # Split par patient : aucun patient commun entre train et test
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, strat_labels, groups=groups))
    assert not (
        set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    ), "Fuite patient train/test"

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # Ratio négatifs/positifs calculé sur le train uniquement
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    print(
        f"Dataset {df.shape} | train/test {len(X_train)}/{len(X_test)} | "
        f"positifs train {y_train.mean():.4f} | scale_pos_weight {scale_pos:.4f}",
        flush=True,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scale_pos_weight": scale_pos,
        "feature_columns": list(X.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "train_patients": int(groups.iloc[train_idx].nunique()),
        "test_patients": int(groups.iloc[test_idx].nunique()),
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }


# --------------------------------------------------------------------------- #
# 2. Entraînement
# --------------------------------------------------------------------------- #
def xgb_params(random_state: int = RANDOM_STATE) -> dict:
    return dict(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        n_jobs=min(4, os.cpu_count() or 2),
        random_state=random_state,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    balanced: bool = False,
    random_state: int = RANDOM_STATE,
):
    """Entraîne un modèle.

    model_type : "xgboost" ou "lightgbm"
    balanced   : si True, scale_pos_weight = nb négatifs / nb positifs (train uniquement)
    """
    weight = float((y_train == 0).sum() / (y_train == 1).sum()) if balanced else 1.0
    t = time.time()

    if model_type == "xgboost":
        model = XGBClassifier(**xgb_params(random_state), scale_pos_weight=weight)
        model.fit(X_train, y_train, verbose=False)
    elif model_type == "lightgbm":
        model = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=50,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            subsample=0.9,
            subsample_freq=1,
            max_bin=127,
            scale_pos_weight=weight,
            random_state=random_state,
            n_jobs=min(4, os.cpu_count() or 2),
            verbosity=-1,
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError("model_type doit être 'xgboost' ou 'lightgbm'")

    print(
        f"Entraînement {model_type} (balanced={balanced}) : {(time.time() - t) / 60:.2f} min",
        flush=True,
    )
    return model


# --------------------------------------------------------------------------- #
# 3. Évaluation
# --------------------------------------------------------------------------- #
def _metrics(name: str, y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "Modèle": name,
        "Seuil": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, pred),
        "Précision séjour long": precision_score(y_true, pred, zero_division=0),
        "Rappel séjour long": recall_score(y_true, pred, zero_division=0),
        "F1 séjour long": f1_score(y_true, pred, zero_division=0),
        "Macro-F1": f1_score(y_true, pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "Prédictions positives (%)": 100 * pred.mean(),
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Modèle",
    thresholds: tuple = (0.5, 0.27),
    verbose: bool = True,
):
    """Évalue un modèle à plusieurs seuils.

    Retourne (DataFrame des métriques, probabilités prédites).
    """
    proba = model.predict_proba(X_test)[:, 1]
    results = pd.DataFrame([_metrics(name, y_test, proba, th) for th in thresholds])

    if verbose:
        for th in thresholds:
            pred = (proba >= th).astype(int)
            print(f"\n[{name}] seuil {th}")
            print("Matrice de confusion :", confusion_matrix(y_test, pred).tolist())
            print(classification_report(y_test, pred, digits=4))
    return results, proba


# --------------------------------------------------------------------------- #
# 4. Sauvegarde / chargement
# --------------------------------------------------------------------------- #
def save_model(model, path: str | Path, **metadata) -> Path:
    """Sauvegarde le modèle avec ses métadonnées (colonnes, seuil, etc.) via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **metadata}, path)
    print(f"Modèle sauvegardé : {path}", flush=True)
    return path


def load_model(path: str | Path) -> dict:
    """Recharge un modèle sauvegardé. Retourne le dictionnaire {'model': ..., ...métadonnées}."""
    bundle = joblib.load(path)
    print(f"Modèle chargé : {path}", flush=True)
    return bundle
