"""
main.py
Étapes du pipeline (appelées par pipeline_prefect.py) et exécution complète.

Usage :
    python main.py
    python main.py --csv diabetic_data.csv --out resultats_xgboost_binaire
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modele_pipeline import (
    RANDOM_STATE,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
    xgb_params,
)

THRESHOLDS = (0.5, 0.27)  # 0.27 : seuil expérimental (à valider sur un jeu de validation)
MODEL_CONFIGS = {
    "XGBoost sans équilibrage": dict(model_type="xgboost", balanced=False),
    "XGBoost équilibré": dict(model_type="xgboost", balanced=True),
    "LightGBM sans équilibrage": dict(model_type="lightgbm", balanced=False),
}
REFERENCE_MODEL = "XGBoost sans équilibrage"


def safe_name(name):
    """Nom de fichier sans espaces ni accents."""
    return name.lower().replace(" ", "_").replace("é", "e")


def step_prepare(csv_path):
    """Étape 2 : chargement et prétraitement des données."""
    return prepare_data(csv_path, random_state=RANDOM_STATE)


def step_train(data):
    """Étape 3 : entraînement des trois modèles sur le même split."""
    return {
        name: train_model(data["X_train"], data["y_train"], **cfg)
        for name, cfg in MODEL_CONFIGS.items()
    }


def step_save(models, data, out):
    """Étape 4 : sauvegarde de chaque modèle avec ses métadonnées."""
    out = Path(out)
    out.mkdir(exist_ok=True)
    for name, model in models.items():
        save_model(
            model,
            out / f"{safe_name(name)}.joblib",
            name=name,
            threshold_reference=0.27,
            feature_columns=data["feature_columns"],
            numeric_columns=data["numeric_columns"],
            categorical_columns=data["categorical_columns"],
            scale_pos_weight=data["scale_pos_weight"],
            random_state=RANDOM_STATE,
        )
    meta = {
        "rows": data["n_rows"],
        "columns": data["n_columns"],
        "train_rows": len(data["X_train"]),
        "test_rows": len(data["X_test"]),
        "train_unique_patients": data["train_patients"],
        "test_unique_patients": data["test_patients"],
        "positive_rate_test": float(data["y_test"].mean()),
        "scale_pos_weight": data["scale_pos_weight"],
        "n_features": len(data["feature_columns"]),
        "params": xgb_params(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


def step_load(out):
    """Étape 5 : rechargement des modèles sauvegardés."""
    out = Path(out)
    models = {}
    for name in MODEL_CONFIGS:
        bundle = load_model(out / f"{safe_name(name)}.joblib")
        models[name] = bundle["model"]
    return models


def step_evaluate(models, data, out):
    """Étape 6 : évaluation, export des métriques et des prédictions."""
    out = Path(out)
    out.mkdir(exist_ok=True)
    X_test, y_test = data["X_test"], data["y_test"]
    all_results, probas = [], {}
    for name, model in models.items():
        res, proba = evaluate_model(model, X_test, y_test, name=name, thresholds=THRESHOLDS)
        all_results.append(res)
        probas[name] = proba
    results = pd.concat(all_results, ignore_index=True)
    print("\nRESULTATS\n", results.round(4).to_string(index=False))
    results.to_csv(out / "metriques_xgboost_comparaison.csv", index=False)

    pred = pd.DataFrame({"y_true": y_test.to_numpy()})
    for name, p in probas.items():
        pred[f"proba_{safe_name(name)}"] = p
        for th in THRESHOLDS:
            pred[f"pred_{safe_name(name)}_{int(th * 100):03d}"] = (p >= th).astype(int)
    pred.to_csv(out / "predictions_test.csv", index=False)
    return results, probas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="diabetic_data.csv")
    parser.add_argument("--out", default="resultats_xgboost_binaire")
    args = parser.parse_args()

    data = step_prepare(args.csv)
    models = step_train(data)
    step_save(models, data, args.out)
    _, probas = step_evaluate(models, data, args.out)

    # Vérification : le modèle rechargé donne les mêmes probabilités
    reloaded = step_load(args.out)[REFERENCE_MODEL]
    again = reloaded.predict_proba(data["X_test"])[:, 1]
    assert np.allclose(again, probas[REFERENCE_MODEL]), "Écart après rechargement"
    print("Vérification OK : prédictions identiques après rechargement.")


if __name__ == "__main__":
    main()
