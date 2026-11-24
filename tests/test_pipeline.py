"""Tests unitaires : préparation, entraînement, évaluation, sauvegarde, étapes de main.py."""

import numpy as np
import pandas as pd
import pytest

import main as pipeline
from modele_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


# ----------------------------- prepare_data -------------------------------- #
def test_prepare_data_structure(data):
    for key in ("X_train", "X_test", "y_train", "y_test", "scale_pos_weight", "feature_columns"):
        assert key in data
    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_test"]) == len(data["y_test"])
    assert list(data["X_train"].columns) == data["feature_columns"]


def test_target_is_stay_of_7_days_or_more(csv_path, data):
    raw = pd.read_csv(csv_path)
    expected_positives = int((raw["time_in_hospital"] >= 7).sum())
    assert set(data["y_train"].unique()) <= {0, 1}
    assert int(data["y_train"].sum() + data["y_test"].sum()) == expected_positives


def test_split_covers_all_rows_and_patients(csv_path, data):
    raw = pd.read_csv(csv_path)
    assert len(data["X_train"]) + len(data["X_test"]) == len(raw)
    assert data["train_patients"] + data["test_patients"] == raw["patient_nbr"].nunique()


def test_leaky_and_identifier_columns_are_excluded(data):
    for col in (
        "time_in_hospital",
        "encounter_id",
        "patient_nbr",
        "discharge_disposition_id",
        "readmitted",
        "weight",
    ):
        assert col not in data["feature_columns"]


def test_age_is_converted_to_numeric(data):
    assert "age" not in data["feature_columns"]
    ages = data["X_train"]["age_num"]
    assert pd.api.types.is_numeric_dtype(ages)
    assert ages.between(40, 70).all()


def test_categorical_columns_are_typed_without_missing_values(data):
    for col in data["categorical_columns"]:
        assert isinstance(data["X_train"][col].dtype, pd.CategoricalDtype)
        assert not data["X_train"][col].isna().any()


def test_scale_pos_weight_is_negatives_over_positives(data):
    y = data["y_train"]
    assert data["scale_pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_prepare_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_data(tmp_path / "absent.csv")


# ----------------------------- train_model --------------------------------- #
def test_train_model_rejects_unknown_type(data):
    with pytest.raises(ValueError):
        train_model(data["X_train"], data["y_train"], model_type="random_forest")


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm"])
def test_train_model_returns_probabilities(data, model_type):
    model = train_model(data["X_train"], data["y_train"], model_type=model_type)
    proba = model.predict_proba(data["X_test"])[:, 1]
    assert len(proba) == len(data["X_test"])
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------- evaluate_model ------------------------------- #
def test_evaluate_model_metrics(data):
    model = train_model(data["X_train"], data["y_train"], model_type="xgboost")
    results, proba = evaluate_model(
        model, data["X_test"], data["y_test"], name="m", thresholds=(0.5, 0.27), verbose=False
    )
    assert list(results["Seuil"]) == [0.5, 0.27]
    for col in ("Accuracy", "Balanced accuracy", "ROC-AUC", "PR-AUC", "Macro-F1"):
        assert results[col].between(0, 1).all()
    low, high = results["Prédictions positives (%)"]
    assert low <= high  # un seuil plus bas prédit au moins autant de positifs
    assert len(proba) == len(data["X_test"])


# --------------------------- save_model / load_model ------------------------ #
def test_save_and_load_roundtrip(data, tmp_path):
    model = train_model(data["X_train"], data["y_train"], model_type="xgboost")
    path = save_model(model, tmp_path / "sub" / "m.joblib", feature_columns=["a"], note="x")
    bundle = load_model(path)
    assert bundle["feature_columns"] == ["a"]
    before = model.predict_proba(data["X_test"])[:, 1]
    after = bundle["model"].predict_proba(data["X_test"])[:, 1]
    assert np.allclose(before, after)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


# ------------------------------ étapes main.py ----------------------------- #
def test_main_steps_end_to_end(csv_path, tmp_path):
    out = tmp_path / "out"
    data = pipeline.step_prepare(csv_path)
    models = pipeline.step_train(data)
    assert set(models) == set(pipeline.MODEL_CONFIGS)

    pipeline.step_save(models, data, out)
    for name in pipeline.MODEL_CONFIGS:
        assert (out / f"{pipeline.safe_name(name)}.joblib").exists()
    assert (out / "metadata.json").exists()

    reloaded = pipeline.step_load(out)
    results, _ = pipeline.step_evaluate(reloaded, data, out)
    assert len(results) == len(pipeline.MODEL_CONFIGS) * len(pipeline.THRESHOLDS)
    assert (out / "metriques_xgboost_comparaison.csv").exists()
    assert (out / "predictions_test.csv").exists()


def test_safe_name_has_no_spaces_or_accents():
    assert pipeline.safe_name("XGBoost équilibré") == "xgboost_equilibre"
