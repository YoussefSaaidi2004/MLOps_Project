"""Tests de sécurité : analyse statique, secrets, appels dangereux, confidentialité des données."""

import ast
import json
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import pandas as pd
import pytest

import main as pipeline
from modele_pipeline import load_model

ROOT = Path(__file__).resolve().parent.parent
FILES = ["modele_pipeline.py", "main.py", "pipeline_prefect.py"]
IDENTIFIERS = {"encounter_id", "patient_nbr"}
DANGEROUS_NAMES = {"eval", "exec"}
DANGEROUS_ATTRS = {("os", "system"), ("os", "popen"), ("pickle", "load"), ("pickle", "loads")}


def _tree(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


# --------------------------- analyse statique ------------------------------ #
def test_bandit_reports_no_medium_or_high_issue():
    cmd = [sys.executable, "-m", "bandit", "-q", "-ll", *FILES]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)  # nosec B603
    assert result.returncode == 0, result.stdout + result.stderr


# ------------------------------- flake8 ------------------------------------ #
def _flake8(*args):
    cmd = [sys.executable, "-m", "flake8", "--max-line-length", "100", *args, *FILES]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)  # nosec B603


def test_flake8_critical_errors():
    """Erreurs bloquantes : syntaxe (E9), erreurs F63/F7, noms indéfinis (F82)."""
    result = _flake8("--select=E9,F63,F7,F82", "--show-source")
    assert result.returncode == 0, result.stdout + result.stderr


def test_flake8_no_unused_or_shadowed_code():
    """Imports inutilisés (F401), variables inutilisées (F841), redéfinitions (F811)."""
    result = _flake8("--select=F401,F841,F811")
    assert result.returncode == 0, result.stdout + result.stderr


def test_flake8_full_report_is_clean():
    """Rapport flake8 complet (style, longueur de ligne, complexité simple)."""
    result = _flake8("--max-complexity=15")
    assert result.returncode == 0, result.stdout + result.stderr


# ------------------------------- secrets ----------------------------------- #
SECRET_PATTERN = re.compile(
    r"(password|passwd|secret|api[_-]?key|token)\s*=\s*[\"'][^\"']+[\"']", re.IGNORECASE
)


@pytest.mark.parametrize("name", FILES)
def test_no_hardcoded_secrets(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert SECRET_PATTERN.search(text) is None


# ---------------------------- appels dangereux ------------------------------ #
@pytest.mark.parametrize("name", FILES)
def test_no_dangerous_calls(name):
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            assert func.id not in DANGEROUS_NAMES, f"{name}:{node.lineno} appelle {func.id}()"
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            assert (
                func.value.id,
                func.attr,
            ) not in DANGEROUS_ATTRS, f"{name}:{node.lineno} appelle {func.value.id}.{func.attr}()"


@pytest.mark.parametrize("name", FILES)
def test_subprocess_calls_use_argument_lists_without_shell(name):
    for node in ast.walk(_tree(name)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"):
            continue
        for kw in node.keywords:
            if kw.arg == "shell":
                assert not (isinstance(kw.value, ast.Constant) and kw.value.value is True)
        if node.args:
            assert not isinstance(
                node.args[0], (ast.Constant, ast.JoinedStr)
            ), f"{name}:{node.lineno} commande passée sous forme de chaîne"


# --------------------- confidentialité des données patients ----------------- #
@pytest.fixture(scope="module")
def outputs(csv_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("sec_out")
    data = pipeline.step_prepare(csv_path)
    models = pipeline.step_train(data)
    pipeline.step_save(models, data, out)
    pipeline.step_evaluate(models, data, out)
    return out, data


def test_predictions_file_has_no_patient_identifiers(outputs):
    out, _ = outputs
    columns = set(pd.read_csv(out / "predictions_test.csv").columns)
    assert not (columns & IDENTIFIERS)


def test_metadata_has_no_patient_identifiers(outputs):
    out, _ = outputs
    meta = json.loads((out / "metadata.json").read_text())
    assert not (set(meta) & IDENTIFIERS)
    assert not any(ident in json.dumps(meta) for ident in IDENTIFIERS)


def test_saved_models_do_not_embed_training_data(outputs):
    out, _ = outputs
    for path in out.glob("*.joblib"):
        bundle = load_model(path)
        for key, value in bundle.items():
            assert not isinstance(value, (pd.DataFrame, pd.Series)), f"{path.name}:{key}"
        assert not (set(bundle["feature_columns"]) & IDENTIFIERS)


def test_model_features_exclude_identifiers(outputs):
    _, data = outputs
    assert not (set(data["feature_columns"]) & IDENTIFIERS)


def test_load_model_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "inexistant.joblib")
