"""
pipeline_prefect.py
Orchestration du pipeline avec Prefect (tasks + flows).

Usage :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train          # alias : entrainement
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code           # qualité du code uniquement
    python pipeline_prefect.py --flow install
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, get_run_logger, task

import main as pipeline

# Étape de qualité/sécurité limitée à nos 3 fichiers (sinon scan du venv, très long)
CODE_FILES = ["modele_pipeline.py", "main.py", "pipeline_prefect.py"]
LINE_LENGTH = "100"


def run_cmd(cmd):
    """Exécute une commande dans l'interpréteur courant ; échoue si le code retour != 0."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.info(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")


# --------------------------------------------------------------------------- #
# Tasks : suivi du code
# --------------------------------------------------------------------------- #
@task(name="Installer les dépendances")
def install_dependencies(requirements="requirements.txt"):
    run_cmd([sys.executable, "-m", "pip", "install", "-r", requirements])


@task(name="Formatage du code")
def format_code():
    run_cmd([sys.executable, "-m", "isort", "--profile", "black", "-l", LINE_LENGTH, *CODE_FILES])
    run_cmd([sys.executable, "-m", "black", "-l", LINE_LENGTH, *CODE_FILES])


@task(name="Qualité du code")
def check_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length", LINE_LENGTH, *CODE_FILES])


@task(name="Sécurité du code")
def check_security():
    # -ll : n'échoue que pour les alertes de sévérité moyenne ou haute
    run_cmd([sys.executable, "-m", "bandit", "-q", "-ll", *CODE_FILES])


@task(name="Tests unitaires")
def run_unit_tests():
    run_cmd([sys.executable, "-m", "pytest", "-q", "tests/test_pipeline.py"])


@task(name="Tests de sécurité")
def run_security_tests():
    run_cmd([sys.executable, "-m", "pytest", "-q", "tests/test_security.py"])


# --------------------------------------------------------------------------- #
# Tasks : data / modèles (appellent les fonctions de main.py)
# --------------------------------------------------------------------------- #
@task(name="Préparation des données")
def prepare_task(csv_path):
    return pipeline.step_prepare(csv_path)


@task(name="Entraînement du modèle")
def train_task(data):
    return pipeline.step_train(data)


@task(name="Sauvegarder le modèle")
def save_task(models, data, out):
    pipeline.step_save(models, data, out)


@task(name="Charger le modèle")
def load_task(out):
    return pipeline.step_load(out)


@task(name="Évaluer le modèle")
def evaluate_task(models, data, out):
    results, _ = pipeline.step_evaluate(models, data, out)
    return results


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #
@flow(name="install")
def install():
    install_dependencies()


@flow(name="code")
def code():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_security_tests()
    run_unit_tests()


@flow(name="train")
def train(csv_path="diabetic_data.csv", out="resultats_xgboost_binaire"):
    data = prepare_task(csv_path)
    models = train_task(data)
    save_task(models, data, out)  # sauvegarde nécessaire pour le flow evaluate


@flow(name="evaluate")
def evaluate(csv_path="diabetic_data.csv", out="resultats_xgboost_binaire"):
    # Même graine => même split train/test que lors de l'entraînement
    data = prepare_task(csv_path)
    models = load_task(out)
    return evaluate_task(models, data, out)


@flow(name="all")
def all_steps(csv_path="diabetic_data.csv", out="resultats_xgboost_binaire"):
    # Étapes de suivi du code avant la préparation des données
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_security_tests()
    run_unit_tests()
    # Étapes 2-3-4 et 6
    data = prepare_task(csv_path)
    models = train_task(data)
    save_task(models, data, out)
    return evaluate_task(models, data, out)


FLOWS = {
    "all": all_steps,
    "train": train,
    "entrainement": train,
    "evaluate": evaluate,
    "code": code,
    "install": install,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=sorted(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
