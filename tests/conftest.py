"""Fixtures partagées : données synthétiques au format de diabetic_data.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modele_pipeline import prepare_data  # noqa: E402


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": rng.integers(0, 250, n),
            "time_in_hospital": rng.integers(1, 15, n),
            "age": rng.choice(["[40-50)", "[50-60)", "[60-70)"], n),
            "gender": rng.choice(["Male", "Female"], n),
            "weight": "?",
            "discharge_disposition_id": rng.integers(1, 5, n),
            "readmitted": rng.choice(["NO", "<30", ">30"], n),
            "num_lab_procedures": rng.integers(1, 80, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 5, n),
            "number_inpatient": rng.integers(0, 5, n),
            "number_diagnoses": rng.integers(1, 9, n),
        }
    )
    path = tmp_path_factory.mktemp("data") / "diabetic_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def data(csv_path):
    return prepare_data(csv_path)
