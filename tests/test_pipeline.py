"""Unit tests for debris-flow-susceptibility training and prediction pipeline."""

import gzip
import pickle
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.train_model import save_artifacts, train_and_plot_models


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_df(repo_root: Path) -> pd.DataFrame:
    csv_path = repo_root / "data" / "torrencialidad.csv"
    assert csv_path.exists(), f"Data file {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    return df


def test_data_integrity(dataset_df: pd.DataFrame):
    """Verify that input dataset contains required features and target column."""
    required_cols = ["A (sq. km)", "Su", "M", "Rh", "FFR"]
    for col in required_cols:
        assert col in dataset_df.columns, f"Missing required column: {col}"

    assert len(dataset_df) > 0, "Dataset is empty"
    assert dataset_df[required_cols].isna().sum().sum() == 0, "Dataset contains NaNs"
    assert set(dataset_df["FFR"].unique()) == {"T", "NT"}, "Target classes mismatch"


@patch("matplotlib.pyplot.show")
def test_train_models(mock_show, dataset_df: pd.DataFrame):
    """Verify that models train without errors and scaler normalizes correctly."""
    feature_pairs = [("A (sq. km)", "Rh"), ("Su", "Rh"), ("M", "Rh")]
    models, scaler = train_and_plot_models(dataset_df, feature_pairs)
    plt.close("all")

    assert len(models) == len(feature_pairs), "Incorrect number of models returned"
    assert hasattr(scaler, "mean_"), "StandardScaler was not fitted"

    # Test prediction with scaled dummy data
    features = ["A (sq. km)", "Su", "M", "Rh"]
    X_sample = dataset_df[features].iloc[:5]
    scaled_sample = scaler.transform(X_sample)

    for idx, model in models.items():
        xi = features.index(feature_pairs[idx][0])
        yi = features.index(feature_pairs[idx][1])
        preds = model.predict(scaled_sample[:, [xi, yi]])
        assert len(preds) == len(X_sample)
        assert set(np.unique(preds)).issubset({0, 1})


def test_save_and_load_artifacts(tmp_path: Path, dataset_df: pd.DataFrame):
    """Verify that artifacts can be serialized and deserialized with gzip compression."""
    feature_pairs = [("A (sq. km)", "Rh"), ("Su", "Rh")]
    with patch("matplotlib.pyplot.show"):
        models, scaler = train_and_plot_models(dataset_df, feature_pairs)
        plt.close("all")

    save_artifacts(models, scaler, tmp_path)

    models_file = tmp_path / "models.pkl.gz"
    scaler_file = tmp_path / "scaler.pkl.gz"
    assert models_file.exists(), "models.pkl.gz was not created"
    assert scaler_file.exists(), "scaler.pkl.gz was not created"

    with gzip.open(models_file, "rb") as mf:
        loaded_models = pickle.load(mf)
    with gzip.open(scaler_file, "rb") as sf:
        loaded_scaler = pickle.load(sf)

    assert 0 in loaded_models
    assert hasattr(loaded_scaler, "mean_")


def test_existing_repository_artifacts(repo_root: Path, dataset_df: pd.DataFrame):
    """Verify that pre-trained artifacts in the repository load and can predict."""
    models_file = repo_root / "models" / "models.pkl.gz"
    scaler_file = repo_root / "models" / "scaler.pkl.gz"
    assert models_file.exists(), "Repository models.pkl.gz missing"
    assert scaler_file.exists(), "Repository scaler.pkl.gz missing"

    with gzip.open(models_file, "rb") as mf:
        models = pickle.load(mf)
    with gzip.open(scaler_file, "rb") as sf:
        scaler = pickle.load(sf)

    assert len(models) == 3
    assert hasattr(scaler, "mean_")

    features = ["A (sq. km)", "Su", "M", "Rh"]
    X_sample = scaler.transform(dataset_df[features].iloc[:5])
    preds = models[0].predict(X_sample[:, [0, 3]])
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1})
