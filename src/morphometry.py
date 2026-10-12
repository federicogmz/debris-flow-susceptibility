# %%
"""
morphometry.py

Visualize morphometric classification boundaries using trained SVM models.
Imports pre-trained models and scaler, applies to input DataFrame.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_morphometry(
    basins: pd.DataFrame,
    feature_pairs: list[tuple[str, str]],
    model_dict: dict[int, object],
    scaler: object,
) -> None:
    """
    Plot decision regions for each feature pair on the scaled morphometric data.

    Parameters:
        basins: DataFrame with columns specified in feature_pairs.
        feature_pairs: List of (x_feature, y_feature) column name pairs.
        model_dict: Dict mapping index to trained SVM models.
        scaler: Fitted scaler to normalize feature data.
    """
    # Consistent feature order matching training
    feature_order = ["A (sq. km)", "Su", "M", "Rh"]
    data = basins[feature_order]
    scaled_data = scaler.transform(data)

    fig, axes = plt.subplots(1, len(feature_pairs), figsize=(18, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    for idx, ((x_feat, y_feat), ax) in enumerate(zip(feature_pairs, axes)):
        xi, yi = feature_order.index(x_feat), feature_order.index(y_feat)
        X_pair = scaled_data[:, [xi, yi]]
        model = model_dict[idx]
        labels = model.predict(X_pair)

        # Plot decision boundary
        xx, yy = np.meshgrid(
            np.linspace(X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1, 200),
            np.linspace(X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        cmap = ListedColormap(["cornflowerblue", "darkorange"])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        ax.scatter(
            X_pair[labels == 0, 0],
            X_pair[labels == 0, 1],
            c="cornflowerblue",
            label="Non-susceptible",
            edgecolor="k",
            s=50,
        )
        ax.scatter(
            X_pair[labels == 1, 0],
            X_pair[labels == 1, 1],
            c="darkorange",
            label="Susceptible",
            edgecolor="k",
            s=50,
        )

        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "torrencialidad.csv"
    shp_path = repo_root / "data" / "torrencialidad.shp"

    if csv_path.exists():
        basins = pd.read_csv(csv_path)
    elif shp_path.exists():
        import geopandas as gpd
        basins = gpd.read_file(shp_path)
    else:
        raise FileNotFoundError(f"No data file found in {repo_root / 'data'}")

    models_path = repo_root / "models" / "models.pkl.gz"
    scaler_path = repo_root / "models" / "scaler.pkl.gz"

    with gzip.open(models_path, "rb") as mf:
        models = pickle.load(mf)
    with gzip.open(scaler_path, "rb") as sf:
        scaler = pickle.load(sf)

    # Define feature pairs
    pair_list = [("A (sq. km)", "Rh"), ("Su", "Rh"), ("M", "Rh")]

    visualize_morphometry(basins, pair_list, models, scaler)
