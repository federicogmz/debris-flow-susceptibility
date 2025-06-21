# %%
"""
morphometry.py

Visualize morphometric classification boundaries using trained SVM models.
Imports pre-trained models and scaler, applies to input GeoDataFrame.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_morphometry(
    basins: gpd.GeoDataFrame,
    feature_pairs: list[tuple[str, str]],
    model_dict: dict[int, object],
    scaler: object,
) -> None:
    """
    Plot decision regions for each feature pair on the scaled morphometric data.

    Parameters:
        basins: GeoDataFrame with columns specified in feature_pairs.
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
    # Example usage: load basins and models, then visualize
    basin_shp = "data/torrencialidad.shp"

    basins = gpd.read_file(basin_shp)

    # Load compressed models and scaler
    with gzip.open("../models/models.pkl.gz", "rb") as mf:
        models = pickle.load(mf)
    with gzip.open("../models/scaler.pkl.gz", "rb") as sf:
        scaler = pickle.load(sf)

    # Define feature pairs
    pair_list = [("A (sq. km)", "Rh"), ("Su", "Rh"), ("M", "Rh")]

    visualize_morphometry(basins, pair_list, models, scaler)
