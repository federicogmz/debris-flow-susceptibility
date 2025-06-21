# %%

"""
train_model.py

Train SVM models for debris-flow susceptibility and visualize results.
Saves trained models and scaler using pickle with gzip compression.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


def train_and_plot_models(
    df: pd.DataFrame, feature_combinations: list
) -> tuple[dict, StandardScaler]:
    """
    Train linear SVM classifiers on specified feature pairs and plot decision boundaries.

    Parameters:
        df: DataFrame containing features ['A (sq. km)', 'Su', 'M', 'Rh'] and target 'FFR'.
        feature_combinations: List of (x_feature, y_feature) pairs.

    Returns:
        models: dict mapping index to trained SVC instances.
        scaler: fitted StandardScaler used for feature normalization.
    """
    df = df.copy()
    df["FFR"] = (df["FFR"] == "T").astype(int)
    features = ["A (sq. km)", "Su", "M", "Rh"]
    X = df[features]
    y = df["FFR"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(feature_combinations), figsize=(18, 6))

    for idx, ((x_col, y_col), ax) in enumerate(zip(feature_combinations, axes)):
        xi = features.index(x_col)
        yi = features.index(y_col)
        X_pair = X_scaled[:, [xi, yi]]
        model = SVC(kernel="linear", C=10, gamma=0.1)
        model.fit(X_pair, y)
        models[idx] = model

        # decision boundary grid
        xx, yy = np.meshgrid(
            np.linspace(X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1, 100),
            np.linspace(X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1, 100),
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cmap = ListedColormap(["cornflowerblue", "darkorange"])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
        ax.scatter(
            X_pair[y == 0, 0],
            X_pair[y == 0, 1],
            c="cornflowerblue",
            label="No torrential",
            edgecolor="k",
            s=80,
            alpha=0.8,
        )
        ax.scatter(
            X_pair[y == 1, 0],
            X_pair[y == 1, 1],
            c="darkorange",
            label="Torrential",
            edgecolor="k",
            s=80,
            alpha=0.8,
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()

    plt.tight_layout()
    plt.show()
    return models, scaler


def save_artifacts(models: dict, scaler: StandardScaler, output_dir: Path) -> None:
    """
    Save models and scaler to output directory using gzip-compressed pickle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "models.pkl.gz", "wb") as mf:
        pickle.dump(models, mf)
    with gzip.open(output_dir / "scaler.pkl.gz", "wb") as sf:
        pickle.dump(scaler, sf)


def main():
    # Load data
    data_path = Path(__file__).parents[1] / "data" / "torrencialidad.csv"
    df = pd.read_csv(data_path)
    feature_pairs = [("A (sq. km)", "Rh"), ("Su", "Rh"), ("M", "Rh")]

    # Train and plot
    models, scaler = train_and_plot_models(df, feature_pairs)

    # Save artifacts
    artifacts_dir = Path(__file__).parents[1] / "models"
    save_artifacts(models, scaler, artifacts_dir)
    print(f"Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    main()
