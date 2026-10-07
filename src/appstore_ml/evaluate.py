from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from .common import FEATURES, write_csv, write_json
from .features import model_inputs


def continuous_score(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)), "decision_function (not a probability)"
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)], "class-1 probability"


def evaluate_models(fitted, test, train, selected, output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    X, y = model_inputs(test), test.high_rated
    summary, matrices = [], {}
    fig_cm, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig_roc, roc_axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    all_predictions = []
    for (name, model), axis in zip(fitted.items(), axes.flat):
        prediction = model.predict(X)
        score, score_type = continuous_score(model, X)
        row = {"model": name, "test_n": len(y),
               "f1": f1_score(y, prediction, zero_division=0),
               "roc_auc": roc_auc_score(y, score),
               "precision": precision_score(y, prediction, zero_division=0),
               "recall": recall_score(y, prediction, zero_division=0),
               "balanced_accuracy": balanced_accuracy_score(y, prediction)}
        summary.append(row)
        matrix = confusion_matrix(y, prediction, labels=[0, 1])
        matrices[name] = {"labels": [0, 1], "rows": "true", "columns": "predicted", "matrix": matrix.tolist()}
        ConfusionMatrixDisplay(matrix, display_labels=["<4.5", ">=4.5"]).plot(ax=axis, colorbar=False)
        axis.set_title(name + (" [selected by CV]" if name == selected else ""))
        RocCurveDisplay.from_predictions(y, score, name=name, ax=roc_axis)
        private = test[["trackId", "high_rated", "averageUserRating", "userRatingCount"] + FEATURES].copy()
        private["model"] = name
        private["predicted"] = prediction
        private["score"] = score
        private["score_type"] = score_type
        private["error_type"] = np.where((y == 0) & (prediction == 1), "FP",
                                         np.where((y == 1) & (prediction == 0), "FN", "correct"))
        all_predictions.append(private)
    roc_axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    roc_axis.legend(loc="lower right")
    fig_cm.savefig(figures / "confusion_matrices.png", dpi=160)
    fig_roc.savefig(figures / "roc_curves.png", dpi=160)
    plt.close("all")
    predictions = pd.concat(all_predictions, ignore_index=True)
    write_csv(predictions, output_dir / "private/test_predictions.csv")
    write_json(output_dir / "reports/confusion_matrices.json", matrices)
    metrics = pd.DataFrame(summary)
    write_csv(metrics, output_dir / "reports/test_metrics.csv")
    selected_errors = predictions.loc[(predictions.model == selected) & (predictions.error_type != "correct")].copy()
    selected_errors["boundary_distance"] = (selected_errors.averageUserRating - 4.5).abs()
    examples = pd.concat([selected_errors.loc[selected_errors.error_type == kind]
                         .sort_values(["boundary_distance", "trackId"]).head(3) for kind in ["FP", "FN"]])
    public = []
    for index, (_, row) in enumerate(examples.iterrows(), 1):
        missing = int(row[FEATURES].isna().sum())
        observations = []
        if row.boundary_distance <= 0.1:
            observations.append("True rating is within 0.1 of the 4.5 boundary")
        if missing:
            observations.append(f"{missing} feature(s) required training-fitted imputation")
        genre_n = int(train.primary_genre.eq(row.primary_genre).sum())
        if genre_n < 20:
            observations.append("Genre has fewer than 20 training examples")
        if not observations:
            observations.append("Metadata alone does not explain the observed rating")
        public.append({"case": f"Case {index}", "error_type": row.error_type,
                       "actual_class": int(row.high_rated), "predicted_class": int(row.predicted),
                       "observation": "; ".join(observations),
                       "interpretation": "Descriptive association; no causal explanation is established."})
    cases = pd.DataFrame(public, columns=["case", "error_type", "actual_class", "predicted_class",
                                         "observation", "interpretation"])
    write_csv(cases, output_dir / "reports/error_cases_anonymized.csv")
    return metrics, cases, matrices


def make_overview(train, test, cv, test_metrics, output_dir):
    figures = Path(output_dir) / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    counts = pd.DataFrame({"Train": train.high_rated.value_counts(), "Test": test.high_rated.value_counts()})
    counts = counts.reindex([0, 1], fill_value=0)
    counts.index = ["Class 0: <4.5", "Class 1: >=4.5"]
    counts.plot.bar(ax=axes[0], rot=0, color=["#164e63", "#f59e0b"])
    axes[0].set_ylabel("Applications")
    axes[0].set_title("Fixed stratified 80/20 split")
    positions = np.arange(len(cv))
    axes[1].bar(positions, cv.cv_f1_mean, yerr=cv.cv_f1_std, capsize=5, color="#164e63",
                label="CV mean +/- sample SD")
    observed = test_metrics.set_index("model").loc[cv.model, "f1"]
    axes[1].scatter(positions, observed, color="#f59e0b", label="Holdout", zorder=3)
    axes[1].set_xticks(positions, cv.model)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("F1 for class 1")
    axes[1].set_title("Baseline matters")
    axes[1].legend(fontsize=9)
    fig.savefig(figures / "overview.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for column, axis in zip(FEATURES[:5], axes.flat):
        for cls, color in [(0, "#f59e0b"), (1, "#164e63")]:
            values = train.loc[train.high_rated.eq(cls), column].dropna()
            axis.hist(values, bins=20, alpha=.6, label=f"Class {cls}", color=color)
        axis.set_title(column)
        axis.set_ylabel("Train count")
    train.primary_genre.value_counts().head(8).sort_values().plot.barh(ax=axes.flat[-1], color="#164e63")
    axes.flat[-1].set_title("Top genres (train only)")
    axes.flat[0].legend()
    fig.savefig(figures / "train_eda.png", dpi=150)
    plt.close(fig)
