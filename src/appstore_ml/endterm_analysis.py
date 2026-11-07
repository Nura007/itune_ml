"""Training-only PCA and frozen-holdout Endterm evaluation.

Public files contain aggregates and coarse, anonymized error descriptions. Detailed
app rows and scores are written only under the ignored ``outputs/private`` tree.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .common import FEATURES, NUMERIC, write_csv, write_json
from .evaluate import continuous_score
from .features import model_inputs
from .pipelines import preprocessing


def _coarse_description(row, train):
    """Describe an actual row without publishing its exact input feature vector."""
    parts = []
    for feature, phrase in [("size_mb", "size"), ("update_recency_days", "update recency")]:
        value = row[feature]
        if pd.isna(value):
            parts.append(f"{phrase} missing")
            continue
        q1, q3 = train[feature].quantile([.25, .75])
        level = "lower quartile" if value <= q1 else "upper quartile" if value >= q3 else "middle half"
        parts.append(f"{phrase} in training {level}")
    return "; ".join(parts)


def analyze_unsupervised(train, output_dir, config):
    """Fit PCA to the seven permitted training inputs, without outcomes or holdout.

    Standardization gives numeric measurements comparable units; one-hot columns
    remain 0/1. Thus category prevalence still affects their variance. PCA loadings
    are descriptive directions, not evidence of natural clusters or causal effects.
    """
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    X = model_inputs(train)
    transformer = preprocessing()
    transformed = transformer.fit_transform(X)
    pca = PCA(svd_solver="full", random_state=config["random_state"])
    scores = pca.fit_transform(transformed)
    if scores.shape[1] < 2:
        raise ValueError("PCA requires at least two components for this analysis.")
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    n90 = min(int(np.searchsorted(cumulative, .90) + 1), len(ratio))
    names = transformer.get_feature_names_out()
    variance = pd.DataFrame({"component": np.arange(1, len(ratio) + 1),
                            "explained_variance_ratio": ratio, "cumulative_variance": cumulative})
    loadings = pd.DataFrame([
        {"component": f"PC{index + 1}", "feature": str(feature), "loading": float(weight)}
        for index, component in enumerate(pca.components_)
        for feature, weight in zip(names, component)
    ])
    write_csv(variance, output_dir / "reports/pca_variance.csv")
    write_csv(loadings, output_dir / "reports/pca_loadings.csv")

    # These are descriptive score strata, not inferred classes or cluster labels.
    groups = pd.qcut(pd.Series(scores[:, 0]), q=4, labels=False, duplicates="drop")
    profiles = []
    for group in sorted(groups.dropna().unique()):
        subset = train.iloc[np.flatnonzero(groups.eq(group).to_numpy())]
        profiles.append({"pc1_stratum": int(group) + 1, "n": len(subset),
                         **{f"median_{feature}": None if subset[feature].dropna().empty
                            else float(subset[feature].median()) for feature in NUMERIC},
                         "most_common_genre": str(subset.primary_genre.fillna("Missing").mode().iloc[0])})
    write_csv(pd.DataFrame(profiles), output_dir / "reports/pca_profiles.csv")

    private = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(scores.shape[1])])
    private.insert(0, "trackId", train.trackId.to_numpy())
    private["pc1_stratum"] = groups + 1
    write_csv(private, output_dir / "private/pca_scores.csv")
    # Examples use the observations nearest the lower/upper training PC1 quartiles.
    examples = []
    for label, quantile in [("Lower PC1 example", .25), ("Upper PC1 example", .75)]:
        index = int(np.argmin(np.abs(scores[:, 0] - np.quantile(scores[:, 0], quantile))))
        examples.append({"example": label, "trackId": int(train.iloc[index].trackId),
                         "description": _coarse_description(train.iloc[index], train)})
    write_csv(pd.DataFrame(examples), output_dir / "private/pca_examples.csv")

    dominant = {}
    for component in [0, 1]:
        order = np.argsort(np.abs(pca.components_[component]))[::-1][:5]
        dominant[f"PC{component + 1}"] = [
            {"feature": str(names[index]), "loading": float(pca.components_[component, index])}
            for index in order
        ]
    result = {
        "method": "PCA; full SVD; training data only",
        "train_n": len(train), "raw_input_count": len(FEATURES),
        "transformed_feature_count": int(transformed.shape[1]),
        "components_for_90_percent": n90,
        "first_two_explained_variance": float(ratio[:2].sum()),
        "pc1_explained_variance": float(ratio[0]),
        "pc2_explained_variance": float(ratio[1]),
        "dominant_loadings": dominant,
        "examples": [{k: value for k, value in row.items() if k != "trackId"} for row in examples],
        "scaling": "Training-median imputation and unit-variance numeric scaling; one-hot categories remain 0/1.",
        "interpretation": (
            "PC1 strata summarize training-feature variation. Inspect signed loadings and median profiles; "
            "component signs are arbitrary. Examples describe actual training objects only in coarse quartiles."
        ),
        "model_connection": (
            "Compare SVM + PCA with SVM using identical training CV folds and the same holdout. "
            "The prediction pipeline fits its own 90%-variance PCA within each training fold; "
            "this exploratory full-training PCA is never reused inside cross-validation."
        ),
        "limitations": [
            "PCA maximizes variance, not rating discrimination; low-variance signals can be lost.",
            "One-hot prevalence and numeric outliers influence component directions.",
            "The two-dimensional density map is not evidence that natural app groups exist.",
            "Explained variance is a reconstruction diagnostic, not a clustering-stability result.",
        ],
    }
    write_json(output_dir / "reports/unsupervised.json", result)

    fig, axis = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    axis.bar(variance.component, ratio, color="#2a7f9e", alpha=.65, label="Individual")
    axis.plot(variance.component, cumulative, color="#b56a19", marker="o", markersize=3,
              label="Cumulative")
    axis.axhline(.9, color="#666666", linestyle="--", label="90% variance")
    axis.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.05),
             title=f"Training-only PCA: {n90} components retain 90% variance")
    axis.legend(fontsize=9)
    fig.savefig(figures / "pca_variance.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    density = axis.hexbin(scores[:, 0], scores[:, 1], gridsize=30, mincnt=3, cmap="Blues")
    fig.colorbar(density, ax=axis, label="Training objects per bin (bins <3 suppressed)")
    axis.set(xlabel=f"PC1 ({ratio[0]:.1%} variance)", ylabel=f"PC2 ({ratio[1]:.1%} variance)",
             title="Training metadata density; no class labels used")
    fig.savefig(figures / "pca_map.png", dpi=170)
    plt.close(fig)
    return result


def evaluate_endterm(fitted, test, train, selected, output_dir):
    """Evaluate already frozen models; no fitting, threshold changes or reselection."""
    if selected not in fitted:
        raise ValueError("The CV-selected model is absent from fitted models.")
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    X, y = model_inputs(test), test.high_rated.to_numpy()
    columns = min(3, len(fitted))
    rows = int(np.ceil(len(fitted) / columns))
    fig_cm, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.8 * rows),
                                constrained_layout=True, squeeze=False)
    fig_roc, roc_axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    summaries, matrices, predictions, cases, profiles = [], {}, [], [], []
    error_masks, predicted_values = {}, {}
    for index, (name, model) in enumerate(fitted.items()):
        prediction = np.asarray(model.predict(X))
        score, score_type = continuous_score(model, X)
        matrix = confusion_matrix(y, prediction, labels=[0, 1])
        tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
        summaries.append({
            "model": name, "test_n": len(y), "f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "specificity": tn / (tn + fp) if tn + fp else 0.,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
        matrices[name] = {"labels": [0, 1], "rows": "true", "columns": "predicted",
                          "matrix": matrix.tolist()}
        axis = axes.flat[index]
        ConfusionMatrixDisplay(matrix, display_labels=["<4.5", ">=4.5"]).plot(ax=axis, colorbar=False)
        axis.set_title(name + (" [CV selected]" if name == selected else ""), fontsize=11)
        RocCurveDisplay.from_predictions(y, score, name=name, ax=roc_axis)
        errors = prediction != y
        error_masks[name] = errors
        predicted_values[name] = prediction
        private = test[["trackId", "high_rated", "averageUserRating", "userRatingCount"] + FEATURES].copy()
        private["model"] = name
        private["predicted"] = prediction
        private["score"] = score
        private["score_type"] = score_type
        private["error_type"] = np.where((y == 0) & (prediction == 1), "FP",
                                          np.where((y == 1) & (prediction == 0), "FN", "correct"))
        predictions.append(private)
        mistakes = private.loc[errors].copy()
        mistakes["boundary_distance"] = (mistakes.averageUserRating - 4.5).abs()
        for error_type in ["FP", "FN"]:
            examples = mistakes.loc[mistakes.error_type.eq(error_type)].sort_values(
                ["boundary_distance", "trackId"], kind="stable").head(2)
            for number, (_, example) in enumerate(examples.iterrows(), 1):
                observations = [_coarse_description(example, train)]
                if example.boundary_distance <= .1:
                    observations.append("true rating lies within 0.1 of the target boundary")
                missing = int(example[FEATURES].isna().sum())
                if missing:
                    observations.append(f"{missing} permitted feature(s) required imputation")
                if int(train.primary_genre.eq(example.primary_genre).sum()) < 20:
                    observations.append("genre has fewer than 20 training objects")
                cases.append({"model": name, "case": f"{error_type} example {number}",
                              "error_type": error_type, "actual_class": int(example.high_rated),
                              "predicted_class": int(example.predicted),
                              "observation": "; ".join(observations),
                              "interpretation": "Observed metadata association; the cause of this error is unknown."})
        # Minimum group size prevents publishing an exact feature profile of a tiny cohort.
        for genre, group in private.groupby("primary_genre", dropna=False):
            if len(group) < 20:
                continue
            profiles.append({"model": name, "genre": "Missing" if pd.isna(genre) else str(genre),
                             "test_n": len(group), "errors": int(group.error_type.ne("correct").sum()),
                             "fp": int(group.error_type.eq("FP").sum()),
                             "fn": int(group.error_type.eq("FN").sum()),
                             "error_rate": float(group.error_type.ne("correct").mean())})
    for axis in list(axes.flat)[len(fitted):]:
        axis.set_visible(False)
    roc_axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    roc_axis.set_title("Frozen holdout: continuous scores; no threshold tuning")
    roc_axis.legend(loc="lower right", fontsize=8)
    fig_cm.savefig(figures / "confusion_matrices.png", dpi=160)
    fig_roc.savefig(figures / "roc_curves.png", dpi=170)
    plt.close(fig_cm)
    plt.close(fig_roc)
    paired = []
    if "SVM" in error_masks:
        reference = error_masks["SVM"]
        for name, errors in error_masks.items():
            paired.append({"model": name, "reference": "SVM", "test_n": len(y),
                           "both_correct": int((~errors & ~reference).sum()),
                           "both_wrong": int((errors & reference).sum()),
                           "model_only_correct": int((~errors & reference).sum()),
                           "svm_only_correct": int((errors & ~reference).sum()),
                           "prediction_disagreements": int((predicted_values[name] != predicted_values["SVM"]).sum()),
                           "interpretation": "Paired descriptive counts; not a significance test."})
    metrics = pd.DataFrame(summaries)
    public_cases = pd.DataFrame(cases, columns=["model", "case", "error_type", "actual_class",
                                              "predicted_class", "observation", "interpretation"])
    write_csv(metrics, output_dir / "reports/test_metrics.csv")
    write_json(output_dir / "reports/confusion_matrices.json", matrices)
    write_csv(public_cases, output_dir / "reports/error_cases_anonymized.csv")
    write_csv(pd.DataFrame(paired, columns=["model", "reference", "test_n", "both_correct", "both_wrong",
                                          "model_only_correct", "svm_only_correct", "prediction_disagreements",
                                          "interpretation"]), output_dir / "reports/paired_errors.csv")
    write_csv(pd.DataFrame(profiles, columns=["model", "genre", "test_n", "errors", "fp", "fn", "error_rate"]),
              output_dir / "reports/error_profiles.csv")
    write_csv(pd.concat(predictions, ignore_index=True), output_dir / "private/test_predictions.csv")
    return metrics, public_cases, matrices
