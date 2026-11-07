"""Endterm training: nested ensemble tuning and a fixed, checked MLP.

This module accepts training rows only. It never accepts or reads a test set.
Search scores are kept separate from outer-CV scores used for comparison.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .common import FEATURES, write_csv, write_json
from .features import model_inputs
from .pipelines import models, preprocessing
from .train import SCORING

MLP_EPOCHS = 120


def model_specs(seed=42):
    """Small prespecified spaces; all searches rank positive-class F1."""
    specs = {name: (pipeline, None) for name, pipeline in models(seed).items()}
    specs["Random Forest"] = (
        Pipeline([("preprocess", preprocessing()),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1))]),
        {"model__max_depth": [None, 12], "model__min_samples_leaf": [2, 10],
         "model__max_features": ["sqrt", 0.8]},
    )
    specs["Gradient Boosting"] = (
        Pipeline([("preprocess", preprocessing()),
                  ("model", GradientBoostingClassifier(random_state=seed))]),
        {"model__n_estimators": [75, 150], "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [1, 2]},
    )
    # The 90% variance rule is fixed in advance and PCA is fitted inside each fold.
    svm = models(seed)["SVM"]
    specs["SVM + PCA"] = (
        Pipeline([("preprocess", preprocessing()),
                  ("pca", PCA(n_components=0.90, svd_solver="full")),
                  ("model", clone(svm["model"]))]), None,
    )
    specs["MLP"] = (
        Pipeline([("preprocess", preprocessing()), ("model", mlp(seed))]), None,
    )
    return specs


def mlp(seed=42, epochs=MLP_EPOCHS):
    """Fixed training budget: learning curves diagnose, never tune on holdout."""
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                         alpha=0.001, batch_size=64, learning_rate_init=0.001,
                         max_iter=epochs, early_stopping=False, tol=0.0,
                         n_iter_no_change=epochs + 1, shuffle=True, random_state=seed)


def score_model(model, X, y):
    prediction = model.predict(X)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
    else:
        score = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return {"f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction))}


def fit_recording_warnings(estimator, X, y):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    return estimator, [str(item.message) for item in observed
                       if issubclass(item.category, ConvergenceWarning)]


def mlp_learning_check(train, output_dir, seed=42, epochs=MLP_EPOCHS):
    """Fit diagnostic preprocessing only on the internal learning subset.

    The validation subset is part of the saved training set. Curves are not used
    to change architecture, epochs, thresholds or any test-facing setting.
    """
    X, y = model_inputs(train), train.high_rated
    fit_indices, validation_indices = train_test_split(
        np.arange(len(train)), test_size=0.2, stratify=y, random_state=seed,
    )
    prep = preprocessing()
    X_fit = prep.fit_transform(X.iloc[fit_indices])
    X_validation = prep.transform(X.iloc[validation_indices])
    y_fit, y_validation = y.iloc[fit_indices], y.iloc[validation_indices]
    estimator = mlp(seed, epochs)
    rows = []
    for epoch in range(1, epochs + 1):
        estimator.partial_fit(X_fit, y_fit, classes=np.array([0, 1]))
        fit_probability = estimator.predict_proba(X_fit)
        validation_probability = estimator.predict_proba(X_validation)
        rows.append({"epoch": epoch,
                     "training_log_loss": float(log_loss(y_fit, fit_probability, labels=[0, 1])),
                     "validation_log_loss": float(log_loss(y_validation, validation_probability, labels=[0, 1])),
                     "training_f1": float(f1_score(y_fit, estimator.predict(X_fit), zero_division=0)),
                     "validation_f1": float(f1_score(y_validation,
                                                       estimator.predict(X_validation), zero_division=0))})
    curves = pd.DataFrame(rows)
    best = curves.loc[curves.validation_log_loss.idxmin()]
    last = curves.iloc[-1]
    window = min(10, epochs)
    earlier = curves.iloc[max(0, epochs - window)]
    diagnostics = {
        "architecture": [32, 16], "activation": "relu", "optimizer": "adam",
        "alpha": 0.001, "learning_rate_init": 0.001, "batch_size": 64,
        "fixed_epochs": epochs, "early_stopping": False,
        "diagnostic_training_n": len(fit_indices), "diagnostic_validation_n": len(validation_indices),
        "validation_source": "stratified 20% of the saved training partition",
        "preprocessing_fit_scope": "diagnostic training subset only",
        "used_to_change_settings": False, "test_used": False,
        "best_observed_validation_epoch": int(best.epoch),
        "best_observed_validation_log_loss": float(best.validation_log_loss),
        "final_training_log_loss": float(last.training_log_loss),
        "final_validation_log_loss": float(last.validation_log_loss),
        "final_validation_f1": float(last.validation_f1),
        "validation_loss_change_last_window": float(last.validation_log_loss - earlier.validation_log_loss),
        "window_epochs": window,
        "validation_loss_above_observed_minimum": float(last.validation_log_loss - best.validation_log_loss),
        "interpretation_rule": (
            "Lower training loss with increasing validation loss suggests overfitting. "
            "A flat validation loss suggests a plateau; fixed-budget termination alone is not convergence. "
            "The observed best epoch is descriptive and is not selected for final fitting."
        ),
    }
    output_dir = Path(output_dir)
    write_csv(curves, output_dir / "reports/mlp_learning_curve.csv")
    write_json(output_dir / "reports/mlp_diagnostics.json", diagnostics)
    # IDs remain private. This permits auditing the diagnostic preprocessing boundary.
    assignments = pd.DataFrame({
        "trackId": train.trackId,
        "diagnostic_split": ["fit" if i in set(fit_indices) else "validation" for i in range(len(train))],
    })
    write_csv(assignments, output_dir / "private/mlp_diagnostic_split.csv")
    return curves, diagnostics


def _search(pipeline, grid, X, y, seed, inner_folds):
    folds = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(clone(pipeline), grid, scoring=SCORING["f1"], refit=True,
                          cv=folds, n_jobs=1, return_train_score=True, error_score="raise")
    search.fit(X, y)
    return search


def _search_rows(search, model_name, scope):
    results = search.cv_results_
    return [{"model": model_name, "scope": scope,
             "parameters": json.dumps(parameters, sort_keys=True),
             "mean_validation_f1": float(results["mean_test_score"][i]),
             # GridSearchCV uses population SD; store sample SD explicitly instead.
             "validation_f1_sample_sd": float(np.std([
                 results[f"split{k}_test_score"][i] for k in range(search.n_splits_)], ddof=1)),
             "mean_training_f1": float(results["mean_train_score"][i]),
             "rank": int(results["rank_test_score"][i]),
             "selected": parameters == search.best_params_}
            for i, parameters in enumerate(results["params"])]


def train_endterm(train, output_dir, config):
    """Return fitted pipelines, honest outer-CV summaries and a CV-selected name."""
    output_dir = Path(output_dir)
    X, y = model_inputs(train), train.high_rated
    seed = config["random_state"]
    outer_count = config["cv_folds"]
    inner_count = 3
    folds = list(StratifiedKFold(n_splits=outer_count, shuffle=True,
                                random_state=seed).split(X, y))
    fold_assignments = [{"trackId": int(train.iloc[i].trackId), "validation_fold": number}
                        for number, (_, val) in enumerate(folds, 1) for i in val]
    write_csv(pd.DataFrame(fold_assignments), output_dir / "private/cv_folds.csv")
    mlp_learning_check(train, output_dir, seed)
    specs = model_specs(seed)
    summaries, details, searches, fitted, selected_parameters, convergence = [], [], [], {}, {}, {}
    for name, (pipeline, grid) in specs.items():
        print(f"Endterm {name}: {outer_count} outer folds" +
              (f", {inner_count}-fold nested tuning" if grid else ", fixed settings"), flush=True)
        model_rows, train_f1s = [], []
        convergence[name] = []
        for number, (fit_indices, val_indices) in enumerate(folds, 1):
            X_fit, y_fit = X.iloc[fit_indices], y.iloc[fit_indices]
            if grid:
                search = _search(pipeline, grid, X_fit, y_fit, seed, inner_count)
                estimator = search.best_estimator_
                searches.extend(_search_rows(search, name, f"outer_fold_{number}"))
            else:
                estimator, messages = fit_recording_warnings(clone(pipeline), X_fit, y_fit)
                convergence[name].extend({"scope": f"outer_fold_{number}", "message": message}
                                         for message in messages)
            fold_scores = score_model(estimator, X.iloc[val_indices], y.iloc[val_indices])
            training_f1 = float(f1_score(y_fit, estimator.predict(X_fit), zero_division=0))
            model_rows.append(fold_scores)
            train_f1s.append(training_f1)
            details.append({"model": name, "fold": number, **fold_scores, "training_f1": training_f1})
            print(f"  fold {number}: F1={fold_scores['f1']:.4f}", flush=True)
        scores = pd.DataFrame(model_rows)
        row = {"model": name, "cv_scheme": "nested" if grid else "fixed settings"}
        for metric in SCORING:
            row[f"cv_{metric}_mean"] = float(scores[metric].mean())
            row[f"cv_{metric}_std"] = float(scores[metric].std(ddof=1))
        row["train_f1_mean"] = float(np.mean(train_f1s))
        row["f1_generalization_gap"] = row["train_f1_mean"] - row["cv_f1_mean"]
        summaries.append(row)
        if grid:
            final_search = _search(pipeline, grid, X, y, seed, inner_count)
            fitted[name] = final_search.best_estimator_
            selected_parameters[name] = final_search.best_params_
            searches.extend(_search_rows(final_search, name, "full_training_refit"))
        else:
            fitted[name], messages = fit_recording_warnings(clone(pipeline), X, y)
            selected_parameters[name] = {"fixed_estimator": str(fitted[name]["model"])}
            if "pca" in fitted[name].named_steps:
                selected_parameters[name]["pca_variance_target"] = 0.9
                selected_parameters[name]["fitted_pca_components"] = int(fitted[name]["pca"].n_components_)
            convergence[name].extend({"scope": "full_training_refit", "message": message}
                                     for message in messages)
        path = output_dir / "models" / (name.lower().replace(" ", "_").replace("+", "plus") + ".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[name], path)
        print(f"{name}: CV F1={row['cv_f1_mean']:.4f} +/- {row['cv_f1_std']:.4f}", flush=True)
    summary = pd.DataFrame(summaries)
    ranked = summary.sort_values(["cv_f1_mean", "model"], ascending=[False, True], kind="stable")
    best = str(ranked.iloc[0].model)
    earlier = ranked.loc[ranked.model.ne("MLP")]
    midterm = ranked.loc[ranked.model.isin(models(seed))]
    write_csv(summary, output_dir / "reports/cv_metrics.csv")
    write_csv(pd.DataFrame(details), output_dir / "reports/cv_folds_metrics.csv")
    write_csv(pd.DataFrame(searches), output_dir / "reports/tuning_results.csv")
    write_json(output_dir / "reports/model_selection.json", {
        "selected_model": best, "best_earlier_model": str(earlier.iloc[0].model),
        "best_repeated_midterm_model": str(midterm.iloc[0].model),
        "original_midterm_selected_model": "SVM", "criterion": "maximum mean outer CV F1 for class 1",
        "tie_break": "model name ascending", "cv_std": "sample standard deviation, ddof=1",
        "outer_folds": outer_count, "inner_folds": inner_count,
        "hyperparameter_tuning": "nested training CV for ensembles; full-training CV for final refit",
        "inner_tie_break": "first candidate in deterministic sklearn ParameterGrid order",
        "search_spaces": {name: grid for name, (_, grid) in specs.items() if grid},
        "selected_parameters": selected_parameters,
        "input_features": FEATURES, "test_used_for_selection": False,
        "threshold_tuning": False, "mlp_fixed_epochs": MLP_EPOCHS,
        "convergence_warnings": convergence,
        "limitations": ["Winning outer-CV mean can be optimistic after model-family selection.",
                        "Nested ensemble CV and fixed-model CV share exactly the same outer folds.",
                        "Final refit search scores are not independent evaluation scores.",
                        "MLP fixed-budget termination is not proof of convergence."],
    })
    return fitted, summary, best
