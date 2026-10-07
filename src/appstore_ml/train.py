from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .common import write_csv, write_json
from .features import model_inputs
from .pipelines import models

SCORING = {
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "balanced_accuracy": "balanced_accuracy"
}


def train_models(train, output_dir, config):
    output_dir = Path(output_dir)
    X, y = model_inputs(train), train.high_rated
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"])
    folds = list(cv.split(X, y))
    fold_assignments = []
    for number, (_, val) in enumerate(folds, 1):
        fold_assignments.extend({"trackId": int(train.iloc[i].trackId), "validation_fold": number} for i in val)
    write_csv(pd.DataFrame(fold_assignments), output_dir / "private/cv_folds.csv")
    summaries, details, fitted = [], [], {}
    for name, pipeline in models(config["random_state"]).items():
        scores = cross_validate(pipeline, X, y, cv=folds, scoring=SCORING,
                                return_train_score=True, n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in SCORING:
            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std(ddof=1))
        row["train_f1_mean"] = float(scores["train_f1"].mean())
        row["f1_generalization_gap"] = row["train_f1_mean"] - row["cv_f1_mean"]
        summaries.append(row)
        for fold in range(config["cv_folds"]):
            details.append({"model": name, "fold": fold + 1,
                            **{metric: float(scores[f"test_{metric}"][fold]) for metric in SCORING}})
        fitted[name] = clone(pipeline).fit(X, y)
        model_path = output_dir / "models" / (name.lower().replace(" ", "_") + ".joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[name], model_path)
        print(f"{name}: CV F1={row['cv_f1_mean']:.4f} +/- {row['cv_f1_std']:.4f}", flush=True)
    summary = pd.DataFrame(summaries)
    # Prespecified model choice, including the baseline as an honest possible winner.
    ranked = summary.sort_values(["cv_f1_mean", "model"], ascending=[False, True], kind="stable")
    best = str(ranked.iloc[0]["model"])
    write_csv(summary, output_dir / "reports/cv_metrics.csv")
    write_csv(pd.DataFrame(details), output_dir / "reports/cv_folds_metrics.csv")
    write_json(output_dir / "reports/model_selection.json",
               {"selected_model": best, "criterion": "maximum mean training CV F1 for class 1",
                "tie_break": "model name ascending", "cv_std": "sample standard deviation, ddof=1",
                "hyperparameters": {name: str(pipe.named_steps["model"]) for name, pipe in fitted.items()},
                "test_used_for_selection": False, "hyperparameter_tuning": "none; fixed prespecified settings"})
    return fitted, summary, best
