"""Validate public aggregates, with read-only --check-only or legacy CI staging.

This structural privacy check complements human review of prose, figures and
anonymized cases. Default staging preserves the existing CI interface.
"""
import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = {
    "outputs/midterm.pptx", "outputs/midterm.pdf", "outputs/slides_contact.png",
    "notebooks/midterm.ipynb", "notebooks/endterm.ipynb",
    "outputs/endterm/Endterm_Report.pdf", "outputs/endterm/Endterm_Report.docx",
    "outputs/endterm/Endterm_Presentation.pptx", "outputs/endterm/Endterm_Presentation.pdf",
    "outputs/endterm/slides_contact.png", "outputs/endterm/speaker_notes.md",
    "outputs/endterm/defense_questions.md", "outputs/endterm/rubric_checklist.md",
}
REPORT_NAMES = {
    "cv_metrics.csv", "cv_folds_metrics.csv", "model_selection.json", "test_metrics.csv",
    "confusion_matrices.json", "error_cases_anonymized.csv", "provenance.json",
    "cleaning_audit.json", "environment.txt", "results.md", "data_card.md",
    "pilot_audit.json", "main_audit.json",
}
ENDTERM_REPORT_NAMES = REPORT_NAMES | {
    "tuning_results.csv", "mlp_learning_curve.csv", "mlp_diagnostics.json",
    "paired_errors.csv", "error_profiles.csv", "unsupervised.json",
    "pca_variance.csv", "pca_loadings.csv", "pca_profiles.csv", "comparison.csv",
    "continuity.json", "source_hashes.json",
}
FIGURE_NAMES = {"confusion_matrices.png", "roc_curves.png", "overview.png", "train_eda.png"}
ENDTERM_FIGURE_NAMES = FIGURE_NAMES | {
    "pca_variance.png", "pca_map.png", "mlp_learning.png", "comparison.png",
}
PRIVATE_PREFIXES = (
    "data/raw/", "data/processed/", "data/splits/", "data/private/", "work/", "private_run/",
    "outputs/private/", "outputs/models/", "outputs/smoke/", ".venv/", ".git/",
)
METRICS = {"f1", "roc_auc", "precision", "recall", "balanced_accuracy", "accuracy"}
CSV_COLUMNS = {
    "cv_metrics.csv": {"model", "cv_scheme", "train_f1_mean", "f1_generalization_gap"}
    | {f"cv_{m}_{stat}" for m in METRICS for stat in ("mean", "std")},
    "cv_folds_metrics.csv": {"model", "fold", "training_f1"} | METRICS,
    "test_metrics.csv": {"model", "test_n", "specificity", "tn", "fp", "fn", "tp"} | METRICS,
    "error_cases_anonymized.csv": {
        "model", "case", "error_type", "actual_class", "predicted_class", "observation", "interpretation"
    },
    "tuning_results.csv": {
        "model", "scope", "parameters", "mean_validation_f1", "validation_f1_sample_sd",
        "mean_training_f1", "rank", "selected",
    },
    "mlp_learning_curve.csv": {"epoch", "training_log_loss", "validation_log_loss",
                               "training_f1", "validation_f1"},
    "paired_errors.csv": {"model", "reference", "test_n", "both_correct", "both_wrong",
                          "model_only_correct", "svm_only_correct", "prediction_disagreements", "interpretation"},
    "error_profiles.csv": {"model", "genre", "test_n", "errors", "fp", "fn", "error_rate"},
    "pca_variance.csv": {"component", "explained_variance_ratio", "cumulative_variance"},
    "pca_loadings.csv": {"component", "feature", "loading"},
    "pca_profiles.csv": {"pc1_stratum", "n", "median_price_usd", "median_size_mb", "median_language_count",
                         "median_release_year", "median_update_recency_days", "most_common_genre"},
    "comparison.csv": {"model", "cv_scheme", "train_f1_mean", "f1_generalization_gap", "test_n",
                       "specificity", "tn", "fp", "fn", "tp"}
    | {f"cv_{m}_{stat}" for m in METRICS for stat in ("mean", "std")} | METRICS,
}
FORBIDDEN_COLUMNS = {
    "trackid", "appid", "id", "trackname", "name", "description", "descriptions",
    "averageuserrating", "userratingcount", "rating", "ratingcount", "score",
    "predicted", "prediction", "probability", "highrated", "artistid", "sellername",
}


def normalized_name(name):
    value = str(name).replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Unsafe publication path: {name}")
    return path.as_posix()


def is_public_artifact(name):
    name = normalized_name(name)
    path = PurePosixPath(name)
    return (
        name in PUBLIC_ROOT
        or (path.parent.as_posix() == "outputs/reports" and path.name in REPORT_NAMES)
        or (path.parent.as_posix() == "outputs/figures" and path.name in FIGURE_NAMES)
        or (path.parent.as_posix() == "outputs/endterm/reports" and path.name in ENDTERM_REPORT_NAMES)
        or (path.parent.as_posix() == "outputs/endterm/figures" and path.name in ENDTERM_FIGURE_NAMES)
    )


def validate_staged_path(name):
    name = normalized_name(name)
    lower = name.lower()
    if (lower.startswith(PRIVATE_PREFIXES) or lower.endswith((".zip", ".joblib", ".pkl", ".pickle"))
            or lower == ".env" or "/.env" in lower):
        raise ValueError(f"Refusing to publish private data, archive or model: {name}")
    if name.startswith(("outputs/", "notebooks/")) and not is_public_artifact(name):
        raise ValueError(f"Artifact is not on the public allowlist: {name}")


def validate_artifact_bytes(name, data):
    if not is_public_artifact(name):
        raise ValueError(f"Artifact is not on the public allowlist: {name}")
    path = PurePosixPath(normalized_name(name))
    if path.suffix == ".csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        fields = reader.fieldnames or []
        normalized = {re.sub(r"[^a-z0-9]", "", f.lower()) for f in fields}
        if normalized & FORBIDDEN_COLUMNS:
            raise ValueError(f"Private app-level columns found in public artifact: {name}")
        allowed = CSV_COLUMNS.get(path.name, set())
        if not fields or len(fields) != len(set(fields)) or not set(fields) <= allowed:
            raise ValueError(f"Unexpected aggregate CSV columns in {name}: {fields}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed aggregate CSV rows: {name}")
        if path.name == "error_cases_anonymized.csv":
            if "model" in fields:
                counts = {}
                for row in rows:
                    counts[row["model"]] = counts.get(row["model"], 0) + 1
                if (len(rows) > 32 or any(n > 4 for n in counts.values())
                        or any(not re.fullmatch(r"(?:FP|FN) example [12]", row["case"]) for row in rows)):
                    raise ValueError("Endterm cases require at most four anonymized examples per model.")
            elif len(rows) > 6 or any(not re.fullmatch(r"Case [1-6]", row.get("case", "")) for row in rows):
                raise ValueError("Midterm cases require at most six non-identifying Case labels.")
        if path.name == "error_profiles.csv" and any(int(row["test_n"]) < 20 for row in rows):
            raise ValueError("Public error profiles require at least 20 apps in each group.")
    elif path.suffix in {".json", ".ipynb"}:
        parsed = json.loads(data.decode("utf-8"))
        if path.suffix == ".ipynb":
            if not isinstance(parsed, dict) or "cells" not in parsed:
                raise ValueError(f"Invalid public notebook: {name}")
            outputs = json.dumps([c.get("outputs", []) for c in parsed["cells"]])
            # Source may explain private inputs; emitted records must remain private.
            for token in ("trackId", "averageUserRatingForCurrentVersion", "userRatingCountForCurrentVersion"):
                if re.search(r"<th[^>]*>\s*" + token + r"\s*</th>", outputs, re.IGNORECASE):
                    raise ValueError(f"Possible private records in notebook output ({token}): {name}")


def public_artifacts(root):
    root = Path(root).resolve()
    files = []
    for folder in (root / "outputs", root / "notebooks"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            if is_public_artifact(name):
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"Refusing linked public artifact: {name}")
                validate_artifact_bytes(name, path.read_bytes())
                files.append(path)
            elif name.startswith("outputs/endterm/"):
                raise ValueError(f"Unexpected file in public Endterm directory: {name}")
    return sorted(files)


def check_publication(root=ROOT, check_only=False):
    root = Path(root).resolve()
    candidates = public_artifacts(root)
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=root
    ).decode().split("\0")
    for name in filter(None, staged):
        validate_staged_path(name)
        if is_public_artifact(name):
            staged_data = subprocess.check_output(["git", "show", f":{name}"], cwd=root)
            validate_artifact_bytes(name, staged_data)
    if not check_only and candidates:
        subprocess.run(["git", "add", "--"] + [p.relative_to(root).as_posix() for p in candidates],
                       cwd=root, check=True)
    print(f"Validated {len(candidates)} public artifacts; "
          + ("index unchanged (--check-only)." if check_only else "allowlisted artifacts staged."))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate without changing the Git index.")
    args = parser.parse_args()
    check_publication(check_only=args.check_only)


if __name__ == "__main__":
    main()
