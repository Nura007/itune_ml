"""Enforce an explicit public-output allowlist before staging research artifacts."""
import subprocess
from pathlib import Path

allowed_root = {"outputs/midterm.pptx", "outputs/midterm.pdf", "outputs/slides_contact.png",
                "notebooks/midterm.ipynb"}
allowed_reports = {
    "cv_metrics.csv", "cv_folds_metrics.csv", "model_selection.json", "test_metrics.csv",
    "confusion_matrices.json", "error_cases_anonymized.csv", "provenance.json",
    "cleaning_audit.json", "environment.txt", "results.md", "data_card.md",
    "pilot_audit.json", "main_audit.json"
}
allowed_figures = {"confusion_matrices.png", "roc_curves.png", "overview.png", "train_eda.png"}
for path in Path("outputs").rglob("*"):
    if not path.is_file():
        continue
    name = path.as_posix()
    if (name in allowed_root or
        (path.parent.as_posix() == "outputs/reports" and path.name in allowed_reports) or
        (path.parent.as_posix() == "outputs/figures" and path.name in allowed_figures)):
        subprocess.run(["git", "add", "--", name], check=True)
if Path("notebooks/midterm.ipynb").is_file():
    subprocess.run(["git", "add", "--", "notebooks/midterm.ipynb"], check=True)
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
for name in staged:
    if name.startswith(("data/raw/", "data/processed/", "data/splits/", "outputs/private/", "outputs/models/")):
        raise RuntimeError("Refusing to publish private data/model files: " + name)
print("Safe public artifacts staged:", len(staged))
