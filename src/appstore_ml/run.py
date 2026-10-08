import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd

from .common import ROOT, settings, sha256, write_csv, write_json
from .evaluate import evaluate_models, make_overview
from .prepare import prepare
from .report import presentation, report
from .split import fixed_split
from .train import train_models


def create_notebook(output_dir, root):
    """Public notebook contains only aggregates; it never displays app-level raw data."""
    relative = Path(output_dir).relative_to(root).as_posix()
    cells = [
        nbformat.v4.new_markdown_cell("# App Store ML — Midterm\n"
            "Research question: can seven metadata features identify rating ≥4.5 among existing US apps "
            "with ≥50 ratings? This notebook reads generated artifacts. Run the pipeline first. "
            "Neural networks, NLP and Streamlit are reserved for Final."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\n"
            "from IPython.display import display, Markdown, Image\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            f"OUT = ROOT / '{relative}'\n"
            "assert (OUT / 'reports/provenance.json').exists(), 'Run python -m appstore_ml.run first'\n"
            "provenance = json.loads((OUT / 'reports/provenance.json').read_text())\n"
            "print('DATA STATUS:', provenance['data_kind'])"),
        nbformat.v4.new_markdown_cell("## Source, cleaning and sampling"),
        nbformat.v4.new_code_cell("display(Markdown((OUT / 'reports/data_card.md').read_text()))"),
        nbformat.v4.new_markdown_cell("## Training-only exploratory analysis"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT / 'figures/train_eda.png')))"),
        nbformat.v4.new_markdown_cell("## Models and validation\n"
            "Dummy, Decision Tree, KNN and SVM use fixed prespecified settings. A permanent stratified "
            "80/20 split uses seed 42. Five-fold CV is performed only inside train. Numeric median "
            "imputation/scaling and categorical imputation/one-hot encoding are Pipeline steps. "
            "Models never receive user-rating fields, descriptions, names or IDs."),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'reports/cv_metrics.csv'))\n"
                                 "display(pd.read_csv(OUT / 'reports/test_metrics.csv'))"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT / 'figures/confusion_matrices.png')))\n"
                                 "display(Image(filename=str(OUT / 'figures/roc_curves.png')))"),
        nbformat.v4.new_markdown_cell("## Error analysis and conclusions"),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'reports/error_cases_anonymized.csv'))\n"
                                 "display(Markdown((OUT / 'reports/results.md').read_text()))"),
        nbformat.v4.new_markdown_cell("## Reproducibility\n"
            "Code, configuration, split fingerprint and package versions are recorded. Private raw "
            "JSON/metadata, cleaned data, split IDs and model files stay outside version control. "
            "The seven-feature experiment does not establish pre-launch success prediction.")
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = Path(root) / "notebooks/midterm.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def experiment(root=ROOT, raw_dir=None, config=None, data_kind="real_api"):
    root = Path(root)
    config = config or settings()
    out = root / "outputs"
    raw_dir = Path(raw_dir or root / "data/raw")
    frame, audit = prepare(raw_dir, root / "data/processed", config)
    if len(frame) < config["min_samples"]:
        raise ValueError(f"Only {len(frame)} eligible apps; minimum is {config['min_samples']}.")
    train, test, split_spec = fixed_split(frame, root / "data/splits", config)
    if set(train.high_rated) != {0, 1} or set(test.high_rated) != {0, 1}:
        raise ValueError("Both classes must occur in train and test.")
    fitted, cv, selected = train_models(train, out, config)
    metrics, cases, matrices = evaluate_models(fitted, test, train, selected, out)
    make_overview(train, test, cv, metrics, out)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-in-git"
    provenance = {
        "data_kind": data_kind, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "sklearn_version": importlib.metadata.version("scikit-learn"),
        "git_commit": commit, **split_spec, "config": config,
        "train_n": len(train), "test_n": len(test),
        "train_class_counts": {str(k): int(v) for k, v in train.high_rated.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test.high_rated.value_counts().items()},
        "source_response_checksums": {p.name: sha256(p) for p in sorted(raw_dir.glob("*.response.json"))}
    }
    write_json(out / "reports/provenance.json", provenance)
    write_json(out / "reports/cleaning_audit.json", audit)
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
                      if dist.metadata["Name"])
    (out / "reports/environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    statement = report(audit, cv, metrics, cases, selected, provenance, out)
    presentation(audit, cv, metrics, cases, selected, statement, out, data_kind)
    create_notebook(out, root)
    print(statement, flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT,
                        help="Separate directory containing data/raw; preserves earlier holdout manifests.")
    args = parser.parse_args()
    experiment(args.run_dir)


if __name__ == "__main__":
    main()
