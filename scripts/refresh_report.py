"""Rebuild prose/slides from frozen aggregate outputs. Never recollect or retrain."""
import json
import subprocess

import pandas as pd

from appstore_ml.common import ROOT, write_json
from appstore_ml.report import presentation, report
from appstore_ml.run import create_notebook


def refresh():
    out = ROOT / "outputs"
    reports = out / "reports"
    def read(name):
        return json.loads((reports / name).read_text(encoding="utf-8"))
    audit = read("cleaning_audit.json")
    provenance = read("provenance.json")
    provenance["report_generation_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # git_commit remains the original collection/training revision.
    selected = read("model_selection.json")["selected_model"]
    cv = pd.read_csv(reports / "cv_metrics.csv")
    metrics = pd.read_csv(reports / "test_metrics.csv")
    cases = pd.read_csv(reports / "error_cases_anonymized.csv")
    statement = report(audit, cv, metrics, cases, selected, provenance, out)
    presentation(audit, cv, metrics, cases, selected, statement, out, provenance["data_kind"])
    create_notebook(out, ROOT)
    write_json(reports / "provenance.json", provenance)
    print("Regenerated report and slides from frozen real-data aggregate outputs.")
    print(statement)


if __name__ == "__main__":
    refresh()
