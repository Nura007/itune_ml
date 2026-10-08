"""Cloud research publishes aggregates only. Raw data exists only during the runner lifetime."""
import json
import sys
from pathlib import Path

from appstore_ml.collect import run
from appstore_ml.common import ROOT, settings
from appstore_ml.run import experiment

config = settings()
request = json.loads((ROOT / "config/collection_run.json").read_text())
stage = request["stage"]
if stage == "report":
    from refresh_report import refresh
    refresh()
    sys.exit(0)
if stage not in {"pilot", "full"}:
    raise ValueError("stage must be pilot, full or report")
run("pilot", ROOT / "data/raw", ROOT / "outputs/reports", config,
    ROOT / "config/source_approval.json", ROOT / "config/search_terms.txt")
audit = json.loads((ROOT / "outputs/reports/pilot_audit.json").read_text())
print("PILOT AUDIT (aggregate only):")
print(json.dumps(audit, indent=2))
if stage == "full":
    run("main", ROOT / "data/raw", ROOT / "outputs/reports", config,
        ROOT / "config/source_approval.json", ROOT / "config/search_terms.txt")
    experiment()
