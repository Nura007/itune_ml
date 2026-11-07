# Run the Endterm work

The main command never collects new data. Use the private submission ZIP, which includes
the frozen data under `work/endterm-2026-11-08/`, or restore that folder from a private backup.
The public repository intentionally cannot reproduce the empirical run by itself.

## Installation

Use Python 3.12 (the delivered run uses 3.12.10 and scikit-learn 1.6.1).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,endterm]"
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest
```

## Frozen-snapshot reproduction

```powershell
.\.venv\Scripts\python.exe -m appstore_ml.endterm --new-experiment
```

`--new-experiment` acknowledges the documented difference from the original Midterm snapshot.
It does not recollect data or change saved split rows. Changed snapshot bytes or split checksums
cause an error. All eight methods, tuning searches, PCA, MLP checks and held-out metrics rerun.
Results are in `outputs/endterm/`; private models/predictions stay in the run folder.
Alternatively select this virtual environment as the kernel of `notebooks/endterm.ipynb`
and use Restart and Run All. Runtime depends on CPU and the fixed search grids.

To regenerate report/figures/slides from already computed aggregate evidence:

```powershell
.\.venv\Scripts\python.exe -m appstore_ml.endterm --report-only
.\.venv\Scripts\python.exe scripts/check_publication.py --check-only
```

## New collection, only for a separately reported experiment

The recorded instructor approval is in `config/source_approval.json`. The collector enforces
at least 3.2 seconds between request starts. Do not run collectors concurrently. Never point
new collection at the frozen submitted snapshot. Use another run directory, run and inspect
the pilot, then run main collection. Do not silently alter sample eligibility or the split.
Source restrictions are documented in the report; course approval is not Apple permission.

## Private submission

```powershell
.\.venv\Scripts\python.exe scripts/package_endterm.py --run-dir work/endterm-2026-11-08
```

The ZIP includes raw data and fitted models and must remain private. Both students submit
the same reviewed archive to Moodle, with the filename pattern
`Endterm_Surname_Name_Group.zip` required by the rubric. Replace the generic archive name
with real identifying details. Submit before 23:59 the day before the actual defense.
No defense date or student identities have been supplied. Do not upload the ZIP to GitHub.
