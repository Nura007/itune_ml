"""Privacy/retention tests use explicitly SYNTHETIC fixtures, never research data."""
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from appstore_ml.common import sha256, write_json
from appstore_ml.prepare import prepare_records
from appstore_ml.split import fixed_split


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publication = load_script("check_publication")
packaging = load_script("package_endterm")


@pytest.mark.parametrize("name", [
    "data/raw/response.json", "data/private/apps.csv", "work/run/outputs/reports/test_metrics.csv",
    "outputs/private/Endterm_Submission.zip", "outputs/models/svm.joblib", "submission.zip",
    "outputs/endterm/predictions.csv", "../private.csv", "C:/secret.csv",
    "private_run/data/raw/response.json",
])
def test_private_paths_never_publish(name):
    with pytest.raises(ValueError):
        publication.validate_staged_path(name)


@pytest.mark.parametrize("column", ["trackId", "description", "averageUserRating", "userRatingCount",
                                   "score", "predicted", "user_rating_count"])
def test_safe_filename_cannot_hide_private_columns(column):
    data = f"model,f1,{column}\nSYNTHETIC,0.5,123\n".encode()
    with pytest.raises(ValueError, match="Private"):
        publication.validate_artifact_bytes("outputs/endterm/reports/test_metrics.csv", data)


def test_unrecognized_csv_columns_fail_closed():
    with pytest.raises(ValueError, match="Unexpected"):
        publication.validate_artifact_bytes("outputs/endterm/reports/test_metrics.csv", b"model,mystery\na,b\n")


def test_check_only_does_not_stage(tmp_path, monkeypatch):
    public = tmp_path / "outputs/endterm/reports/test_metrics.csv"
    public.parent.mkdir(parents=True)
    public.write_text("model,test_n,f1\nSYNTHETIC,120,0.5\n", encoding="utf-8")
    monkeypatch.setattr(publication.subprocess, "check_output", lambda *a, **kw: b"")

    def refuse_mutation(*args, **kwargs):
        pytest.fail("--check-only attempted to mutate Git")

    monkeypatch.setattr(publication.subprocess, "run", refuse_mutation)
    assert publication.check_publication(tmp_path, check_only=True) == [public]


def test_validate_index_bytes_not_only_worktree(tmp_path, monkeypatch):
    name = "outputs/endterm/reports/test_metrics.csv"
    public = tmp_path / name
    public.parent.mkdir(parents=True)
    public.write_text("model,f1\nSYNTHETIC,0.5\n", encoding="utf-8")

    def staged_data(args, **kwargs):
        return name.encode() + b"\0" if args[1] == "diff" else b"model,trackId\nSYNTHETIC,1\n"

    monkeypatch.setattr(publication.subprocess, "check_output", staged_data)
    with pytest.raises(ValueError, match="Private"):
        publication.check_publication(tmp_path, check_only=True)


def test_unlisted_endterm_file_is_rejected(tmp_path):
    path = tmp_path / "outputs/endterm/app_records.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        publication.public_artifacts(tmp_path)


@pytest.fixture
def synthetic_archive_run(tmp_path, records, config):
    """Emulate real-run metadata solely to exercise integrity checks on fake records."""
    run = tmp_path / "work/synthetic-fixture"
    raw = run / "data/raw"
    raw.mkdir(parents=True)
    response = raw / "synthetic.response.json"
    write_json(response, {"fixture": "SYNTHETIC - not an empirical result", "results": records})
    write_json(raw / "synthetic.meta.json", {
        "sha256": sha256(response), "collected_at": records[0]["_collected_at"],
        "params": {"term": "SYNTHETIC FIXTURE"},
    })
    frame, audit = prepare_records(records, config)
    processed = run / "data/processed"
    processed.mkdir()
    frame.to_csv(processed / "apps.csv", index=False)
    write_json(processed / "cleaning_audit.json", audit)
    train, test, spec = fixed_split(frame, run / "data/splits", config)
    checksums = {response.name: sha256(response)}
    write_json(run / "data/snapshot.json", {"dataset_sha256": spec["dataset_sha256"], "raw_checksums": checksums})
    provenance = {
        **spec, "data_kind": "real_api", "fixture": "SYNTHETIC metadata emulation for unit tests only",
        "config": config, "train_n": len(train), "test_n": len(test), "source_response_checksums": checksums,
    }
    write_json(run / "outputs/reports/provenance.json", provenance)
    write_json(run / "outputs/reports/model_selection.json", {"selected_model": "SVM + PCA"})
    write_json(run / "outputs/reports/source_hashes.json", {"scripts/SYNTHETIC_fixture.py": "0" * 64})
    pd.DataFrame({"model": ["SVM + PCA"], "test_n": [len(test)]}).to_csv(
        run / "outputs/reports/test_metrics.csv", index=False)
    for name in ("cv_folds.csv", "test_predictions.csv"):
        path = run / "outputs/private" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture\nSYNTHETIC\n")
    model = run / "outputs/models/svm_plus_pca.joblib"
    model.parent.mkdir()
    model.write_bytes(b"SYNTHETIC MODEL FILE - never deserialized")
    return run, provenance


def test_archive_checks_all_retained_bytes(synthetic_archive_run):
    run, provenance = synthetic_archive_run
    assert packaging.validate_run(run)["dataset_sha256"] == provenance["dataset_sha256"]
    response = run / "data/raw/synthetic.response.json"
    response.write_text(response.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        packaging.validate_run(run)


def test_archive_rejects_changed_manifest(synthetic_archive_run):
    run, _ = synthetic_archive_run
    manifest = run / "data/splits/split.csv"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest checksum"):
        packaging.validate_run(run)


def test_archive_rejects_missing_model(synthetic_archive_run):
    run, _ = synthetic_archive_run
    (run / "outputs/models/svm_plus_pca.joblib").unlink()
    with pytest.raises(ValueError, match="No fitted model"):
        packaging.validate_run(run)


def test_archive_rejects_nested_raw_folders(synthetic_archive_run):
    run, _ = synthetic_archive_run
    nested = run / "data/raw/nested"
    nested.mkdir()
    (nested / "extra.response.json").write_text('{"fixture": "SYNTHETIC"}')
    with pytest.raises(ValueError, match="Nested raw folders"):
        packaging.validate_run(run)


def test_package_contains_private_manifest_and_snapshot(tmp_path, synthetic_archive_run):
    run, provenance = synthetic_archive_run
    for relative in ("README.md", "AGENTS.md", "pyproject.toml", ".gitignore",
                     "outputs/endterm/Endterm_Report.pdf", "outputs/endterm/Endterm_Presentation.pdf",
                     "outputs/endterm/rubric_checklist.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC FIXTURE - not a real submission")
    write_json(tmp_path / "notebooks/endterm.ipynb", {"cells": []})
    write_json(tmp_path / "outputs/endterm/reports/provenance.json", provenance)
    output = packaging.package_endterm(run, root=tmp_path)
    assert output.is_relative_to(tmp_path / "outputs/private")
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        run_prefix = "work/" + provenance["config"]["endterm"]["experiment_id"]
        assert run_prefix + "/data/snapshot.json" in manifest["files"]
        assert run_prefix + "/outputs/models/svm_plus_pca.joblib" in manifest["files"]
        assert archive.read(run_prefix + "/.gitignore") == b"*\n"
        assert manifest["archive_run_path"] == run_prefix
        final_hashes = json.loads(archive.read("final_source_hashes.json"))
        assert final_hashes["files"]["README.md"] == sha256(tmp_path / "README.md")
        assert set(archive.namelist()) == set(manifest["files"]) | {"MANIFEST.json"}
        for name, entry in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        packaging.package_endterm(run, root=tmp_path)


def test_private_zip_cannot_be_written_to_public_directory(tmp_path):
    with pytest.raises(ValueError, match="ignored outputs/private"):
        packaging.package_endterm(tmp_path / "work/run", tmp_path / "outputs/endterm/submission.zip", tmp_path)
