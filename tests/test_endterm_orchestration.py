"""Synthetic control-flow tests; no generated fixture is research evidence."""
import copy
import json

import pandas as pd
import pytest

from appstore_ml import endterm
from appstore_ml.common import sha256, write_json
from appstore_ml.prepare import prepare_records
from appstore_ml.split import dataset_fingerprint, fixed_split


def _synthetic_raw(run_dir, records):
    raw = run_dir / "data/raw"
    raw.mkdir(parents=True)
    response = raw / "synthetic.response.json"
    write_json(response, {"resultCount": len(records), "results": records})
    write_json(raw / "synthetic.meta.json", {
        "sha256": sha256(response), "collected_at": records[0]["_collected_at"],
        "params": {"term": "SYNTHETIC FIXTURE - NOT API RESEARCH"},
    })
    return response


def test_new_snapshot_requires_explicit_disclosure(records, config):
    frame, _ = prepare_records(records, config)
    with pytest.raises(ValueError, match="--new-experiment"):
        endterm.verify_continuity(frame, config, new_experiment=False)
    status = endterm.verify_continuity(frame, config, new_experiment=True)
    assert status["status"] == "separate_experiment"
    assert status["same_dataset"] is False
    assert status["same_original_split_verified"] is False
    assert status["current_dataset_sha256"] == dataset_fingerprint(frame)


def test_matching_dataset_cannot_claim_original_split_without_evidence(records, config):
    frame, _ = prepare_records(records, config)
    matching = {**config, "endterm": {**config["endterm"],
                                     "original_dataset_sha256": dataset_fingerprint(frame)}}
    with pytest.raises(ValueError, match="--new-experiment"):
        endterm.verify_continuity(frame, matching, new_experiment=False)
    status = endterm.verify_continuity(frame, matching, new_experiment=True)
    assert status["same_dataset"] is True
    assert status["same_original_split_verified"] is False
    assert status["status"] == "original_split_not_verified"


def test_raw_tampering_stops_before_any_model_fitting(records, config, tmp_path, monkeypatch):
    from appstore_ml import endterm_training

    response = _synthetic_raw(tmp_path, records)
    response.write_text(response.read_text() + " ")
    monkeypatch.setattr(endterm, "settings", lambda: {**config, "min_samples": 100})
    monkeypatch.setattr(endterm_training, "train_endterm",
                        lambda *args: pytest.fail("Training must not run after raw checksum failure"))
    with pytest.raises(ValueError, match="Raw response checksum mismatch"):
        endterm.experiment(tmp_path, tmp_path / "public", new_experiment=True, render=False)
    assert not (tmp_path / "outputs/reports/provenance.json").exists()


def test_saved_snapshot_mismatch_stops_before_fitting(records, config, tmp_path, monkeypatch):
    from appstore_ml import endterm_training

    _synthetic_raw(tmp_path, records)
    monkeypatch.setattr(endterm, "settings", lambda: {**config, "min_samples": 100})
    monkeypatch.setattr(endterm_training, "train_endterm",
                        lambda *args: pytest.fail("Training must not run after snapshot mismatch"))
    write_json(tmp_path / "data/snapshot.json", {"dataset_sha256": "invalid", "raw_checksums": {}})
    with pytest.raises(ValueError, match="Frozen snapshot changed"):
        endterm.experiment(tmp_path, tmp_path / "public", new_experiment=True, render=False)
    assert json.loads((tmp_path / "data/snapshot.json").read_text())["dataset_sha256"] == "invalid"


def test_saved_split_tampering_stops_before_fitting(records, config, tmp_path, monkeypatch):
    from appstore_ml import endterm_training

    _synthetic_raw(tmp_path, records)
    frame, _ = prepare_records(records, config)
    fixed_split(frame, tmp_path / "data/splits", config)
    manifest_path = tmp_path / "data/splits/split.csv"
    manifest = pd.read_csv(manifest_path)
    indices = [manifest.index[manifest["split"].eq(value)][0] for value in ["train", "test"]]
    manifest.loc[indices, "split"] = ["test", "train"]
    manifest.to_csv(manifest_path, index=False)
    monkeypatch.setattr(endterm, "settings", lambda: {**config, "min_samples": 100})
    monkeypatch.setattr(endterm_training, "train_endterm",
                        lambda *args: pytest.fail("Training must not run after split tampering"))
    with pytest.raises(ValueError, match="manifest checksum"):
        endterm.experiment(tmp_path, tmp_path / "public", new_experiment=True, render=False)


def test_public_copy_never_walks_private_model_or_prediction_directories(tmp_path):
    run_outputs = tmp_path / "run/outputs"
    (run_outputs / "reports").mkdir(parents=True)
    (run_outputs / "figures").mkdir()
    (run_outputs / "private").mkdir()
    (run_outputs / "models").mkdir()
    (run_outputs / "reports/test_metrics.csv").write_text("model,test_n,f1\nDummy,24,0.8\n")
    (run_outputs / "private/test_predictions.csv").write_text("trackId,predicted\n123456,1\n")
    (run_outputs / "models/dummy.joblib").write_bytes(b"SYNTHETIC TEST MODEL")
    public = tmp_path / "public"
    endterm.publish_aggregates(run_outputs, public)
    assert (public / "reports/test_metrics.csv").is_file()
    assert not list(public.rglob("*.joblib"))
    assert not list(public.rglob("test_predictions.csv"))
    assert not any("123456" in path.read_text() for path in public.rglob("*") if path.is_file())


@pytest.mark.parametrize("name,value", [
    ("inner_folds", 5), ("outer_folds", 3), ("pca_retained_variance", .95),
    ("mlp_hidden_layers", [64, 32]), ("mlp_epochs", 200), ("mlp_alpha", .01),
    ("mlp_learning_rate", .1), ("primary_metric", "accuracy"),
])
def test_changed_descriptive_config_cannot_silently_change_claims(config, name, value):
    endterm.validate_endterm_config(config)
    changed = copy.deepcopy(config)
    changed["endterm"][name] = value
    with pytest.raises(ValueError, match=name):
        endterm.validate_endterm_config(changed)


def test_public_copy_validates_every_source_before_any_copy(tmp_path):
    run_outputs = tmp_path / "run/outputs"
    (run_outputs / "reports").mkdir(parents=True)
    (run_outputs / "figures").mkdir()
    (run_outputs / "reports/test_metrics.csv").write_text("model,test_n,f1\nDummy,24,0.8\n")
    (run_outputs / "reports/cv_metrics.csv").write_text("trackId,high_rated\n123456,1\n")
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="Private app-level columns"):
        endterm.publish_aggregates(run_outputs, public)
    assert not list(public.rglob("*.csv"))
