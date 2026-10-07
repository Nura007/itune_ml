import copy
import json

import numpy as np
import pandas as pd
import pytest

from appstore_ml.collect import check_approval, Collector
from appstore_ml.common import FEATURES, sha256, write_json
from appstore_ml.features import model_inputs
from appstore_ml.pipelines import models
from appstore_ml.prepare import prepare_records
from appstore_ml.run import experiment
from appstore_ml.split import fixed_split


def test_thresholds_and_latest_observation(records, config):
    rows = copy.deepcopy(records[:5])
    rows[0]["averageUserRating"] = 4.5
    rows[0]["userRatingCount"] = 50
    rows[1]["userRatingCount"] = 49
    rows[2]["averageUserRating"] = 4.4999
    rows[3]["averageUserRating"] = None
    newer = {**rows[4], "averageUserRating": 2.0, "_collected_at": "2026-01-02T00:00:00Z"}
    frame, audit = prepare_records(rows + [newer], config)
    assert frame.trackId.tolist() == [100000, 100002, 100004]
    assert frame.high_rated.tolist() == [1, 0, 0]
    assert audit["excluded"]["duplicate_track_id"] == 1
    assert audit["raw_rows"] == audit["clean_rows"] + sum(audit["excluded"].values())


def test_bad_features_and_no_rating_leakage(records, config):
    rows = copy.deepcopy(records[:2])
    rows[0].update(price=-3, fileSizeBytes="bad", languageCodesISO2A=None,
                   currentVersionReleaseDate="2030-01-01", releaseDate="1900-01-01",
                   primaryGenreName="")
    rows[1]["languageCodesISO2A"] = []
    frame, _ = prepare_records(rows, config)
    assert frame.loc[0, ["price_usd", "size_mb", "language_count", "release_year",
                         "update_recency_days", "primary_genre"]].isna().all()
    assert frame.loc[1, "language_count"] == 0
    assert list(model_inputs(frame).columns) == FEATURES
    assert "averageUserRating" not in model_inputs(frame)
    assert frame.loc[1, "size_mb"] == 11


def test_reject_bad_labels_and_foreign_currency(records, config):
    rows = copy.deepcopy(records[:5])
    rows[0]["currency"] = "EUR"
    rows[1]["userRatingCount"] = 50.5
    rows[2]["averageUserRating"] = float("inf")
    rows[3]["trackId"] = 3.2
    rows[4]["_collected_at"] = "invalid"
    frame, audit = prepare_records(rows, config)
    assert len(frame) == 0
    assert sum(audit["excluded"].values()) == 5


def test_permanent_split_and_dataset_change(records, config, tmp_path):
    frame, _ = prepare_records(records, config)
    train, test, _ = fixed_split(frame, tmp_path, config)
    train_again, test_again, _ = fixed_split(frame.sample(frac=1, random_state=7), tmp_path, config)
    assert train.trackId.tolist() == train_again.trackId.tolist()
    assert test.trackId.tolist() == test_again.trackId.tolist()
    assert not set(train.trackId) & set(test.trackId)
    frame.loc[0, "price_usd"] += 1
    with pytest.raises(ValueError, match="changed"):
        fixed_split(frame, tmp_path, config)


def test_preprocessing_training_only_unknown_category(records, config):
    frame, _ = prepare_records(records, config)
    X = model_inputs(frame)
    fitted = models()["KNN"].fit(X.iloc[:90], frame.high_rated.iloc[:90])
    medians = fitted["preprocess"].named_transformers_["numeric"]["impute"].statistics_.copy()
    heldout = X.iloc[90:].copy()
    heldout["price_usd"] = 1e12
    heldout["primary_genre"] = "UNSEEN TEST CATEGORY"
    assert len(fitted.predict(heldout)) == 30
    np.testing.assert_equal(medians, fitted["preprocess"].named_transformers_["numeric"]["impute"].statistics_)
    np.testing.assert_equal(medians, X.iloc[:90][config["numeric_features"]].median().values)


def test_approval_blocks_network(tmp_path):
    approval = tmp_path / "approval.json"
    write_json(approval, {"status": "pending"})
    with pytest.raises(PermissionError):
        check_approval(approval)


def test_collector_cache_and_metadata(tmp_path, config):
    class Response:
        status_code = 200
        headers = {}
        url = "https://itunes.apple.com/search?term=fixture"
        content = b'{"resultCount": 0, "results": []}'
        def raise_for_status(self):
            pass
        def json(self):
            return json.loads(self.content)
    class Session:
        headers = {}
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()
    session = Session()
    collector = Collector(tmp_path, config, {"status": "approved"}, session=session)
    collector.fetch("fixture", 50)
    collector.fetch("fixture", 50)
    assert session.calls == 1
    response = next(tmp_path.glob("*.response.json"))
    meta = json.loads(next(tmp_path.glob("*.meta.json")).read_text())
    assert meta["sha256"] == sha256(response)
    assert meta["params"]["country"] == "us"


def test_end_to_end_synthetic_only(records, config, tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    body = raw / "fixture.response.json"
    write_json(body, {"resultCount": len(records), "results": records})
    write_json(raw / "fixture.meta.json", {"sha256": sha256(body),
               "collected_at": records[0]["_collected_at"], "params": {"term": "synthetic"}})
    test_config = {**config, "min_samples": 100}
    result = experiment(tmp_path, config=test_config, data_kind="synthetic_test_NOT_RESEARCH")
    assert result["train_n"] == 96 and result["test_n"] == 24
    metrics = pd.read_csv(tmp_path / "outputs/reports/test_metrics.csv")
    assert len(metrics) == 4 and metrics.f1.between(0, 1).all()
    assert metrics.loc[metrics.model.eq("Dummy"), "roc_auc"].iloc[0] == .5
    assert (tmp_path / "outputs/midterm.pptx").is_file()
    assert len(list((tmp_path / "outputs/models").glob("*.joblib"))) == 4
    report = (tmp_path / "outputs/reports/results.md").read_text()
    assert "synthetic_test_NOT_RESEARCH" in report
    with pytest.raises(ValueError, match="minimum"):
        experiment(tmp_path, config=config)


def test_empty_results_have_auditable_zero_rows(config):
    frame, audit = prepare_records([], config)
    assert len(frame) == 0
    assert audit["raw_rows"] == 0 and not audit["minimum_met"]


def test_split_tampering_is_detected(records, config, tmp_path):
    frame, _ = prepare_records(records, config)
    fixed_split(frame, tmp_path, config)
    p = tmp_path / "split.csv"
    manifest = pd.read_csv(p)
    indices = [manifest.index[manifest["split"].eq(label)][0] for label in ["train", "test"]]
    manifest.loc[indices, "split"] = ["test", "train"]
    manifest.to_csv(p, index=False)
    with pytest.raises(ValueError, match="checksum"):
        fixed_split(frame, tmp_path, config)
