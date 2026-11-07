"""Explicitly synthetic fixtures; these checks produce no research evidence."""
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from appstore_ml.endterm_analysis import analyze_unsupervised, evaluate_endterm
from appstore_ml.features import model_inputs
from appstore_ml.pipelines import preprocessing
from appstore_ml.prepare import prepare_records
from appstore_ml.split import fixed_split


def test_pca_uses_only_permitted_training_features(records, config, tmp_path):
    frame, _ = prepare_records(records, config)
    train, _, _ = fixed_split(frame, tmp_path / "split", config)
    result = analyze_unsupervised(train, tmp_path / "first", config)
    changed = train.copy()
    changed["high_rated"] = 1 - changed.high_rated
    changed["averageUserRating"] = 0
    changed["userRatingCount"] = 99999999
    changed["description"] = "POISONED EXCLUDED SYNTHETIC TEXT"
    repeat = analyze_unsupervised(changed, tmp_path / "second", config)
    assert result == repeat
    variance = pd.read_csv(tmp_path / "first/reports/pca_variance.csv")
    assert np.isclose(variance.explained_variance_ratio.sum(), 1)
    assert result["train_n"] == len(train)
    assert result["raw_input_count"] == 7
    public = json.dumps(result)
    assert "trackId" not in public and "100000" not in public
    assert (tmp_path / "first/private/pca_scores.csv").is_file()
    profiles = pd.read_csv(tmp_path / "first/reports/pca_profiles.csv")
    assert profiles.n.sum() == len(train)


def test_evaluates_every_model_and_keeps_rows_private(records, config, tmp_path):
    frame, _ = prepare_records(records, config)
    train, test, _ = fixed_split(frame, tmp_path / "split", config)
    names = ["Dummy", "SVM", "Decision Tree", "KNN", "Random Forest", "Boosting", "MLP", "SVM + PCA"]
    fitted = {}
    for index, name in enumerate(names):
        fitted[name] = Pipeline([("preprocess", preprocessing()),
                                 ("model", DummyClassifier(strategy="constant", constant=index % 2))])
        fitted[name].fit(model_inputs(train), train.high_rated)
    frozen_medians = fitted["SVM"]["preprocess"].named_transformers_["numeric"]["impute"].statistics_.copy()
    metrics, cases, matrices = evaluate_endterm(fitted, test, train, "SVM", tmp_path)
    assert set(metrics.model) == set(names) == set(matrices)
    assert len(metrics) == 8 and metrics.test_n.eq(len(test)).all()
    np.testing.assert_equal(frozen_medians,
                            fitted["SVM"]["preprocess"].named_transformers_["numeric"]["impute"].statistics_)
    for _, row in metrics.iterrows():
        assert row.tn + row.fp + row.fn + row.tp == len(test)
        assert np.isclose(row.specificity, row.tn / (row.tn + row.fp))
    assert set(cases.model) == set(names)
    assert cases.groupby(["model", "error_type"]).size().max() <= 2
    assert not set(["trackId", "averageUserRating", "userRatingCount", "primary_genre"]) & set(cases.columns)
    paired = pd.read_csv(tmp_path / "reports/paired_errors.csv")
    assert paired[["both_correct", "both_wrong", "model_only_correct", "svm_only_correct"]].sum(axis=1).eq(len(test)).all()
    private = pd.read_csv(tmp_path / "private/test_predictions.csv")
    assert len(private) == len(test) * 8 and "trackId" in private
