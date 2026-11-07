"""Explicitly synthetic fixtures: these tests produce no research results."""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from appstore_ml import endterm_training as endterm
from appstore_ml.common import FEATURES
from appstore_ml.features import model_inputs
from appstore_ml.prepare import prepare_records


def test_nested_search_rejects_rating_inputs_and_keeps_outer_rows_out(records, config, tmp_path, monkeypatch):
    frame, _ = prepare_records(records, config)
    original_specs = endterm.model_specs(config["random_state"])
    reduced_specs = {name: (clone(pipeline), grid) for name, (pipeline, grid) in original_specs.items()}
    reduced_specs["Random Forest"][0].set_params(model__n_estimators=4)
    reduced_specs["Random Forest"] = (reduced_specs["Random Forest"][0],
                                      {"model__max_depth": [2], "model__min_samples_leaf": [2]})
    reduced_specs["Gradient Boosting"] = (reduced_specs["Gradient Boosting"][0],
                                          {"model__n_estimators": [4], "model__max_depth": [1]})
    reduced_specs["MLP"][0].set_params(model__max_iter=4, model__n_iter_no_change=5)
    monkeypatch.setattr(endterm, "model_specs", lambda seed: reduced_specs)
    calls = []
    original_search = endterm._search

    def audited_search(pipeline, grid, X, y, seed, inner_folds):
        assert list(X.columns) == FEATURES
        assert X.index.equals(y.index)
        calls.append(set(X.index))
        return original_search(pipeline, grid, X, y, seed, inner_folds)

    monkeypatch.setattr(endterm, "_search", audited_search)
    fitted, cv, selected = endterm.train_endterm(frame, tmp_path, config)
    assignments = pd.read_csv(tmp_path / "private/cv_folds.csv")
    for ensemble_index in range(2):
        for fold in range(1, 6):
            validation_ids = set(assignments.loc[assignments.validation_fold.eq(fold), "trackId"])
            fit_ids = set(frame.loc[list(calls[ensemble_index * 6 + fold - 1]), "trackId"])
            assert not fit_ids & validation_ids
            assert fit_ids | validation_ids == set(frame.trackId)
        assert calls[ensemble_index * 6 + 5] == set(frame.index)
    assert len(fitted) == 8
    assert set(cv.model) == set(fitted)
    assert selected == cv.sort_values(["cv_f1_mean", "model"], ascending=[False, True]).iloc[0].model
    folds = pd.read_csv(tmp_path / "reports/cv_folds_metrics.csv")
    assert len(folds) == 40
    for name in fitted:
        values = folds.loc[folds.model.eq(name), "f1"]
        row = cv.loc[cv.model.eq(name)].iloc[0]
        np.testing.assert_allclose(row.cv_f1_mean, values.mean())
        np.testing.assert_allclose(row.cv_f1_std, values.std(ddof=1), atol=1e-14)
        assert list(fitted[name].feature_names_in_) == FEATURES
    assert (tmp_path / "models/mlp.joblib").exists()
    selection = json.loads((tmp_path / "reports/model_selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert selection["best_earlier_model"] != "MLP"
    assert selection["inner_folds"] == 3
    tuning = pd.read_csv(tmp_path / "reports/tuning_results.csv")
    assert set(tuning.scope) == {*(f"outer_fold_{i}" for i in range(1, 6)), "full_training_refit"}
    assert not ({"trackId", "averageUserRating", "userRatingCount"} & set(tuning.columns))


def test_mlp_diagnostic_preprocessing_never_fits_validation_rows(records, config, tmp_path, monkeypatch):
    frame, _ = prepare_records(records, config)
    prep = endterm.preprocessing()
    monkeypatch.setattr(endterm, "preprocessing", lambda: prep)
    curves, diagnostics = endterm.mlp_learning_check(frame, tmp_path, epochs=4)
    assignments = pd.read_csv(tmp_path / "private/mlp_diagnostic_split.csv")
    fit_ids = assignments.loc[assignments.diagnostic_split.eq("fit"), "trackId"]
    fitting = model_inputs(frame.loc[frame.trackId.isin(fit_ids)])
    observed = prep.named_transformers_["numeric"]["impute"].statistics_
    np.testing.assert_allclose(observed, fitting[config["numeric_features"]].median())
    assert len(curves) == 4
    assert curves.epoch.tolist() == [1, 2, 3, 4]
    assert np.isfinite(curves.select_dtypes("number").to_numpy()).all()
    assert diagnostics["test_used"] is False
    assert diagnostics["used_to_change_settings"] is False
    assert diagnostics["diagnostic_training_n"] == 96
    assert diagnostics["diagnostic_validation_n"] == 24


def test_pca_is_inside_classifier_pipeline(records, config):
    frame, _ = prepare_records(records, config)
    pipeline, grid = endterm.model_specs()["SVM + PCA"]
    assert grid is None
    assert list(pipeline.named_steps) == ["preprocess", "pca", "model"]
    fitted = pipeline.fit(model_inputs(frame.iloc[:90]), frame.high_rated.iloc[:90])
    before = fitted["pca"].components_.copy()
    heldout = model_inputs(frame.iloc[90:]).copy()
    heldout["price_usd"] = 1e12
    heldout["primary_genre"] = "SYNTHETIC UNSEEN CATEGORY"
    fitted.predict(heldout)
    np.testing.assert_array_equal(before, fitted["pca"].components_)
    assert fitted["pca"].explained_variance_ratio_.sum() >= 0.90
