"""Generate English Endterm artifacts strictly from published real-run aggregates."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FEATURES, ROOT


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value, digits=3):
    return f"{float(value):.{digits}f}"


def _markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows])


def _plots(comparison, learning, figures):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ["#af6d23" if name == "MLP" else "#23657a" for name in comparison.model]
    positions = np.arange(len(comparison))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].barh(positions, comparison.cv_f1_mean, xerr=comparison.cv_f1_std,
                 color=colors, alpha=.85, capsize=3, label="CV mean +/- sample SD")
    axes[0].scatter(comparison.f1, positions, color="#131d30", marker="D", s=25,
                    label="Frozen holdout F1", zorder=3)
    axes[0].set(yticks=positions, yticklabels=comparison.model, xlabel="F1, class 1", xlim=(0, 1.05))
    axes[0].invert_yaxis()
    axes[0].set_title("High positive-class F1 needs a baseline")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].barh(positions - .17, comparison.roc_auc, height=.32, color="#23657a", label="Holdout ROC-AUC")
    axes[1].barh(positions + .17, comparison.balanced_accuracy, height=.32,
                 color="#af6d23", label="Holdout balanced accuracy")
    axes[1].axvline(.5, color="#596474", linestyle="--", linewidth=1)
    axes[1].set(yticks=positions, yticklabels=[""] * len(positions), xlabel="Score", xlim=(0, 1.05))
    axes[1].invert_yaxis()
    axes[1].set_title("Discrimination and both classes")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.savefig(figures / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    for key, label, color in [("training", "Diagnostic training", "#23657a"),
                              ("validation", "Internal validation", "#af6d23")]:
        axes[0].plot(learning.epoch, learning[f"{key}_log_loss"], label=label, color=color)
        axes[1].plot(learning.epoch, learning[f"{key}_f1"], label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Unpenalized log loss", title="MLP learning: lower loss is better")
    axes[1].set(xlabel="Epoch", ylabel="F1, class 1", ylim=(0, 1.02), title="Fixed 120-epoch diagnostic")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.savefig(figures / "mlp_learning.png", dpi=180)
    plt.close(fig)


def _sections(public_dir):
    team = _json(ROOT / "config/team.json")
    student_names = "; ".join(student["name"] for student in team["students"])
    reports = public_dir / "reports"
    comparison = pd.read_csv(reports / "comparison.csv")
    selection = _json(reports / "model_selection.json")
    provenance = _json(reports / "provenance.json")
    continuity = _json(reports / "continuity.json")
    audit = _json(reports / "cleaning_audit.json")
    unsupervised = _json(reports / "unsupervised.json")
    mlp = _json(reports / "mlp_diagnostics.json")
    errors = pd.read_csv(reports / "error_cases_anonymized.csv")
    paired = pd.read_csv(reports / "paired_errors.csv")
    profiles = pd.read_csv(reports / "error_profiles.csv")
    pca_profiles = pd.read_csv(reports / "pca_profiles.csv").sort_values("pc1_stratum")
    learning = pd.read_csv(reports / "mlp_learning_curve.csv")
    if provenance["data_kind"] != "real_api":
        raise ValueError("Public Endterm research reports require a real API run.")
    selected = comparison.set_index("model").loc[selection["selected_model"]]
    indexed = comparison.set_index("model")
    svm, pca, network, dummy = [indexed.loc[name] for name in ["SVM", "SVM + PCA", "MLP", "Dummy"]]
    earlier_name = selection["best_earlier_model"]
    earlier = indexed.loc[earlier_name]
    n = provenance["train_n"] + provenance["test_n"]
    positive_n = int(audit["class_counts"]["1"])
    negative_n = int(audit["class_counts"]["0"])
    selected_name = selection["selected_model"]
    continuity_text = (
        "CONTINUITY GAP - The original Midterm raw data and saved split are unavailable. "
        "This report is a separately collected experiment. Every model below uses the same new frozen "
        "split, but none of these comparisons establishes improvement on the original Midterm holdout. "
        "Criterion 1's same-data/split requirement remains unfulfilled. No instructor exception is claimed."
    )
    if continuity.get("same_original_split_verified"):
        raise ValueError("Report wording must be reviewed before claiming restored original-split continuity.")
    sections = []

    def section(title, subtitle, items):
        sections.append({"title": title, "subtitle": subtitle, "items": items})

    section("App Store ratings: Endterm study", "01 / Research question, evidence and continuity", [
        ("p", f"Students: {student_names}. Group: {team['group']}."),
        ("callout", continuity_text),
        ("p", "Research question: can seven metadata attributes distinguish existing US App Store apps "
         "rated at least 4.5 from lower-rated apps, among apps with at least 50 user ratings? The target "
         "describes the collection snapshot. It does not measure pre-launch success or establish causality."),
        ("table", ["Study property", "Observed record"], [
            ["Eligible real apps", f"{n:,}; class 1: {positive_n:,} ({positive_n/n:.1%}); class 0: {negative_n:,}"],
            ["Saved partition", f"{provenance['train_n']:,} training / {provenance['test_n']:,} test; stratified 80/20; seed 42"],
            ["Eligible-record timestamps, UTC", f"{audit['collection_start']} to {audit['collection_end']}"],
            ["Source responses", f"{len(provenance['source_response_checksums'])} checksummed API JSON responses"],
            ["Raw rows / deduplicated exclusions", f"{audit['raw_rows']:,} rows; {audit['excluded'].get('duplicate_track_id', 0):,} duplicate observations removed"],
            ["Experiment", provenance["experiment_id"]],
        ], [135, 370]),
        ("p", f"The training-CV rule selected {selected_name}. Its holdout F1 is {selected.f1:.4f}, "
         f"versus {dummy.f1:.4f} for the majority baseline (difference {selected.f1-dummy.f1:+.4f}). "
         f"ROC-AUC is {selected.roc_auc:.4f}; balanced accuracy is {selected.balanced_accuracy:.4f}. "
         f"It identifies {int(selected.tn)}/{int(selected.tn+selected.fp)} lower-rated apps. "
         "These are descriptive results, not a statistical-significance claim."),
        ("p", "The Endterm extension adds nested ensemble tuning, training-only PCA, a fold-local PCA ablation, "
         "MLP learning diagnostics, complete cross-model errors and a persistent private snapshot. "
         "Code and aggregate artifacts are reviewable; human contributions and defense performance remain the students' responsibility."),
    ])

    descriptions = ["Price in US dollars", "File size in decimal MB", "Distinct supported-language count",
                    "Original release year", "Full days since latest update at collection time",
                    "Primary app category", "Age-suitability category; not a user rating"]
    section("Seven inputs, one protected holdout", "02 / Features, preprocessing and evaluation protocol", [
        ("table", ["Permitted model input", "Definition", "Missing rows"],
         [[feature, description, str(audit["missing_features"][feature])]
          for feature, description in zip(FEATURES, descriptions)], [153, 287, 65]),
        ("p", "averageUserRating creates class 1 at >=4.5; integer userRatingCount >=50 is an eligibility "
         "filter. Ratings, rating counts, IDs, app names, developer IDs, descriptions and all other text are "
         "excluded from model inputs. Latest observations are deduplicated by trackId before eligibility filtering."),
        ("p", "Numeric features receive training-fold median imputation and standardization; an entirely missing "
         "training column is retained with zero imputation. Categorical values receive a Missing category and "
         "one-hot encoding; unseen categories are ignored. The imputer, encoder and scaler are inside every "
         "pipeline. PCA used for prediction is also fitted within each training fold."),
        ("p", "All models share the identical five stratified outer folds inside training. Random forest and "
         "boosting use three inner folds to choose settings, then score the unseen outer fold. Final ensemble "
         "settings use a fresh three-fold search on all training rows. The holdout is evaluated after the "
         "selection rule is frozen. No threshold tuning, test-guided preprocessing or test-based reselection occurs."),
        ("p", "F1 for class 1 preserves the Midterm decision criterion and balances precision with recall for "
         "high-rated apps. The majority-positive distribution makes this score forgiving of missed class-0 apps. "
         "ROC-AUC, balanced accuracy, specificity and both confusion-matrix rows therefore remain essential. "
         "CV spread is sample standard deviation across five folds, not a confidence interval. SVM ROC-AUC "
         "uses decision scores, which are not calibrated probabilities."),
    ])

    search_rows = []
    for name, space in selection["search_spaces"].items():
        search_rows.append([name, "; ".join(f"{k.replace('model__', '')}: {v}" for k, v in space.items()),
                            str(int(np.prod([len(v) for v in space.values()])))])
    parameter_rows = [[name, "; ".join(f"{key.replace('model__', '')}={value}"
                                       for key, value in selection["selected_parameters"][name].items())]
                      for name in ["Random Forest", "Gradient Boosting"]]
    ensemble_rows = [[name, f"{indexed.loc[name].cv_f1_mean:.4f} +/- {indexed.loc[name].cv_f1_std:.4f}",
                      f"{indexed.loc[name].f1:.4f}", f"{indexed.loc[name].f1-svm.f1:+.4f}"]
                     for name in ["SVM", "Random Forest", "Gradient Boosting"]]
    section("Tuned ensembles with honest CV", "03 / Search choices and the earlier-model reference", [
        ("p", "Random forest averages 100 decorrelated trees. Its search varies depth, leaf size and feature "
         "subsampling to control complexity and diversity. Gradient boosting adds small decision trees "
         "sequentially; its search trades learning rate against the number and depth of trees. These compact "
         "spaces were fixed before the holdout was evaluated and are feasible for this sample size."),
        ("table", ["Ensemble", "Prespecified search space", "Candidates"], search_rows, [95, 350, 60]),
        ("p", "Each of eight candidates is evaluated on three inner folds within each of five outer folds. "
         "A separate full-training search chooses the final refit. Candidate search scores are tuning evidence; "
         "the outer scores provide the model comparison. Ties within a search use deterministic parameter order; "
         "the final model-family tie-break is alphabetical name."),
        ("table", ["Ensemble", "Selected full-training settings"], parameter_rows, [108, 397]),
        ("table", ["Model", "Outer CV F1 +/- SD", "Test F1", "Test F1 minus SVM"], ensemble_rows,
         [116, 167, 90, 132]),
        ("p", "The earlier fixed models are majority Dummy; Decision Tree (depth 8, minimum leaf 10); "
         "distance-weighted KNN (k=15); and RBF SVM (C=1, gamma=scale). The original Midterm selected SVM. "
         "All four are rerun on this new snapshot; their results must not be conflated with the original experiment."),
        ("p", "The complete candidate-level means, sample SDs and ranks are in reports/tuning_results.csv. "
         "Selected parameters and outer-fold scores are also retained. A winning outer-CV mean may still be "
         "optimistic because the same outer scores choose a model family."),
    ])

    table_rows = [[row.model, f"{row.cv_f1_mean:.3f} +/- {row.cv_f1_std:.3f}",
                   *[f"{getattr(row, name):.3f}" for name in ["f1", "roc_auc", "precision", "recall",
                                                            "balanced_accuracy", "specificity"]]]
                  for row in comparison.itertuples()]
    section("Every model on the same rows", "04 / Combined performance comparison", [
        ("p", f"Selection is based on training outer-CV F1: {selected_name}. Test columns are reported after "
         "selection and do not change that choice. All eight models share the same frozen holdout. "
         "Only the ensemble rows use inner tuning; all other settings are fixed."),
        ("table", ["Model", "CV F1 +/- SD", "Test F1", "AUC", "Prec.", "Recall", "Bal. acc.", "Spec."],
         table_rows, [100, 93, 52, 52, 52, 52, 52, 52]),
        ("image", "comparison.png", 505, 211),
        ("p", "Precision is the fraction of predicted high-rated apps that are truly class 1. Recall is the "
         "fraction of high-rated apps detected. Specificity is recall for lower-rated apps. Balanced accuracy "
         "averages recall and specificity. These complementary views reveal trade-offs that positive-class F1 alone conceals."),
        ("p", "Paired test differences are descriptive. There is one convenience sample and one holdout, "
         "without an independent external validation cohort or a prespecified significance test. Small ranking "
         "changes are not evidence of reliable general superiority."),
    ])

    loadings = [[component, entry["feature"].replace("numeric__", "").replace("categorical__", ""),
                 f"{entry['loading']:+.3f}"]
                for component, entries in unsupervised["dominant_loadings"].items() for entry in entries[:3]]
    lower, upper = pca_profiles.iloc[0], pca_profiles.iloc[-1]
    profile_interpretation = (
        f"The lowest versus highest PC1 quartiles have median sizes of {lower.median_size_mb:.1f} versus "
        f"{upper.median_size_mb:.1f} MB, release years {lower.median_release_year:.0f} versus "
        f"{upper.median_release_year:.0f}, and update ages {lower.median_update_recency_days:.0f} versus "
        f"{upper.median_update_recency_days:.0f} days. This gives the direction an observable metadata "
        "interpretation rather than treating a plotted separation as a discovered app category."
    )
    section("What varies in app metadata?", "05 / Training-only PCA and its predictive check", [
        ("p", f"PCA uses full SVD on {unsupervised['train_n']:,} training objects and "
         f"{unsupervised['transformed_feature_count']} transformed columns. Numeric columns are standardized; "
         "one-hot categories stay at 0/1, so category prevalence influences their contribution. Labels and "
         "holdout rows are not used to fit this descriptive projection."),
        ("p", f"PC1 explains {unsupervised['pc1_explained_variance']:.1%} and PC2 "
         f"{unsupervised['pc2_explained_variance']:.1%} of training variance. The first two together explain "
         f"{unsupervised['first_two_explained_variance']:.1%}; {unsupervised['components_for_90_percent']} "
         "components are required for 90%. Explained variance assesses information retained, not natural-group stability."),
        ("table", ["Direction", "Largest signed loadings (top three by magnitude)", "Loading"],
         loadings, [68, 365, 72]),
        ("small", profile_interpretation),
        ("image_pair", "pca_variance.png", "pca_map.png"),
        ("p", "Actual training objects, reported coarsely: " + "; ".join(
            f"{row['example']}: {row['description']}" for row in unsupervised["examples"]) + ". "
         "PC1 quartile median profiles are saved in reports/pca_profiles.csv. Component signs are arbitrary; "
         "the density map does not prove categories, causation or distinct clusters."),
        ("p", f"Predictive connection: fold-local 90%-variance PCA changes SVM CV F1 by "
         f"{pca.cv_f1_mean-svm.cv_f1_mean:+.4f} and test F1 by {pca.f1-svm.f1:+.4f}. "
         "That pipeline refits its own PCA inside every fold; it does not reuse the full-training visualization. "
         "Variance preservation can discard low-variance rating signals, and outliers can dominate directions."),
    ])

    final_gap = mlp["final_validation_log_loss"] - mlp["final_training_log_loss"]
    last_change = mlp["validation_loss_change_last_window"]
    learning_interpretation = (
        f"The final validation loss is {mlp['final_validation_log_loss']:.4f}, versus training loss "
        f"{mlp['final_training_log_loss']:.4f} (gap {final_gap:+.4f}). Validation loss changed by "
        f"{last_change:+.4f} over the last {mlp['window_epochs']} recorded epochs. "
        f"Its minimum was at epoch {mlp['best_observed_validation_epoch']}, "
        f"{mlp['validation_loss_above_observed_minimum']:.4f} below the final value. "
    )
    if mlp["validation_loss_above_observed_minimum"] > .01:
        learning_interpretation += "The late validation deterioration is consistent with overfitting. "
    elif abs(last_change) < .005:
        learning_interpretation += "The small late validation change is consistent with a plateau. "
    else:
        learning_interpretation += "Learning is still changing at the fixed budget; convergence is not established. "
    section("MLP: check learning before claims", "06 / Architecture, internal validation and fair reference", [
        ("p", "The MLP has two hidden layers (32 and 16 ReLU units), a binary output, Adam optimization, "
         "learning rate 0.001, L2 alpha 0.001, batch size 64 and a fixed 120-epoch budget. It uses the same "
         "seven inputs and fold-local preprocessing as other models. No architecture, epoch or threshold search is performed."),
        ("image", "mlp_learning.png", 505, 207),
        ("p", f"The diagnostic split is entirely inside training: {mlp['diagnostic_training_n']:,} learning "
         f"objects and {mlp['diagnostic_validation_n']:,} validation objects. Its imputer/scaler/encoder is "
         "fitted only on the learning subset. Curves diagnose the fixed recipe; the best observed epoch is "
         "not used to change settings. CV then independently fits the fixed recipe in every outer training fold."),
        ("p", learning_interpretation + "Reaching 120 iterations is not a mathematical convergence guarantee. "
         "Full fold/refit convergence warnings are retained in model_selection.json."),
        ("table", ["Reference chosen using CV", "CV F1 +/- SD", "Test F1", "Test AUC"], [
            [earlier_name, f"{earlier.cv_f1_mean:.4f} +/- {earlier.cv_f1_std:.4f}",
             f"{earlier.f1:.4f}", f"{earlier.roc_auc:.4f}"],
            ["MLP", f"{network.cv_f1_mean:.4f} +/- {network.cv_f1_std:.4f}",
             f"{network.f1:.4f}", f"{network.roc_auc:.4f}"],
        ], [180, 155, 85, 85]),
        ("p", f"Relative to the best earlier non-MLP model, MLP CV F1 differs by "
         f"{network.cv_f1_mean-earlier.cv_f1_mean:+.4f}; test F1 differs by {network.f1-earlier.f1:+.4f}. "
         "A neural network is not inherently better for this small tabular task. Its flexibility, finite optimization "
         "budget and the limited information in metadata can all affect performance; no causal explanation is established."),
    ])

    section("Confusion matrices expose the trade-off", "07 / Frozen-holdout errors for every model", [
        ("p", "Rows are true classes and columns are predictions. Class 0 is rating <4.5; class 1 is rating "
         ">=4.5. False positives are lower-rated apps called high-rated. False negatives are high-rated apps "
         "called lower-rated. Matrix entries are exact aggregate counts on the common holdout."),
        ("image", "confusion_matrices.png", 505, 446),
        ("p", f"The selected {selected_name} model makes {int(selected.fp)} false positives and "
         f"{int(selected.fn)} false negatives. The majority Dummy makes {int(dummy.fp)} false positives "
         f"and {int(dummy.fn)} false negatives. Positive-class F1 must be interpreted alongside these counts; "
         "the majority classifier can score highly while never recognizing a lower-rated app."),
    ])

    case_rows = []
    error_paragraphs = []
    for row in comparison.itertuples():
        model_errors = errors.loc[errors.model.eq(row.model)]
        chosen = []
        for kind in ["FP", "FN"]:
            candidates = model_errors.loc[model_errors.error_type.eq(kind)]
            if len(candidates):
                chosen.append(f"{kind}: {candidates.iloc[0].observation}")
            else:
                chosen.append(f"{kind}: no such error occurred on this holdout")
        case_rows.append([row.model, f"FP {row.fp}\nFN {row.fn}", "; ".join(chosen)])
        error_paragraphs.append(f"{row.model}: {row.fp} FP and {row.fn} FN; precision {row.precision:.3f}, "
                                f"recall {row.recall:.3f}, specificity {row.specificity:.3f}. "
                                f"Training-minus-CV F1 gap {row.f1_generalization_gap:+.3f}.")
    section("Real errors, anonymized for publication", "08 / Actual examples across all eight models", [
        ("p", "Examples are real held-out errors, selected within each model/error type by distance from the "
         "4.5 boundary, then ID. At most two per error type are saved; one per type is summarized here. "
         "Quartiles are computed on training data. Exact ratings, IDs, names, descriptions and input vectors remain private."),
        ("table", ["Model", "All errors", "Coarse observations from actual held-out errors"], case_rows,
         [87, 53, 365]),
        ("p", "These observations locate cases where metadata prediction fails. They do not explain why users "
         "gave a rating or why a model made an individual error. Borderline labels may be unstable as ratings "
         "change, and the size/update summaries cannot capture usability, bugs or user expectations. "
         "The full anonymized case table is reports/error_cases_anonymized.csv; detailed predictions stay private."),
    ])

    paired_rows = [[row.model, row.model_only_correct, row.svm_only_correct, row.both_wrong,
                    row.prediction_disagreements] for row in paired.itertuples()]
    profile_text = "Genre-level error rates are reported only for groups of at least 20 held-out apps. "
    selected_profiles = profiles.loc[profiles.model.eq(selected_name)] if len(profiles) else profiles
    if len(selected_profiles):
        worst = selected_profiles.sort_values(["error_rate", "genre"], ascending=[False, True]).iloc[0]
        profile_text += (f"For {selected_name}, the highest observed eligible-group error rate is "
                         f"{worst.error_rate:.1%} in {worst.genre} ({int(worst.errors)}/{int(worst.test_n)}). "
                         "This is an exploratory, sample-specific association, without a causal claim or multiple-group correction.")
    else:
        profile_text += "No eligible group is available for the selected model in this run."
    section("Where the models disagree", "09 / Paired comparisons, model trade-offs and error cohorts", [
        ("p", "Each paired count compares predictions on the same app. Model-only correct means the candidate "
         "fixes an SVM error; SVM-only correct means it introduces an error where SVM was right. "
         "These counts characterize trade-offs, not statistical significance."),
        ("table", ["Model vs SVM", "Model only correct", "SVM only correct", "Both wrong", "Disagreements"],
         paired_rows, [125, 95, 95, 95, 95]),
        *[("small", item) for item in error_paragraphs],
        ("p", f"The baseline detects {int(dummy.tn)} lower-rated apps. Relative to SVM, Decision Tree changes "
         f"FP/FN counts by {int(indexed.loc['Decision Tree'].fp-svm.fp):+d}/"
         f"{int(indexed.loc['Decision Tree'].fn-svm.fn):+d}, and KNN by "
         f"{int(indexed.loc['KNN'].fp-svm.fp):+d}/{int(indexed.loc['KNN'].fn-svm.fn):+d}. "
         f"KNN's training F1 is {indexed.loc['KNN'].train_f1_mean:.3f}, with a "
         f"{indexed.loc['KNN'].f1_generalization_gap:.3f} CV gap: fitting the training sample well does not "
         "ensure generalization. Random Forest AUC is "
         f"{indexed.loc['Random Forest'].roc_auc:.3f}, versus SVM {svm.roc_auc:.3f}. "
         "The prespecified F1 rule does not select by AUC; this is not a reason to reselect after the test."),
        ("p", f"Gradient Boosting misses only {int(indexed.loc['Gradient Boosting'].fn)} positive app(s), "
         f"while recognizing just {int(indexed.loc['Gradient Boosting'].tn)} lower-rated app(s). PCA changes "
         f"{int(paired.loc[paired.model.eq('SVM + PCA'), 'prediction_disagreements'].iloc[0])} SVM predictions, "
         f"with CV F1 difference {pca.cv_f1_mean-svm.cv_f1_mean:+.4f}. MLP recognizes "
         f"{int(network.tn)} class-0 objects but misses {int(network.fn)} positives. Its CV F1 difference "
         f"from the best earlier reference is {network.cv_f1_mean-earlier.cv_f1_mean:+.4f}; the learning "
         "curves must be read alongside this trade-off."),
        ("p", profile_text),
        ("p", "Large training-minus-CV gaps suggest fitting variation that does not generalize under this "
         "validation design. A smaller gap does not by itself imply strong discrimination: the Dummy can "
         "have a near-zero gap while missing every class-0 app. Model selection remains tied to the prespecified F1 rule."),
    ])

    conclusion = (f"On this separately collected snapshot, {selected_name} is selected by training CV. "
                  f"Its holdout AUC of {selected.roc_auc:.3f} and balanced accuracy of "
                  f"{selected.balanced_accuracy:.3f} describe how much the seven metadata inputs distinguish "
                  "the two rating classes. The observed F1 difference from Dummy is "
                  f"{selected.f1-dummy.f1:+.4f}. This answers a snapshot classification question, "
                  "not pre-launch prediction, business success or the causes of satisfaction.")
    section("Conclusions and a feasible Final", "10 / Limits tied to the data and a lawful source plan", [
        ("p", conclusion),
        ("p", "What changed since Midterm: two ensembles now have documented nested searches; PCA is both "
         "interpreted and tested in a fold-local predictive ablation; MLP learning is checked before comparison; "
         "all models have paired errors and minority-class measures; raw data and a checksummed manifest are "
         "retained privately. The new snapshot and missing original archive prevent a same-test-set before/after claim."),
        ("p", "Limitations: keyword-ranked collection is a convenience sample. The >=50-ratings filter selects "
         "established apps. Developer-related apps may cross the random split. Metadata and labels evolve over "
         "time, update recency can reflect popularity, and threshold-adjacent labels can change. Class imbalance "
         "makes F1 appear strong even when lower-rated apps are missed. No feature captures experienced bugs, "
         "usability or expectation mismatch, so coarse error associations are not explanations."),
        ("callout", "FINAL SOURCE PLAN - Retained developer description text from the documented iTunes Search API "
         "is a candidate source, not an assertion of reuse rights. Review the current source conditions and obtain "
         "appropriate permission before text modeling or redistribution. Instructor course approval is not copyright "
         "permission and does not override robots.txt or Apple's terms. If permission is unavailable, use an "
         "appropriately licensed, app-linked alternative and document its license."),
        ("p", "Linkage and purpose: join approved descriptions to the existing objects using trackId privately. "
         "Keep only the same frozen eligible cohort, record text availability and checksums, and report unmatched "
         "objects. Descriptions may encode intended functions and use cases absent from seven metadata inputs; "
         "they may still omit actual defects and are promotional, so their usefulness is a hypothesis."),
        ("p", "Next experiment: prespecify a simple word/character TF-IDF representation with a linear classifier, "
         "then compare metadata-only, text-only and combined inputs. Fit vocabulary, weighting and all tuning "
         "inside training folds; avoid names, IDs or explicit rating claims as shortcuts. Compare the same metrics, "
         "class-0 errors and paired mistakes on matched objects. Because this holdout has now been inspected, "
         "use a later untouched, licensed snapshot or a prespecified developer/time holdout for a strong confirmatory "
         "Final claim. Any new snapshot is a separate experiment. A local application remains Final scope."),
    ])

    criterion_rows = [
        ["1 / Continuity and fixes", "Incomplete: original data/split unavailable. New snapshot, change log and checksums retained."],
        ["2 / Ensembles", "RF and boosting search spaces, nested CV, selected settings and SVM comparison."],
        ["3 / Unsupervised", "Training PCA variance, loadings, actual coarse examples and fold-local SVM ablation."],
        ["4 / Neural network", "MLP train/validation learning curves, fixed settings and CV-best earlier reference."],
        ["5 / Comparison and errors", "Eight-model table, confusion matrices, actual anonymized errors and paired counts."],
        ["6 / Conclusions and Final", "Qualified findings, observed limits and a permission-dependent linked-text plan."],
        ["7 / Seven-minute talk", "Three named speakers share 420 seconds. Students must rehearse and confirm team-size acceptance."],
        ["8 / Individual defense", "Names and assigned roles recorded; completed contributions and independent understanding require verification."],
    ]
    section("Reproduce, disclose and defend", "11 / Submission evidence, remaining duties and references", [
        ("p", "Use docs/endterm_run.md for installation and execution, and docs/endterm_changes.md for the "
         "change log. The private submission package should preserve raw responses/metadata, cleaned data, "
         "saved split, fitted models, predictions and checksums. Public artifacts contain only aggregates. "
         "Do not force-add ignored files or publish the private package. The notebook is designed to rerun the "
         "pipeline using the saved snapshot; the source code never needs test rows during training."),
        ("table", ["Rubric criterion", "Evidence / honest status"], criterion_rows, [145, 360]),
        ("table", ["Student / group " + team["group"], "Assigned responsibility (completion unverified)"],
         [[student["name"], student["responsibility"]] for student in team["students"]], [145, 360]),
        ("p", "AI disclosure: OpenAI Codex assisted with implementation, analysis, automated checks and English "
         "report/presentation drafting. The students must verify all claims, document their actual contributions "
         "in docs/contributions.md and explain any part independently. Names were supplied by the owner. "
         "Assigned roles are a work plan, not a claim of completed human work. No hours, instructor "
         "exception or defense performance have been invented."),
        ("small", "The owner listed three students. The rubric describes two partners; instructor acceptance "
         "of this team size has not been supplied. All three names are retained without assuming an exception."),
        ("small", "Submission: team members use the same complete archive and the rubric's "
         "Endterm_Surname_Name_Group.zip naming convention by 23:59 on the day before their defense. "
         "The defense date has not been supplied."),
        ("small", "References: Endterm Rubric.pdf, supplied by the owner and read for this implementation. "
         "Apple Search API: https://performance-partners.apple.com/search-api ; source robots: "
         "https://itunes.apple.com/robots.txt . These are source-context references, not a claim of reuse permission."),
        ("small", "Method references: https://scikit-learn.org/1.6/modules/cross_validation.html ; "
         "https://scikit-learn.org/1.6/modules/grid_search.html ; "
         "https://scikit-learn.org/1.6/modules/decomposition.html#pca ; "
         "https://scikit-learn.org/1.6/modules/neural_networks_supervised.html . "
         "Software versions and source hashes are recorded in reports/environment.txt and reports/source_hashes.json."),
    ])
    return sections, comparison, learning, provenance


def _write_markdown(sections, path):
    parts = []
    for section in sections:
        parts.extend(["# " + section["title"], section["subtitle"], ""])
        for item in section["items"]:
            if item[0] in {"p", "small", "callout"}:
                parts.append(("**" + item[1] + "**") if item[0] == "callout" else item[1])
            elif item[0] == "table":
                parts.append(_markdown_table(item[1], item[2]))
            elif item[0] == "image":
                parts.append(f"![{Path(item[1]).stem.replace('_', ' ')}](../figures/{item[1]})")
            elif item[0] == "image_pair":
                parts.extend([f"![{Path(name).stem.replace('_', ' ')}](../figures/{name})" for name in item[1:]])
            parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def _pdf(sections, path, figures, experiment_id):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                   Spacer, Table, TableStyle)

    ink, teal, amber = colors.HexColor("#182536"), colors.HexColor("#23657a"), colors.HexColor("#9b5d1f")
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=23, leading=27,
                                textColor=ink, spaceAfter=10),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Bold", fontSize=9, leading=12,
                                   textColor=teal, spaceAfter=14),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.1,
                               textColor=ink, spaceAfter=9),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.7, leading=11.7,
                                textColor=ink, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.3, leading=10.7,
                               textColor=ink, alignment=TA_LEFT),
        "head": ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8.3, leading=10.7,
                               textColor=colors.white),
        "callout": ParagraphStyle("callout", fontName="Helvetica-Bold", fontSize=9.2, leading=13,
                                  textColor=amber),
    }

    def paragraph(text, style="body"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(headers, rows, widths):
        content = [[paragraph(cell, "head") for cell in headers]]
        content += [[paragraph(cell, "cell") for cell in row] for row in rows]
        value = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        value.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), teal),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eff4f6"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), .5, teal),
        ]))
        return value

    def image_file(name, width, max_height):
        from PIL import Image as PILImage
        with PILImage.open(figures / name) as source:
            aspect = source.height / source.width
        height = width * aspect
        if height > max_height:
            width *= max_height / height
            height = max_height
        return Image(str(figures / name), width=width, height=height, hAlign="CENTER")

    story = []
    for index, section in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.extend([paragraph(section["title"], "title"), paragraph(section["subtitle"], "subtitle")])
        for item in section["items"]:
            kind = item[0]
            if kind in {"p", "small"}:
                story.append(paragraph(item[1], "body" if kind == "p" else "small"))
            elif kind == "callout":
                box = Table([[paragraph(item[1], "callout")]], colWidths=[505])
                box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e7")),
                                         ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#e5c399")),
                                         ("TOPPADDING", (0, 0), (-1, -1), 10),
                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                         ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
                story.extend([box, Spacer(1, 11)])
            elif kind == "table":
                story.extend([table(item[1], item[2], item[3]), Spacer(1, 11)])
            elif kind == "image":
                story.extend([image_file(item[1], item[2], item[3]), Spacer(1, 9)])
            elif kind == "image_pair":
                pair = Table([[image_file(item[1], 248, 178), image_file(item[2], 248, 178)]],
                             colWidths=[252.5, 252.5])
                pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                          ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                story.extend([pair, Spacer(1, 9)])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c8d4da"))
        canvas.line(45, 35, A4[0] - 45, 35)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#526675"))
        canvas.drawString(45, 23, "APP STORE ML | ENDTERM | SEPARATE SNAPSHOT")
        canvas.drawRightString(A4[0] - 45, 23, f"{experiment_id} | {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=45, leftMargin=45,
                                 topMargin=43, bottomMargin=47, title="App Store ML - Endterm Report",
                                 author="Project team; prepared with disclosed Codex assistance")
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def build_deliverables(public_dir):
    """Create the report and plots after the real experiment's aggregates exist."""
    public_dir = Path(public_dir)
    figures = public_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sections, comparison, learning, provenance = _sections(public_dir)
    _plots(comparison, learning, figures)
    _write_markdown(sections, public_dir / "reports/results.md")
    path = public_dir / "Endterm_Report.pdf"
    _pdf(sections, path, figures, provenance["experiment_id"])
    print(f"Created English Endterm report: {path}", flush=True)
    return path
