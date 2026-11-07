# Endterm protocol

Specification recorded before the Endterm model run on November 8, 2026 (Asia/Qyzylorda).
Source: all four pages of the supplied Endterm Rubric.pdf, read directly in this session.
The owner's request changes the previous MLP deferral: ensembles, PCA and MLP belong here.
Text/image modeling and Streamlit remain for Final.

## Continuity and experiment identity

The original October 8 Midterm snapshot and split are absent. Its public fingerprint and
aggregates cannot reconstruct app records or test IDs. The new run is a separately reported
experiment, `endterm-2026-11-08`, with all four Midterm baselines refitted on its own frozen
80/20 split. It does not satisfy the rubric's original-data-and-split requirement.
No instructor exception is claimed. Restore the original private archive to resolve that gap.
The Endterm command requires `--new-experiment` for a differing fingerprint. Collection is
never triggered by training or a notebook rerun. Saved raw checksums and split hashes are verified.

## Data and leakage controls

Keep the original US/USD eligibility (at least 50 ratings), target (rating at least 4.5),
seven-feature whitelist, deduplication, seed 42 and fixed stratified 80/20 split.
Exactly five numeric inputs and two categorical inputs are used. Ratings, rating counts,
identifiers, developer identity, names and descriptions never enter any estimator.
Median imputation and standard scaling apply to numeric inputs. Categorical inputs receive
constant imputation and one-hot encoding with unknown categories ignored. Every transform
is fitted on the relevant training fold only. PCA operates on these transformed inputs,
so its components are derived features, not new external model inputs.

## Comparison and tuning

Retain majority Dummy, depth-8/leaf-10 Decision Tree, distance-weighted KNN(k=15), and RBF
SVM(C=1, gamma=scale). Tune random forest and gradient boosting over explicit small grids
recorded by `endterm_training.py` and saved with the results. Depth/leaf size control forest
complexity; boosting rate, rounds and shallow-tree depth control its bias/variance trade-off.
Five identical outer training folds compare every method. Three inner stratified folds
tune each ensemble inside each outer training fold. A final three-fold training-only search
selects settings for refitting on all training rows. Inner search scores are not outer CV scores.
Use a fixed SVM+PCA(90% variance) ablation, with PCA fitted inside every training fold.

Primary metric remains positive-class F1 for continuity of the research question. The positive
class is common, so high F1 can be achieved by a majority predictor. Report ROC-AUC, precision,
recall, balanced accuracy, class-0 recall and complete confusion matrices to expose that weakness.
Select the overall model and best earlier (non-MLP) model by outer CV mean F1, ties by model name.
CV spread is sample SD, not a confidence interval. No significance claim and no threshold tuning.
Only after freezing selection, evaluate all methods on the same held-out rows.

## MLP learning check

Fixed architecture: two hidden layers (32,16), ReLU, Adam, learning rate 0.001,
L2 alpha 0.001, batch size up to 64, 120 epochs, seed 42. Numeric scaling is essential
for gradient optimization. A separate 80/20 split inside training supplies learning curves.
Its preprocessing is fitted on the diagnostic training subset, excluding validation rows.
Plot training and validation log loss and F1. Record the validation-loss minimum and final gap,
and discuss convergence/overfitting without changing settings in response to holdout results.
The fixed 120 epochs are used in the outer comparison. A loss minimum is diagnostic, not a
post hoc early-stopping claim. Compare the MLP to the best earlier method chosen by CV.

## Unsupervised analysis and errors

Fit descriptive PCA on training rows only. Standardized numeric inputs coexist with unscaled
one-hot indicators, an explicit weighting choice. Report explained variance, component loadings,
coarse examples and aggregate feature profiles. A two-dimensional density map is not proof of
natural groups. Connect structure to prediction through the fold-local SVM+PCA ablation.
Show real false positives and false negatives across every model, with boundary/feature summaries,
paired error changes versus SVM, and all-model confusion matrices. Public examples are anonymized.
Keep exact raw values, IDs, detailed predictions, app-level PCA scores and model files private.

## Retention and deliverables

Keep the new run under ignored `work/endterm-2026-11-08/`. The private submission ZIP contains
the raw/cleaned snapshot, saved split and checksums, models, predictions, source, notebook and
public report. Never publish this ZIP or private records to Git. Use the publication checker
for aggregate outputs. Preserve the historical Midterm outputs in place.
Provide an English report, seven-minute PDF slide deck, timed notes for two speakers,
run instructions, change log, criterion evidence matrix and defense questions. Student names,
group, actual contributions, rehearsal and individual defense remain human responsibilities.
