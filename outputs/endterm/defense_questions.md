# Endterm defense preparation

Use this guide to explain the submitted work in your own words. Do not memorize
claims that you cannot trace to code and saved outputs. The team has seven minutes
to present and five minutes for questions; either partner may be asked about any
part. A written answer bank cannot satisfy the individual oral-defense criterion.

The report and notebook use a separately collected Endterm snapshot. The original
Midterm raw data and split are missing, and no instructor exception has been
recorded. State this limitation at the start of the talk. Do not call the new
experiment a comparison on the original Midterm test set.

## Evidence to keep open

- `outputs/endterm/reports/comparison.csv`: outer-CV means, sample SD and holdout metrics.
- `outputs/endterm/reports/model_selection.json`: CV-selected model, earlier reference,
  search spaces, selected parameters and convergence warnings.
- `outputs/endterm/reports/tuning_results.csv`: inner search results, separated by
  outer fold and final training-only refit.
- `outputs/endterm/reports/mlp_diagnostics.json` and `mlp_learning_curve.csv`:
  diagnostic learning progress and the fixed epoch budget.
- `outputs/endterm/reports/unsupervised.json`, `pca_variance.csv`,
  `pca_loadings.csv` and `pca_profiles.csv`: PCA interpretation.
- `outputs/endterm/reports/confusion_matrices.json`, `paired_errors.csv`,
  `error_profiles.csv` and `error_cases_anonymized.csv`: error evidence.
- `outputs/endterm/reports/provenance.json` and `continuity.json`: experiment
  identity, dataset and split checksums, and the continuity limitation.

Read numerical answers from these generated files. Do not substitute historical
Midterm values or synthetic test outputs. Public artifacts deliberately omit app
IDs, raw descriptions, exact app-level predictions and model files.

## Questions and answers

**1. What is the research question?**
Can seven metadata attributes distinguish existing US App Store apps with an
average user rating of at least 4.5 from lower-rated apps, among apps with at least
50 ratings? This is classification at collection time, not prediction of a new
app's future success or an explanation of what causes satisfaction.

**2. What changed since Midterm, and what remains unresolved?**
We added nested-CV tuning of random forest and gradient boosting, training-only
PCA with an SVM ablation, an MLP learning diagnostic, and a unified error comparison.
The new run retains private data and the saved split. However, the original
snapshot cannot be recovered from its hashes or aggregate scores, so continuity
with the original Midterm data and split remains unmet.

**3. Why not reconstruct the old split by reusing seed 42?**
A seed reproduces the splitting algorithm only for the same input objects and
ordering. A new API snapshot can contain different apps, ratings and metadata.
Hashes can check bytes; they cannot recover them. We preserve the new split
manifest and checksum instead of claiming it is the missing old split.

**4. What are the seven permitted inputs?**
Price in USD, size in decimal MB, number of supported languages, release year,
days since the latest update at collection time, primary genre, and age-based
content advisory rating. The latter describes age suitability, not satisfaction.
The model-input function explicitly selects these seven columns.

**5. How do you prevent target leakage?**
Average user rating defines the label, while rating count determines eligibility;
neither is an input. IDs, names and descriptions are excluded. Numeric imputers,
scalers, categorical encoders and any PCA step are fitted only on the applicable
training subset. Held-out data do not choose models, parameters or thresholds.

**6. Why keep F1 for the positive class as the primary metric?**
It preserves the stated high-rating classification objective and original metric.
But positive apps are common, so predicting all apps as positive can produce high
F1. We therefore compare with the majority Dummy and also report ROC-AUC, balanced
accuracy, class-0 recall, precision, recall and both rows of each confusion matrix.

**7. How does nested cross-validation work here?**
Five outer folds split only the saved training partition and are identical across
methods. Inside each outer training fold, three inner folds select ensemble
settings. The fitted winner predicts the untouched outer validation fold. These
outer scores estimate the entire tuning procedure. A final training-only search
chooses the settings used to refit each ensemble before holdout evaluation.

**8. Why is the winning inner search score not your comparison score?**
The search selected the candidate partly because it scored well on those inner
folds; that score can be optimistic. The outer folds evaluate the selected
procedure on rows excluded from its inner search. Selecting the best model family
from outer means can still introduce some optimism, which is disclosed.

**9. What do the ensemble search spaces test?**
The random forest varies maximum depth, minimum leaf size and features considered
at a split; it uses 100 trees. These choices control tree complexity and diversity.
Gradient boosting varies learning rate, number of stages and shallow-tree depth.
The grids are deliberately small and recorded before evaluation. Read the winning
settings from the full-training refit record, not a conveniently chosen outer fold.

**10. What does the CV spread mean?**
It is the sample standard deviation of five outer-fold scores, calculated with
`ddof=1`. It describes variation across these folds. It is not a confidence
interval, a significance test, or proof that a small difference generalizes.

**11. What did PCA learn, and why scale first?**
PCA identifies directions with high variance in the transformed training inputs.
Numeric scaling prevents units such as days or MB from dominating merely because
their values are larger. One-hot indicators remain 0/1, so category prevalence
still affects their variance. Explain the actual leading loadings and profiles
from the generated files; their signs are arbitrary.

**12. Do visible areas in the PCA map prove app categories exist?**
No. The map is a two-dimensional density summary and can discard substantial
variation. Explained variance describes reconstruction, not semantic group
validity or predictive quality. We did not claim that PCA discovered true clusters.
The anonymous examples illustrate actual training objects using coarse quartiles.

**13. How does unsupervised analysis connect to prediction?**
The SVM + PCA pipeline retains 90% of training variance and is compared with the
same fixed SVM on identical outer folds and holdout rows. PCA is fitted anew inside
each fold; the full-training exploratory PCA is not reused in CV. Read the actual
CV difference from the comparison table. Less dimensionality need not improve F1,
because predictive information can occur in low-variance directions.

**14. What is the neural-network architecture?**
It is a tabular MLP with two hidden layers of 32 and 16 units, ReLU activation,
Adam, learning rate 0.001, L2 alpha 0.001, batch size 64 and a fixed 120-epoch
budget. The inputs are the same preprocessed metadata as the other methods.
The network does not use text, images or a pretrained language model.

**15. How did you check whether the MLP learned?**
A stratified validation subset is taken from the saved training partition.
Diagnostic preprocessing is fitted only on its fitting subset. Training and
validation log loss and F1 are recorded each epoch. Lower training loss with
worsening validation loss suggests overfitting; a flat validation curve suggests
a plateau. Read the actual curves before describing which pattern occurred.

**16. Did you early-stop at the best validation epoch?**
No. The minimum validation-loss epoch is recorded as a diagnostic. All comparison
fits use the prespecified 120 epochs; the diagnostic does not change their settings.
Reaching the budget, or recording a convergence warning, is not proof of convergence.
Explain the warning record and validation-loss behavior honestly.

**17. What is the best earlier reference for the MLP?**
It is the highest outer-CV F1 method among the non-MLP candidates, including the
Midterm models, tuned ensembles and the fixed PCA ablation. Ties use alphabetical
model name. Read `best_earlier_model` in the selection record. A higher holdout score
for another model does not change the frozen choice.

**18. What do false positives and false negatives mean here?**
A false positive is a lower-rated eligible app predicted to be highly rated; a
false negative is a highly rated app predicted to be lower-rated. Compare the
counts, specificity and recall across methods. The anonymized cases are selected
deterministically near the rating boundary, not cherry-picked. Coarse metadata
observations do not establish why the app received its rating or why a model failed.

**19. Can you claim an improvement, and what is the Final plan?**
Describe only the observed within-snapshot comparison, the baseline difference,
minority-class performance and trade-offs. Do not claim statistical significance
or improvement on the unavailable original holdout. The Final plan is to assess
private app-description text linked by app ID, subject to documented permitted
access and use. Text preprocessing must stay inside training folds; compare
metadata-only, text-only and combined models on the same objects and frozen split.
New modality inputs need an explicit scope update before implementation.

**20. Who did what, and how was AI used?**
Codex assisted with implementation, tests, analysis code and English deliverable
preparation. Each student must state only their actual work and independently
explain the resulting code and conclusions. Do not attribute generated work to a
student who did not perform or review it. AI assistance does not establish that a
student understood the project; the defense tests that directly.

## Two-partner rehearsal

Use the generated slide notes for the exact seven-minute allocation. A practical
split is Speaker 1 for the question, continuity, data and tuning, and Speaker 2 for
PCA, the MLP, comparison, limitations and Final plan. This is a proposed speaking
assignment, not a claim about past contributions. Both should know every section.

1. Each partner independently explains the seven inputs, split protections,
   nested-CV diagram and one confusion matrix without reading this guide.
2. Rehearse the complete presentation with a timer. Reserve the handoff and
   closing sentence within seven minutes; do not read code on slides.
3. Swap sections and repeat the explanation. Ask each other why the Dummy matters,
   why the original split is missing, and why the MLP may not improve performance.
4. Verify every spoken number against the generated comparison and diagnostic
   files. Distinguish current-run results from historical Midterm evidence.
5. Spend five minutes on randomly selected questions above. If unsure, identify
   the relevant evidence instead of inventing an explanation.

Before private submission, both partners must enter their actual names, group and
contributions in `docs/contributions.md`, confirm the final archive contents, and
upload the same complete archive using the required surname/name/group filename.
The defense date and completed rehearsal have not been supplied or verified.

## Contribution and AI disclosure form

Complete this with truthful statements; the blank entries are not submitted facts.

| Entry | Actual information to provide |
| --- | --- |
| Student 1 name and group | Not yet provided by the students |
| Student 1 contribution | Specific work performed, files reviewed and decisions explained |
| Student 2 name and group | Not yet provided by the students |
| Student 2 contribution | Specific work performed, files reviewed and decisions explained |
| AI assistance | Codex: implementation assistance, tests, analysis code and English deliverables |
| Human verification | Record which outputs and methods each student actually checked |

Keep the private snapshot under `work/endterm-2026-11-08/` in the working checkout.
The private submission archive stores it under `private_run/`. Training can be
rerun with the corresponding directory argument; it must not recollect API data.
Use the repository run instructions, and never publish the private archive.
