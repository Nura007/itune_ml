# Endterm criterion evidence

Source: Endterm Rubric.pdf, all four pages directly read on November 8, 2026.
This is an evidence checklist, not a predicted grade or a claim of instructor approval.

| Criterion | Weight | Evidence | Status and limits |
| --- | --- | --- | --- |
| 1. Continuity and fixes | 10 | docs/endterm_changes.md; reports/continuity.json; private data and saved split; executed notebook | PARTIAL: original Midterm data/split lost. New real experiment retained and explicitly separated. Same-data requirement unresolved |
| 2. Tuned ensembles | 15 | reports/tuning_results.csv, model_selection.json, cv_metrics.csv; endterm_training.py | Implemented and run: forest and boosting, eight candidates each, train-only nested CV, full-training refit |
| 3. Unsupervised analysis | 10 | reports/unsupervised.json; PCA variance/loadings/profiles; figures/pca_*.png | Implemented and interpreted: nine components retain 90%; first two 36.6%; fold-local PCA ablation compares with SVM |
| 4. MLP comparison and learning check | 15 | reports/mlp_learning_curve.csv, mlp_diagnostics.json; figures/mlp_learning.png | Implemented and run: fixed (32,16) MLP, isolated validation, overfitting discussion; compared with best earlier CV model |
| 5. Fair comparison and errors | 15 | reports/comparison.csv, error_cases_anonymized.csv, paired_errors.csv; confusion matrices | Eight models on identical new split; original split comparison remains unavailable; CV SD and all required test metrics included |
| 6. Conclusions, limitations, Final plan | 10 | Endterm_Report.pdf; reports/results.md | Evidence-based conclusion and permission-checked description-text proposal linked by private IDs; source permission not invented |
| 7. Seven-minute presentation | 10 | Endterm_Presentation.pdf; speaker_notes.md | Nine slides and 420-second proposed two-speaker plan prepared. Students must rehearse and actually both speak |
| 8. Individual answers and contributions | 15 each | docs/endterm_defense.md; docs/contributions.md | AI use disclosed and preparation guide supplied. Real names, group, contributions and independent oral answers remain human tasks |

## Submission checklist

- Review the report's continuity limitation before submission; no exception has been obtained.
- Supply both students' names, group and actual contributions. Do not claim unperformed work.
- Rehearse the seven-minute presentation with both speakers, then five minutes of questions.
- Keep the private archive private: it includes raw data, split IDs, predictions and models.
- Both partners upload the same reviewed ZIP to Moodle with the required
  `Endterm_Surname_Name_Group.zip` naming pattern, before 23:59 on the day before the defense.
- The defense date has not been supplied, so no calendar deadline is invented.
