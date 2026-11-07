# Endterm criterion evidence

Source: Endterm Rubric.pdf, all four pages directly read on November 8, 2026.
This is an evidence checklist, not a predicted grade or a claim of instructor approval.

The owner supplied **Nurtilek Koblandy, Erkebulan Zhuparbay and Mansur Ryskali**, group
**SE-2423**, as a three-person team. The rubric describes two partners; instructor acceptance
of this team size is unverified. Actual contributions remain unprovided.

| Criterion | Weight | Evidence | Status and limits |
| --- | --- | --- | --- |
| 1. Continuity and fixes | 10 | docs/endterm_changes.md; reports/continuity.json; private data and saved split; executed notebook | PARTIAL: original Midterm data/split lost. New real experiment retained and explicitly separated. Same-data requirement unresolved |
| 2. Tuned ensembles | 15 | reports/tuning_results.csv, model_selection.json, cv_metrics.csv; endterm_training.py | Implemented and run: forest and boosting, eight candidates each, train-only nested CV, full-training refit |
| 3. Unsupervised analysis | 10 | reports/unsupervised.json; PCA variance/loadings/profiles; figures/pca_*.png | Implemented and interpreted: nine components retain 90%; first two 36.6%; fold-local PCA ablation compares with SVM |
| 4. MLP comparison and learning check | 15 | reports/mlp_learning_curve.csv, mlp_diagnostics.json; figures/mlp_learning.png | Implemented and run: fixed (32,16) MLP, isolated validation, overfitting discussion; compared with best earlier CV model |
| 5. Fair comparison and errors | 15 | reports/comparison.csv, error_cases_anonymized.csv, paired_errors.csv; confusion matrices | Eight models on identical new split; original split comparison remains unavailable; CV SD and all required test metrics included |
| 6. Conclusions, limitations, Final plan | 10 | Endterm_Report.pdf; reports/results.md | Evidence-based conclusion and permission-checked description-text proposal linked by private IDs; source permission not invented |
| 7. Seven-minute presentation | 10 | Endterm_Presentation.pdf; speaker_notes.md | Nine slides and a 420-second proposed three-speaker plan prepared; instructor acceptance of team size unverified. Rehearsal and actual delivery remain student tasks |
| 8. Individual answers and contributions | 15 each | docs/endterm_defense.md; docs/contributions.md | Three names and SE-2423 supplied; AI use disclosed and preparation guide provided. Actual contributions and independent oral answers remain human tasks |

## Submission checklist

- Review the report's continuity limitation before submission; no exception has been obtained.
- Record each named student's actual contributions. Do not claim unperformed work.
- Confirm instructor acceptance of the three-person team against the rubric's two-partner wording.
- Rehearse the proposed seven-minute presentation: Nurtilek, slides 1-3 (130 seconds);
  Erkebulan, slides 4-6 (155 seconds); Mansur, slides 7-9 (135 seconds). These future roles
  total 420 seconds and do not establish past contributions. Then rehearse five minutes of questions.
- Keep the private archive private: it includes raw data, split IDs, predictions and models.
- The rubric instructs both partners to upload the same reviewed ZIP to Moodle. Confirm how this
  applies to the supplied three-person team, and use the required `Endterm_Surname_Name_Group.zip`
  naming pattern before 23:59 on the day before the defense.
- The defense date has not been supplied, so no calendar deadline is invented.
