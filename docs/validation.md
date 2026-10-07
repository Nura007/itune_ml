# Validation record — 2026-10-08 (Asia/Qyzylorda)

## Executed checks

- Real collection/training run: https://github.com/Nura007/itune_ml/actions/runs/37685277008 — successful.
- Final source checks: https://github.com/Nura007/itune_ml/actions/runs/37686302002 — Ruff and all 10 tests passed.
- Report-only refresh: https://github.com/Nura007/itune_ml/actions/runs/37686294665 — successful; no recollection/retraining.
- The refreshed notebook executed successfully.
- LibreOffice exported the refreshed presentation to PDF. Nine rendered pages were verified.
- The refreshed contact sheet was visually inspected: the metric table fits on the slide, and error examples include FP and FN.
- Public Git tree inspected: no raw JSON, descriptions, row-level datasets, split IDs, model dumps or private predictions.
- Core metrics in README, report, notebook and slides agree with the generated CSVs.

## Observed experiment

2,763 returned rows minus 276 duplicate records and 458 ineligible count records = 2,029 eligible apps.
Train/test: 1,623 / 406. Test classes: 330 positive and 76 negative.
SVM selected by CV F1: test F1 0.900826, ROC-AUC 0.619976, confusion matrix [[7,69],[3,327]].
Dummy test F1: 0.896739. The small improvement is described without a significance claim.

## Limitations still requiring local access

The cloud raw-data/model archive is not retained after runner deletion and was not published.
For a durable private archive, run the documented local workflow.
The referenced Midterm Rubric.docx and Project Guide.pdf remain unread because the local runtime
fails to start with "setup refresh had errors". The implemented requirement matrix follows the explicit user request.
