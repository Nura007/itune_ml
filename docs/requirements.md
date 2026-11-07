# Requirement traceability

## Verified Endterm source (November 8, 2026)

The supplied `C:/Users/ASUS/Downloads/Endterm Rubric.pdf` was read directly and visually inspected,
all four pages. Its eight criteria total 100 points: 85 shared team points and 15 individual
defense points. The delivered evidence matrix is `outputs/endterm/rubric_checklist.md`.
It requires the original data and split, two tuned ensembles, interpreted unsupervised analysis,
an MLP learning check, fair comparisons and errors, conclusions and a Final source plan,
a seven-minute talk with both students speaking, and truthful contributions/AI disclosure.

The owner's request authorizes these Endterm methods. It does not authorize inventing missing
data continuity, student contributions, approval exceptions or defense performance. The original
private snapshot is missing, so the new API run is explicitly a separate experiment. This remains
a real compliance gap even though all eight models share the new run's frozen split.

Only this Endterm PDF was directly verified in the current task. The owner's existing README
records a later historical review of the earlier Midterm/Project Guide documents. The historical
implementation notes below are retained as such; they are not the verification status of Endterm.

## Historical Midterm implementation notes

Source of requirements: repository owner's explicit chat request, including the instructor's relayed conditions.
The local files Midterm Rubric.docx and Project Guide.pdf were referenced but could not be opened because
the local execution runtime failed before starting. Their contents have NOT been verified.
No additional requirements from those attachments are claimed or inferred.

| Explicit user requirement | Implementation / evidence |
| --- | --- |
| Existing apps with >=50 ratings; target >=4.5 | prepare.py; boundary tests |
| Clarify API/robots discrepancy | README; source_approval.json; data card |
| Respect instructor's rate and publication conditions | collect.py; gitignore; publication allowlist |
| Pilot with different queries | collect pilot; pilot_audit.json |
| Aim 2,000; minimum 1,500 after filtering | settings.json; collection and training guards |
| Raw JSON, code, timestamps, descriptions | private local data/raw; checksummed metadata; descriptions retained |
| Seven allowed features only | common.FEATURES; features.model_inputs; ColumnTransformer |
| No rating/count leakage | explicit column whitelist and tests |
| Fixed stratified 80/20, seed 42 | split.py; fingerprinted manifest |
| Five-fold CV inside train | train.py; fold assignments private, metric rows public |
| Pipeline-local imputation/encoding/scaling | pipelines.py; train-only statistics test |
| Dummy, Decision Tree, KNN, SVM | pipelines.py |
| F1 class 1, ROC-AUC, precision, recall; CV mean/SD; test | train.py; evaluate.py |
| Matrices and actual errors | evaluate.py; anonymous public cases, private audit |
| Notebook, conclusions, 7–9 slides / 7–10 min | run.py; report.py: 9 slides, 9-minute speaker notes |
| NN, NLP and Streamlit for Final | app/README.md |
| Team contribution record | docs/contributions.md; names/hours not invented |

Known delivery limits: a cloud experiment cannot preserve unpublished raw data after its runner is destroyed.
The local command retains those files privately. The supplied source documents still need direct rubric review.
