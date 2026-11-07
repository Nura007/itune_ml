# Change log: Midterm to Endterm

This log links observed weaknesses and the verified Endterm rubric to implemented changes.
It does not invent instructor feedback. Historical Midterm outputs remain unchanged.

| Evidence or requirement | Before | Endterm fix and check |
| --- | --- | --- |
| Private cloud artifacts were lost | Original data and split cannot be reconstructed | Separate real API snapshot retained locally with raw SHA-256, saved split checksum and a private submission ZIP; original continuity remains unresolved |
| Only fixed basic models existed | Dummy, tree, KNN and SVM | All four rerun on new snapshot, plus nested-CV forest and boosting; compare in one table |
| Midterm SVM detected only 7/76 lower-rated apps | High positive F1 obscured class-0 errors | Specificity, balanced accuracy, all-model matrices, paired errors and real anonymized examples |
| No hyperparameter search | Fixed settings only | Inner three-fold tuning inside identical outer five-fold training CV; candidate settings and scores retained |
| No structure analysis | Training EDA only | Train-only PCA, variance/loadings/profiles and fold-local SVM+PCA ablation |
| Earlier notes deferred MLP to Final | No neural model | Verified Endterm rubric places MLP here; fixed network and isolated training/validation learning diagnostics |
| Midterm presentation was planned for nine minutes | Nine-slide Midterm deck | New timed seven-minute Endterm PDF presentation with three proposed speaker roles, subject to instructor acceptance of team size |
| Notebook displayed saved aggregates only | No training on Restart and Run All | Endterm notebook executes the frozen-snapshot experiment and regenerates its report |
| Unknown human contributions | Names and work unverified | Owner supplied three names and SE-2423; explicit AI disclosure and defense guide provided; actual contribution records remain unfinished |

Before/after metrics are fair only within the new snapshot. Differences from October's
historical scores mix data change and method change and are not claimed as model improvements.

The owner identified **Nurtilek Koblandy, Erkebulan Zhuparbay and Mansur Ryskali**, group
**SE-2423**. Instructor acceptance of a three-person team is unverified because the rubric
describes two partners. Proposed future speaking roles are Nurtilek, slides 1-3 (130 seconds);
Erkebulan, slides 4-6 (155 seconds); and Mansur, slides 7-9 (135 seconds). This 420-second
plan does not establish actual contributions or completed rehearsal.
