# Prespecified Midterm protocol

The US storefront and USD keep price units and rating context consistent. Keyword search is a convenience sample.
The 132 interleaved terms are configuration, not a claim that a supplied starter archive was inspected.

Numeric features: price_usd, size_mb (decimal MB), language_count, release_year, update_recency_days.
Categorical features: primary_genre, content_advisory_rating.
Target: averageUserRating >=4.5. Eligibility: integer userRatingCount >=50.

We preserve the latest complete row per trackId before applying eligibility. No mixing across timestamps.
Feature transforms are deterministic, row-local operations. Missing numeric inputs receive fold-local medians
(or 0 for an entirely missing training column); missing categories receive "Missing". Unseen categories are ignored.
All numeric inputs are standardized within each training fold. No outcome-based feature selection is performed.

Split: stratified 80/20 seed42; ID manifest and full modeling-dataset fingerprint are retained privately.
CV: StratifiedKFold(5, shuffle=True, random_state=42), using identical folds for all models.
Models: majority Dummy; Decision Tree(max_depth=8, min_samples_leaf=10); KNN(k=15, distance weights);
SVC(C=1, RBF, gamma=scale). Fixed settings, no hyperparameter tuning or threshold tuning.
Select among all four by largest mean CV F1(class1), with alphabetical name tie-break.
Report every model on the same holdout after selection. Do not reselect from holdout scores.

Metrics: F1(class1) primary; ROC-AUC (continuous scores), precision(class1), recall(class1);
balanced accuracy supplementary. Fold variation is sample SD, not a confidence interval.
SVM's decision function is not presented as a calibrated probability.
High majority-baseline F1 is a valid potential result and must be stated honestly.

Error cases: up to three FP and three FN from the selected model, ordered by absolute distance to 4.5
and trackId. The public summary anonymizes identifying/exact raw fields. Discuss descriptive associations only.

Follow-up methods for Final: additional fixed-data evaluation, developer-group/time splits, network model,
description text, app interface. Current observations cannot establish causal explanations or pre-launch performance.

References:
- https://performance-partners.apple.com/search-api
- https://itunes.apple.com/robots.txt
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/common_pitfalls.html
