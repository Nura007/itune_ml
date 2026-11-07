# Seven-minute speaker notes

Team: Nurtilek Koblandy; Erkebulan Zhuparbay; Mansur Ryskali. Group: SE-2423.

These are assigned responsibilities and proposed speaking roles, not completed-work claims. Total planned time: 420 seconds. Every student must understand all slides. Allow five additional minutes for questions. The rubric describes two partners; instructor acceptance of the supplied three-person team is unverified.

| Student | Assigned responsibility | Slides | Seconds |
| --- | --- | --- | --- |
| Nurtilek Koblandy | Data preparation, provenance, leakage checks and baseline models | 1-3 | 130 |
| Erkebulan Zhuparbay | Ensemble tuning, model comparison and PCA analysis | 4-6 | 155 |
| Mansur Ryskali | MLP learning checks, error analysis, conclusions and Final plan | 7-9 | 135 |

## 1. Question and main result

Speaker: Nurtilek Koblandy; 35 seconds; 0:00-0:35

Our question is whether seven metadata attributes can identify highly rated existing US App Store apps. Highly rated means at least 4.5 stars, and every app has at least 50 ratings. We compared eight methods. Training cross-validation selected SVM. Its test F1 was 0.887, but class-zero recall was only 8.0%. That second number matters because the positive class dominates this sample.

## 2. Data and continuity

Speaker: Nurtilek Koblandy; 45 seconds; 0:35-1:20

The original cloud run retained public results but lost its private app records and split manifest. A checksum cannot reconstruct those missing rows. We therefore state a continuity gap rather than claiming a complete continuation. The separately collected November dataset has 2,179 eligible apps, split into 1,743 training and 436 test rows. All baselines and new models use this same new split. Raw response checksums, split IDs and models are now retained in the private submission. Restoring the original archive is still needed to meet the rubric's same-data condition. No instructor exception has been assumed.

## 3. Fair evaluation

Speaker: Nurtilek Koblandy; 50 seconds; 1:20-2:10

We kept the original seven raw inputs. User ratings define the label, and rating counts define eligibility, but neither enters a model. Identifiers and text are also excluded. Imputation, scaling and one-hot encoding are fitted within training folds. Every method uses the same five outer folds. For each ensemble, a three-fold inner search chooses parameters without seeing the outer validation rows. A final search on all training rows produces the fitted ensemble. Positive-class F1 remains primary, while AUC, balanced accuracy and class-zero recall check discrimination. The test set is used only after selection, and thresholds are unchanged.

## 4. Complete comparison

Speaker: Erkebulan Zhuparbay; 50 seconds; 2:10-3:00

The table compares the majority baseline, three earlier classifiers, two tuned ensembles, a PCA ablation and the MLP. SVM has the highest mean outer CV F1 and is marked with an asterisk. Its test F1 difference from Dummy is -0.0006. A high F1 alone is not strong evidence of useful classification: the majority model receives a high value while finding no lower-rated apps. The spread shown here is the sample standard deviation across five folds, not a confidence interval. We did not change the winner after inspecting test AUC or balanced accuracy.

## 5. Tuned ensembles

Speaker: Erkebulan Zhuparbay; 55 seconds; 3:00-3:55

Both ensembles use eight prespecified parameter combinations. The random forest always uses 100 trees and searches depth, minimum leaf size and the number of candidate features. Larger leaves and restricted depth reduce variance. Boosting searches 75 or 150 rounds, learning rates of 0.03 or 0.1, and depths one or two. This balances correction strength against overfitting. The selected full-training settings are shown here; outer folds may select different settings. Random forest achieved outer CV F1 0.888, and boosting achieved 0.890. These are estimates of the tuning procedures, not the optimistic best scores from the final search.

## 6. Unsupervised structure

Speaker: Erkebulan Zhuparbay; 50 seconds; 3:55-4:45

Training-only PCA requires 9 components to retain 90 percent of the transformed variance. The first two retain 36.6%. The largest PC1 loadings include size_mb, update_recency_days. The report connects signed loadings to median feature profiles and two coarse examples of actual training apps. Numeric attributes are standardized, while categorical indicators remain zero or one, so this is an explicit weighting choice. A density map is not proof of real groups. To test usefulness, we refit PCA inside each SVM training fold. The change in mean CV F1 was -0.0007. PCA maximizes variance, not the ability to distinguish ratings, so compression need not improve prediction.

## 7. Neural network learning

Speaker: Mansur Ryskali; 55 seconds; 4:45-5:40

Our MLP has hidden layers of 32 and 16 ReLU units, trained with Adam at learning rate 0.001 and L2 regularization 0.001. It uses a fixed 120 epochs. The learning curves come from a separate 80/20 split inside the training partition, with preprocessing fitted only on the diagnostic training subset. Validation log loss was lowest at epoch 35 and then increased by 0.067, while training loss fell. This supports an overfitting diagnosis. We did not retrospectively select that epoch. The MLP's CV F1 was 0.875, compared with 0.893 for the CV-selected earlier reference, SVM. More capacity did not automatically solve the metadata limitation.

## 8. Errors and trade-offs

Speaker: Mansur Ryskali; 45 seconds; 5:40-6:25

The selected model identified 7 of 88 lower-rated apps. It made 81 false positives and 6 false negatives. Against SVM, the selected model alone corrected 0 apps while SVM alone corrected 0. The report includes confusion matrices for every model and actual anonymized error examples, with exact records retained privately. Some errors lie near the 4.5 threshold, where a small rating difference changes the label. Coarse metadata can describe these cases but cannot establish why users liked or disliked an app. Different models trade positive recall against recognizing lower-rated apps.

## 9. Conclusions and Final plan

Speaker: Mansur Ryskali; 35 seconds; 6:25-7:00

The answer is cautious: metadata provides limited discrimination in this convenience sample, and high positive-class F1 hides important failures. For Final, the proposed source is retained app descriptions linked privately by trackId. We must confirm permitted use or obtain suitable licensed text before training. We will compare text-only, metadata-only and combined models using fold-fitted TF-IDF and training-only selection. Codex assisted implementation and writing. Every student must confirm their completed contributions, rehearse, and explain the entire project.

## Evidence and citations

All numerical statements come from outputs/endterm/reports; this deck is regenerated by appstore_ml.endterm_slides. The attached Endterm Rubric was read directly (four pages).

- Apple Search API: https://performance-partners.apple.com/search-api
- Source robots policy: https://itunes.apple.com/robots.txt
- Nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- MLP: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca
