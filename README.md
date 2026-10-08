# App Store ML — Midterm

A binary classification project for **existing US App Store apps with at least 50 user ratings**:

- **Class 1:** `averageUserRating >= 4.5`.
- **Class 0:** `averageUserRating < 4.5`.

**Research question:** Can seven app metadata attributes distinguish highly rated apps from
lower-rated existing US App Store apps with at least 50 user ratings at collection time?

This experiment evaluates apps that already have user ratings. It does not establish that
metadata can predict an app's success before launch. This repository implements the Midterm stage;
neural networks, text features, and a Streamlit app are outside the current implementation.

## Recorded experiment

Real API collection took place on **October 8, 2026, in Asia/Qyzylorda**
(October 7, 2026, in UTC). The cleaned dataset contained **2,029 apps**:
1,651 in class 1 and 378 in class 0. The train/test split contained **1,623/406 apps**.

The collector completed 3 pilot requests and 14 main requests from a planned list of 132
search terms, then stopped after reaching the target sample size.

| Model | CV F1 ± SD | Test F1 | Test ROC-AUC |
| --- | --- | --- | --- |
| Dummy | 0.8974 ± 0.0009 | 0.8967 | 0.5000 |
| Decision Tree | 0.8779 ± 0.0068 | 0.8883 | 0.6777 |
| KNN | 0.8960 ± 0.0085 | 0.8964 | 0.6386 |
| SVM | 0.8989 ± 0.0027 | 0.9008 | 0.6200 |

SVM was selected by mean cross-validation F1 for class 1. Its test F1 improvement over Dummy
was only **+0.0041**. It correctly identified just **7 of the 76 class-0 apps**, with a balanced
accuracy of 0.5415. The high positive-class F1 partly reflects class imbalance and does not
demonstrate strong discrimination between both classes. No statistical significance is claimed.

The values above come from `outputs/reports/*.csv`, not from synthetic test fixtures.
See the [completed experiment in GitHub Actions](https://github.com/Nura007/itune_ml/actions/runs/37685277008)
and the [results report](outputs/reports/results.md).

## Implemented workflow

- Pilot collection using three search terms, followed by main collection from a list of 132 terms.
- US storefront, USD prices, up to 200 results per request, a minimum 3.2-second request interval,
  caching, and bounded retries.
- Raw JSON, request parameters, collection timestamps, and SHA-256 checksums written to the
  ignored `data/raw/` directory on the machine running collection.
- Deduplication by `trackId`, the minimum-50-ratings filter, numeric and date validation,
  and a cleaning audit.
- Seven explicitly allowed model features, with rating fields excluded.
- A stratified 80/20 train/test split with seed 42 and a saved split manifest.
- Five-fold stratified cross-validation within training data.
- DummyClassifier, Decision Tree, KNN, and SVM, with preprocessing inside each model's Pipeline.
- Class-1 F1, ROC-AUC, precision, recall, and balanced accuracy; CV means and standard deviations,
  followed by evaluation on the fixed test set.
- Confusion matrices, ROC curves, training-only exploratory analysis, anonymized real error cases,
  and a data card.
- An executed results notebook and nine English slides with speaker notes for approximately
  nine minutes of presentation.
- Automated protocol tests and a separate workflow for the real experiment.

The code supports saving and reusing a split. The original cloud run's private dataset and
split files were not retained after the runner was removed; see the storage limitation below.

## Features and target leakage

| Feature | Meaning |
| --- | --- |
| `price_usd` | App price in US dollars |
| `size_mb` | App size in decimal megabytes |
| `primary_genre` | Primary app genre |
| `content_advisory_rating` | Age-based content category |
| `language_count` | Number of distinct supported languages |
| `release_year` | Original release year |
| `update_recency_days` | Full days since the latest update at collection time |

`averageUserRating` creates the target, while `userRatingCount` is used only for the
eligibility filter. Neither is passed to a model. Other user-rating fields, app IDs, names,
developer IDs, and descriptions are also excluded.

`contentAdvisoryRating` is allowed because it describes age suitability, not user satisfaction.
Descriptions were collected for the later text-analysis stage but are not used as Midterm inputs.

Numeric imputation and scaling, and categorical imputation and one-hot encoding, are fitted
inside the Pipeline on each training fold. Test data do not influence preprocessing, model
selection, or decision thresholds. Hyperparameters are fixed; no hyperparameter or threshold
search was performed.

## Source and instructor conditions

The repository owner relayed the instructor's conditions: pause between requests, use a search-term
list, remove duplicate apps, exclude rating fields from model inputs, do not publish raw data,
and document Apple's promotional-content context. This record is in
[config/source_approval.json](config/source_approval.json).

[Apple's API documentation](https://performance-partners.apple.com/search-api) describes the
Search API. The project's source check recorded `Disallow: /search*` in
[iTunes robots.txt](https://itunes.apple.com/robots.txt).
**We do not claim that robots.txt permits collection.** Instructor approval is course approval;
it is not permission from Apple and does not override the source's terms.

Do not add raw JSON, descriptions, app-level tables, split IDs, detailed predictions, or model
files to this public repository. Automated publication uses an explicit allowlist of aggregate
outputs.

## Local setup and collection

Use Python 3.11. Clone the repository and work on `main`:

```bash
git clone https://github.com/Nura007/itune_ml.git
cd itune_ml
git switch main
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies, run the tests, and collect the pilot:

```bash
python -m pip install -e ".[dev]"
pytest
python -m appstore_ml.collect pilot
```

Inspect `outputs/reports/pilot_audit.json` for returned fields, missing values, duplicates,
and the number of eligible apps. After reviewing the pilot, run:

```bash
python -m appstore_ml.collect main
python -m appstore_ml.prepare
python -m appstore_ml.run
```

Main collection stops after reaching **at least 2,000 eligible apps**. If fewer than
**1,500 apps** remain after all queries, training stops with an explanation.
The pipeline does not silently relax the research criteria.

Repeated collection uses the same cache. A changed dataset cannot silently replace an existing
test split. Use a separate run directory for a new snapshot:

```bash
python -m appstore_ml.collect pilot --raw-dir work/run2/data/raw --report-dir work/run2/outputs/reports
python -m appstore_ml.collect main --raw-dir work/run2/data/raw --report-dir work/run2/outputs/reports
python -m appstore_ml.run --run-dir work/run2
```

Do not run two collectors simultaneously. Keep a private backup of `data/` and
`outputs/models/`; leave the Git ignore rules in place.

## Results and presentation

- [Results and conclusions](outputs/reports/results.md)
- [Data card](outputs/reports/data_card.md)
- [Cross-validation metrics](outputs/reports/cv_metrics.csv) and [test metrics](outputs/reports/test_metrics.csv)
- [Notebook](notebooks/midterm.ipynb)
- [PowerPoint](outputs/midterm.pptx), [PDF](outputs/midterm.pdf), and [slide preview](outputs/slides_contact.png)
- [Provenance](outputs/reports/provenance.json): source commit, package versions, collection dates,
  split sizes, and dataset fingerprint

The notebook displays generated reports and figures. To rerun its cells, keep it with the
repository's `outputs/` directory; model training is performed by the Python pipeline.

The cloud workflow generates the PDF and slide preview using LibreOffice. The local Python
pipeline creates a PPTX; open it in PowerPoint or LibreOffice to export a PDF.

Synthetic data are used only in tests and temporary pytest directories. They are not the
research dataset and do not contribute to the reported empirical results.

## GitHub Actions

`Protocol tests` checks filtering boundaries, leakage prevention, split consistency, and the
end-to-end pipeline using an **explicitly labeled synthetic fixture**, without real API collection.

`Approved Midterm experiment` runs when `config/collection_run.json` changes:

- `stage: pilot` runs pilot collection.
- `stage: full` runs the pilot and the main experiment.
- `stage: report` refreshes presentation outputs from saved aggregate results without recollection
  or retraining.

The experiment runs only after tests pass. Aggregate results are committed to the same branch.

## Data retention and reproducibility limitation

**The original cloud run did not retain its raw JSON, cleaned app-level dataset, split manifest,
detailed predictions, or fitted models after the temporary runner was removed.** These files
were deliberately excluded from public artifacts, but no persistent private archive was created.

The repository retains code, logs, aggregate results, and fingerprints. These do not reconstruct
the original dataset or identify the original test rows. The requirement to retain the raw
snapshot and reuse its exact split is therefore not yet fulfilled.

A local run can retain these private files. If the original snapshot cannot be recovered,
recollection must be reported as a new experiment: rerun all baselines on that snapshot and
preserve its data and split. Results from a new snapshot must not be presented as improvements
measured on the original test set.

## Repository structure

```text
config/                     Protocol settings, 132 search terms, source approval record
src/appstore_ml/             Collection, preparation, features, split, models, training, evaluation, reporting
tests/                      Protocol checks without real API collection
scripts/                    Cloud orchestration and publication checks
notebooks/midterm.ipynb      Research explanation and generated results
data/{raw,processed,splits}  Private data created during a run; ignored by Git
outputs/reports/             Public aggregate tables, data card, and conclusions
outputs/figures/             Research figures
outputs/{private,models}/    Private predictions and fitted models; ignored by Git
docs/                       Protocol, requirement notes, and contributions
app/                        Placeholder for the Final application
```

## Course requirements and remaining work

Implementation initially followed the detailed protocol supplied in the project request because
local access to `Midterm Rubric.docx` and `Project Guide.pdf` was unavailable.
Both documents were subsequently read on October 8, 2026, during defense preparation.

That review identified requirements that remain open:

- Retain the raw and cleaned dataset and the exact split in a persistent private archive.
  The original cloud run did not preserve them.
- Complete [the contribution record](docs/contributions.md) with the actual team members and
  their work, including disclosure of AI assistance.
- Align the next-stage plan with the guide: ensembles and hyperparameter tuning, unsupervised
  analysis such as clustering/PCA, and a neural network belong to **Endterm**. Text or image
  features and a local model application belong to **Final**.

The existing slides and some earlier documents still defer the neural network to Final;
their schedule needs updating. The [requirements matrix](docs/requirements.md) also retains
the earlier statement that the source documents had not been read. This README records the
later review; full rubric compliance is not claimed.
