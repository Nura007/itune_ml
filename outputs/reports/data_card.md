# Data card
Status: real_api. Source: Apple iTunes Search API, US storefront; software only.
Collection: 2026-10-07T20:53:06.749227+00:00 to 2026-10-07T20:53:58.426835+00:00 (UTC).
Instructor conditions are recorded in config/source_approval.json.
Apple describes API content as promotional. The API documentation is not evidence that
robots.txt permits collection: /search* is disallowed there. Instructor approval is course approval,
not a statement about Apple's permission. Raw responses and descriptions must not be published.

## Sampling and provenance
132 interleaved, prespecified keyword queries; pilot uses fitness, education, puzzle game.
Main responses are limited to 200 each; minimum 3.2 seconds between request starts, retries included.
Stop after reaching at least 2,000 eligible distinct apps; require at least 1,500 to train.
Raw bytes, request parameters, timestamp and SHA-256 are stored under ignored data/raw.
Search results overlap and are ranking-biased. The dataset is not representative of all apps.

## Cleaning audit
{
  "raw_rows": 2763,
  "excluded": {
    "invalid_track_id": 0,
    "invalid_collection_time": 0,
    "not_software": 0,
    "duplicate_track_id": 276,
    "non_usd_or_missing_currency": 0,
    "invalid_or_below_50_ratings": 458,
    "invalid_or_missing_rating": 0
  },
  "clean_rows": 2029,
  "class_counts": {
    "0": 378,
    "1": 1651
  },
  "missing_features": {
    "price_usd": 0,
    "size_mb": 0,
    "language_count": 0,
    "release_year": 0,
    "update_recency_days": 0,
    "primary_genre": 0,
    "content_advisory_rating": 0
  },
  "missing_description": 0,
  "collection_start": "2026-10-07T20:53:06.749227+00:00",
  "collection_end": "2026-10-07T20:53:58.426835+00:00",
  "minimum_met": true
}

## Decisions
1. Require positive integer trackId, a valid collection timestamp and kind=software.
2. Keep the latest complete observation per trackId (stable source-order tie-break); never mix fields across times.
3. Keep USD rows; do not assume a foreign or unknown currency is dollars.
4. Require an integer userRatingCount >=50 and a finite averageUserRating in [0,5].
5. Define high_rated = int(averageUserRating >=4.5); exactly 4.5 belongs to class 1.
6. Negative price, nonpositive size and malformed values become missing features.
7. fileSizeBytes / 1,000,000 gives decimal MB. Languages are distinct normalized valid codes;
   an explicit empty list means zero, an absent list means missing.
8. Release year before 2008 or after collection is missing. Future updates, and updates preceding
   a valid release, are missing. Recency uses full elapsed days at each row's collection timestamp.
9. Preserve missing features for fold-local imputation; do not fill with global medians.
10. Descriptions remain private for Final. Models see only price_usd, size_mb, language_count, release_year, update_recency_days, primary_genre, content_advisory_rating.

## Target leakage
averageUserRating forms the label. userRatingCount only filters eligibility.
Neither is a feature. No current-version rating, ID, app name, description or developer ID enters the model.
contentAdvisoryRating is an allowed age-content category, not a user rating.

## Split and storage
data/splits/split.csv preserves track IDs and train/test assignments privately.
The fingerprint detects data changes. Recollection requires a separate run directory and a new reported experiment.
Public outputs are aggregate metrics/figures and anonymized diagnostic cases.
Local runs retain original JSON and models in ignored directories.
Cloud runs retain raw files only for their runner lifetime; their public artifact excludes raw data,
app-level cleaned data, split IDs, models and private predictions. Run locally to keep a private raw archive.
Do not upload these excluded files to this public repository.

Sources: https://performance-partners.apple.com/search-api; https://itunes.apple.com/robots.txt; https://scikit-learn.org/stable/modules/cross_validation.html.
