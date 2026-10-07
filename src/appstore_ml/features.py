import numpy as np
import pandas as pd

from .common import FEATURES


def numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values))


def build_features(frame):
    """Deterministic row transformations only. Imputation is done inside CV pipelines."""
    df = frame.copy()
    for field in ["price", "fileSizeBytes", "primaryGenreName", "contentAdvisoryRating",
                  "languageCodesISO2A", "releaseDate", "currentVersionReleaseDate"]:
        if field not in df:
            df[field] = None
    price = numeric(df["price"])
    size = numeric(df["fileSizeBytes"])
    df["price_usd"] = price.where(price >= 0)
    # Decimal megabytes, not MiB.
    df["size_mb"] = (size / 1_000_000).where(size > 0)
    for source, target in [("primaryGenreName", "primary_genre"),
                           ("contentAdvisoryRating", "content_advisory_rating")]:
        df[target] = df[source].map(
            lambda x: x.strip() if isinstance(x, str) and x.strip() else np.nan
        )
    df["language_count"] = df["languageCodesISO2A"].map(
        lambda x: len({v.strip().upper() for v in x if isinstance(v, str) and v.strip()})
        if isinstance(x, list) else np.nan
    )
    collected = pd.to_datetime(df["_collected_at"], errors="coerce", utc=True, format="mixed")
    release = pd.to_datetime(df["releaseDate"], errors="coerce", utc=True, format="mixed")
    update = pd.to_datetime(df["currentVersionReleaseDate"], errors="coerce", utc=True, format="mixed")
    valid_release = (release <= collected) & (release.dt.year >= 2008)
    release = release.where(valid_release)
    update = update.where((update <= collected) & (release.isna() | (update >= release)))
    df["release_year"] = release.dt.year.astype(float)
    # Full elapsed days at this row's own collection timestamp; never today's date.
    df["update_recency_days"] = (collected - update).dt.total_seconds().floordiv(86400)
    return df


def model_inputs(df):
    """Whitelist: IDs, descriptions and every user-rating field are excluded."""
    return df.loc[:, FEATURES].copy()
