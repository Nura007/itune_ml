from datetime import datetime, timedelta, timezone

import pytest

from appstore_ml.common import settings


@pytest.fixture
def config():
    return settings()


@pytest.fixture
def records():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [{
        "trackId": 100000 + i, "kind": "software", "currency": "USD",
        "trackName": f"SYNTHETIC TEST APP {i}", "averageUserRating": 4.7 if i % 4 else 3.8,
        "userRatingCount": 50 + i, "price": float(i % 5), "fileSizeBytes": str(1_000_000 * (i + 10)),
        "primaryGenreName": ["Games", "Education", "Utilities"][i % 3],
        "contentAdvisoryRating": "4+" if i % 3 else "12+",
        "languageCodesISO2A": ["en", "fr"] if i % 2 else ["en"],
        "releaseDate": f"{2010 + i % 14}-01-01T00:00:00Z",
        "currentVersionReleaseDate": (base - timedelta(days=i % 80)).isoformat(),
        "_collected_at": base.isoformat(), "description": "Synthetic fixture; not App Store data."
    } for i in range(120)]
