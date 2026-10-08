import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUMERIC = ["price_usd", "size_mb", "language_count", "release_year", "update_recency_days"]
CATEGORICAL = ["primary_genre", "content_advisory_rating"]
FEATURES = NUMERIC + CATEGORICAL


def settings(path=None):
    config = json.loads(Path(path or ROOT / "config/settings.json").read_text(encoding="utf-8"))
    if config["numeric_features"] != NUMERIC or config["categorical_features"] != CATEGORICAL:
        raise ValueError("This protocol requires exactly the seven prespecified input features.")
    if config["country"] != "us" or config["currency"] != "USD":
        raise ValueError("This study is prespecified for the US storefront and USD.")
    return config


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
