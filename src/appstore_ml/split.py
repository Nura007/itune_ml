import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import FEATURES, sha256, write_csv, write_json


def dataset_fingerprint(df):
    columns = ["trackId", "high_rated", "_collected_at"] + FEATURES
    stable = df[columns].sort_values("trackId").to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(stable.encode()).hexdigest()


def fixed_split(df, directory, config):
    """Reuse an identical manifest; fail instead of silently reassigning the test set."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(df)
    spec = {"dataset_sha256": fingerprint, "test_size": config["test_size"],
            "random_state": config["random_state"], "stratify": "high_rated"}
    metadata = directory / "metadata.json"
    manifest_path = directory / "split.csv"
    if metadata.exists() != manifest_path.exists():
        raise ValueError("Incomplete split directory; restore both files.")
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        saved_manifest_hash = saved.pop("manifest_sha256", None)
        if saved != spec:
            raise ValueError("Dataset/protocol changed. Use a new run directory; never overwrite the holdout.")
        if saved_manifest_hash != sha256(manifest_path):
            raise ValueError("Split manifest checksum mismatch.")
        manifest = pd.read_csv(manifest_path)
    else:
        if df.high_rated.nunique() != 2 or df.high_rated.value_counts().min() < 7:
            raise ValueError("Need both classes and enough minority examples for stratified 5-fold CV.")
        train_ids, test_ids = train_test_split(
            df.trackId, test_size=config["test_size"], random_state=config["random_state"],
            stratify=df.high_rated
        )
        manifest = pd.DataFrame({"trackId": list(train_ids) + list(test_ids),
                                "split": ["train"] * len(train_ids) + ["test"] * len(test_ids)})
        manifest = manifest.sort_values("trackId").reset_index(drop=True)
        write_csv(manifest, manifest_path)
        write_json(metadata, {**spec, "manifest_sha256": sha256(manifest_path)})
    if manifest.trackId.duplicated().any() or set(manifest.trackId) != set(df.trackId):
        raise ValueError("Corrupt split manifest: IDs differ from dataset.")
    if set(manifest["split"]) != {"train", "test"}:
        raise ValueError("Corrupt split labels.")
    indexed = df.set_index("trackId", drop=False)
    train = indexed.loc[manifest.loc[manifest["split"].eq("train"), "trackId"]].reset_index(drop=True)
    test = indexed.loc[manifest.loc[manifest["split"].eq("test"), "trackId"]].reset_index(drop=True)
    if train.high_rated.value_counts().min() < config["cv_folds"]:
        raise ValueError("Not enough training examples of each class for CV.")
    return train, test, spec
