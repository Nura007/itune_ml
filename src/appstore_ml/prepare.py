import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FEATURES, ROOT, settings, sha256, write_csv, write_json
from .features import build_features, numeric


def read_raw(raw_dir):
    rows = []
    sources = sorted(Path(raw_dir).rglob("*.response.json"))
    if not sources:
        raise ValueError("No raw responses. Run the approved pilot/collection first.")
    for source in sources:
        meta = json.loads(source.with_name(source.name.replace(".response.json", ".meta.json"))
                          .read_text(encoding="utf-8"))
        if sha256(source) != meta["sha256"]:
            raise ValueError(f"Raw response checksum mismatch: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        for row in payload["results"]:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object result in {source}")
            rows.append({**row, "_collected_at": meta["collected_at"],
                         "_query": meta["params"]["term"], "_source_file": source.name})
    return rows


def prepare_records(records, config):
    df = pd.DataFrame(records)
    audit = {"raw_rows": len(df), "excluded": {}}
    for col in ["trackId", "kind", "currency", "averageUserRating", "userRatingCount", "_collected_at"]:
        if col not in df:
            df[col] = None

    def keep(mask, reason):
        nonlocal df
        mask = mask.fillna(False)
        audit["excluded"][reason] = int((~mask).sum())
        df = df.loc[mask].copy()

    df["trackId"] = numeric(df["trackId"])
    keep((df["trackId"] > 0) & (df["trackId"] % 1 == 0), "invalid_track_id")
    df["trackId"] = df["trackId"].astype("int64")
    df["_parsed_collection"] = pd.to_datetime(df["_collected_at"], errors="coerce",
                                            utc=True, format="mixed")
    keep(df["_parsed_collection"].notna(), "invalid_collection_time")
    keep(df["kind"].eq("software"), "not_software")
    df = df.sort_values(["trackId", "_parsed_collection"], kind="stable")
    duplicates = df.duplicated("trackId", keep="last")
    audit["excluded"]["duplicate_track_id"] = int(duplicates.sum())
    df = df.loc[~duplicates].copy()
    keep(df["currency"].eq(config["currency"]), "non_usd_or_missing_currency")
    df["averageUserRating"] = numeric(df["averageUserRating"])
    df["userRatingCount"] = numeric(df["userRatingCount"])
    keep(df["userRatingCount"].notna() & (df["userRatingCount"] % 1 == 0)
         & (df["userRatingCount"] >= config["min_rating_count"]), "invalid_or_below_50_ratings")
    keep(df["averageUserRating"].between(0, 5, inclusive="both"), "invalid_or_missing_rating")
    df["high_rated"] = (df["averageUserRating"] >= config["target_threshold"]).astype(int)
    df = build_features(df).sort_values("trackId").reset_index(drop=True)
    # Keep descriptions for Final, only in ignored private data.
    if "description" not in df:
        df["description"] = None
    df = df.drop(columns=["_parsed_collection"])
    audit["clean_rows"] = len(df)
    audit["class_counts"] = {str(k): int(v) for k, v in df.high_rated.value_counts().sort_index().items()}
    audit["missing_features"] = {k: int(df[k].isna().sum()) for k in FEATURES}
    audit["missing_description"] = int(df.description.isna().sum())
    audit["collection_start"] = str(df["_collected_at"].min()) if len(df) else None
    audit["collection_end"] = str(df["_collected_at"].max()) if len(df) else None
    audit["minimum_met"] = len(df) >= config["min_samples"]
    assert audit["raw_rows"] == audit["clean_rows"] + sum(audit["excluded"].values())
    assert not df["trackId"].duplicated().any()
    assert not np.isinf(df.select_dtypes("number")).any().any()
    return df, audit


def prepare(raw_dir, output_dir, config):
    df, audit = prepare_records(read_raw(raw_dir), config)
    write_csv(df, Path(output_dir) / "apps.csv")
    write_json(Path(output_dir) / "cleaning_audit.json", audit)
    return df, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    _, audit = prepare(args.raw_dir, args.output_dir, settings())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
