"""Approved collection only; response bodies remain in ignored data/raw."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .common import ROOT, settings, sha256, write_json
from .prepare import prepare_records, read_raw

ENDPOINT = "https://itunes.apple.com/search"


def check_approval(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ["approved_by", "approved_on", "scope", "evidence"]
    if record.get("status") != "approved" or any(not record.get(k) for k in required):
        raise PermissionError("Record instructor approval and its conditions before collection.")
    return record


class Collector:
    def __init__(self, raw_dir, config, approval, session=None):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "AppStoreML-CourseResearch/0.1"})
        self.last_call = None
        write_json(self.raw_dir / "approval_used.json", approval)

    def fetch(self, term, limit):
        params = {"term": term, "country": self.config["country"], "media": "software",
                  "entity": "software", "limit": limit, "lang": "en_us", "explicit": "Yes"}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        dest = self.raw_dir / (key + ".response.json")
        meta_path = self.raw_dir / (key + ".meta.json")
        if dest.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if sha256(dest) != meta["sha256"]:
                raise ValueError(f"Cached response was modified: {dest}")
            return json.loads(dest.read_text(encoding="utf-8"))
        pause = max(3.2, self.config["request_pause_seconds"])
        for attempt in range(self.config["retries"]):
            delay = pause if self.last_call is None else max(0, pause - (time.monotonic() - self.last_call))
            time.sleep(delay)  # Also separate adjacent pilot/main Collector instances.
            self.last_call = time.monotonic()
            try:
                response = self.session.get(ENDPOINT, params=params, timeout=self.config["timeout_seconds"])
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else pause * 2 ** (attempt + 1)
                    time.sleep(max(wait, pause))
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                    raise ValueError("Unexpected API response schema")
                if payload.get("resultCount") != len(payload["results"]):
                    raise ValueError("resultCount does not match results")
                dest.write_bytes(response.content)
                write_json(meta_path, {
                    "collected_at": datetime.now(timezone.utc).isoformat(), "url": response.url,
                    "params": params, "status_code": response.status_code, "sha256": sha256(dest),
                    "attempt": attempt + 1, "result_count": len(payload["results"])
                })
                return payload
            except requests.HTTPError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise  # No workaround for 401/403 or other permanent refusal.
                if attempt + 1 == self.config["retries"]:
                    raise
            except (requests.ConnectionError, requests.Timeout, ValueError):
                if attempt + 1 == self.config["retries"]:
                    raise
                time.sleep(pause * 2 ** attempt)
        raise RuntimeError("Request exhausted retries")

    def audit(self, output):
        rows = read_raw(self.raw_dir)
        _, audit = prepare_records(rows, self.config)
        fields = sorted(set().union(*(row.keys() for row in rows)))
        audit["raw_field_missing_counts"] = {
            field: sum(row.get(field) is None or row.get(field) == "" for row in rows) for field in fields
        }
        audit["queries_completed"] = len(list(self.raw_dir.glob("*.response.json")))
        write_json(output, audit)
        return audit


def run(mode, raw_dir, report_dir, config, approval_path, terms_path):
    approval = check_approval(approval_path)
    collector = Collector(raw_dir, config, approval)
    report_dir = Path(report_dir)
    if mode == "pilot":
        terms = config["pilot_terms"]
        limit = 50
    else:
        if not (report_dir / "pilot_audit.json").exists():
            raise ValueError("Run and inspect the pilot before the main collection.")
        terms = [line.strip() for line in Path(terms_path).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        limit = config["request_limit"]
    for index, term in enumerate(terms):
        result = collector.fetch(term, limit)
        audit = collector.audit(report_dir / f"{mode}_audit.json")
        print(f"{mode} {index + 1}/{len(terms)}: {len(result['results'])} returned; "
              f"{audit['clean_rows']} unique eligible apps", flush=True)
        if mode == "main" and audit["clean_rows"] >= config["target_samples"]:
            break
    if mode == "main" and not audit["minimum_met"]:
        raise RuntimeError(f"Only {audit['clean_rows']} eligible apps; need {config['min_samples']}. "
                           "Expand the documented term list and resume; do not relax the research criteria.")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "main"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/reports")
    parser.add_argument("--approval", type=Path, default=ROOT / "config/source_approval.json")
    parser.add_argument("--terms", type=Path, default=ROOT / "config/search_terms.txt")
    args = parser.parse_args()
    run(args.mode, args.raw_dir, args.report_dir, settings(), args.approval, args.terms)


if __name__ == "__main__":
    main()
