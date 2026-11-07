"""Create a PRIVATE instructor submission ZIP; never upload or add it to Git.

The selected experiment is archived at work/<experiment_id>/ alongside source
and public deliverables, preserving the documented notebook/CLI paths. Run this
only after report generation and visual review.
"""
import argparse
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from appstore_ml.common import ROOT, sha256
from appstore_ml.prepare import prepare_records, read_raw
from appstore_ml.split import dataset_fingerprint

try:
    from scripts.check_publication import public_artifacts
except ModuleNotFoundError:
    from check_publication import public_artifacts

EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", ".git", ".venv"}
CODE_EXTENSIONS = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"}
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def contained_file(path, base):
    """Do not follow symlinks/junctions or package files outside the selected tree."""
    path, base = Path(path), Path(base).resolve()
    if not path.is_file() or not path.resolve().is_relative_to(base):
        raise ValueError(f"Missing or out-of-tree submission file: {path}")
    for part in [path, *path.parents]:
        if part == base:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Linked submission files are not accepted: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Unexpectedly large submission file: {path}")
    return path


def require_file(path):
    if not Path(path).is_file():
        raise ValueError(f"Submission is incomplete; required file is missing: {path}")
    return Path(path)


def validate_run(run_dir):
    """Check raw bytes, prepared data and original saved split without regenerating them."""
    run_dir = Path(run_dir).resolve()
    raw = run_dir / "data/raw"
    if raw.is_dir() and any(p.is_dir() for p in raw.iterdir()):
        raise ValueError("Nested raw folders are not supported; preserve one flat checksummed response archive.")
    responses = sorted(raw.glob("*.response.json"))
    if not responses:
        raise ValueError("Submission needs the real API raw response archive; no responses were found.")
    for response in responses:
        contained_file(response, run_dir)
        contained_file(require_file(response.with_name(response.name.replace(".response.json", ".meta.json"))),
                       run_dir)
    apps = contained_file(require_file(run_dir / "data/processed/apps.csv"), run_dir)
    split_file = contained_file(require_file(run_dir / "data/splits/split.csv"), run_dir)
    metadata_file = contained_file(require_file(run_dir / "data/splits/metadata.json"), run_dir)
    snapshot_file = contained_file(require_file(run_dir / "data/snapshot.json"), run_dir)
    provenance_file = contained_file(require_file(run_dir / "outputs/reports/provenance.json"), run_dir)
    selection_file = contained_file(require_file(run_dir / "outputs/reports/model_selection.json"), run_dir)
    contained_file(require_file(run_dir / "outputs/reports/source_hashes.json"), run_dir)
    contained_file(require_file(run_dir / "outputs/private/test_predictions.csv"), run_dir)
    contained_file(require_file(run_dir / "outputs/private/cv_folds.csv"), run_dir)
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance.get("data_kind") != "real_api":
        raise ValueError("Submission packaging requires a real_api experiment, never a synthetic fixture.")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != sha256(split_file):
        raise ValueError("Saved split manifest checksum mismatch; restore the original manifest.")
    frame = pd.read_csv(apps)
    fingerprint = dataset_fingerprint(frame)
    if fingerprint != metadata.get("dataset_sha256") or fingerprint != provenance.get("dataset_sha256"):
        raise ValueError("Processed dataset fingerprint does not match the saved split and provenance.")
    config = provenance.get("config")
    if not config:
        raise ValueError("Run provenance must preserve its exact experiment configuration.")
    rebuilt, _ = prepare_records(read_raw(raw), config)
    if dataset_fingerprint(rebuilt) != fingerprint:
        raise ValueError("Raw API responses do not reconstruct the archived processed dataset.")
    manifest = pd.read_csv(split_file)
    if (list(manifest.columns) != ["trackId", "split"] or manifest.trackId.duplicated().any()
            or set(manifest.trackId) != set(frame.trackId)
            or set(manifest["split"]) != {"train", "test"}):
        raise ValueError("Saved split contains invalid, duplicate or missing app assignments.")
    for split in ("train", "test"):
        if int(manifest["split"].eq(split).sum()) != provenance.get(f"{split}_n"):
            raise ValueError(f"Saved {split} count does not agree with provenance.")
    expected_checksums = provenance.get("source_response_checksums")
    if expected_checksums != {p.name: sha256(p) for p in responses}:
        raise ValueError("Raw response list/checksums differ from run provenance.")
    frozen = json.loads(snapshot_file.read_text(encoding="utf-8"))
    if frozen != {"dataset_sha256": fingerprint, "raw_checksums": expected_checksums}:
        raise ValueError("Frozen snapshot record disagrees with the dataset/raw archive.")
    selected = json.loads(selection_file.read_text(encoding="utf-8")).get("selected_model")
    metrics_file = contained_file(require_file(run_dir / "outputs/reports/test_metrics.csv"), run_dir)
    evaluated_models = set(pd.read_csv(metrics_file)["model"])
    if not selected or selected not in evaluated_models:
        raise ValueError("Model selection and evaluated-model records disagree.")
    model_files = sorted((run_dir / "outputs/models").glob("*.joblib"))
    if not model_files:
        raise ValueError("No fitted model files were retained for the private submission.")
    # Naming follows the project's training pipeline. Models are never deserialized here.
    for name in evaluated_models:
        expected = run_dir / "outputs/models" / (name.lower().replace(" ", "_").replace("+", "plus") + ".joblib")
        contained_file(require_file(expected), run_dir)
    return provenance


def submission_files(root, run_dir, archive_run_prefix):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    members = {}

    def add(path, name):
        path = contained_file(path, run_dir if str(name).startswith(archive_run_prefix + "/") else root)
        if name in members or ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError(f"Duplicate or unsafe archive member: {name}")
        members[name] = path

    for name in ("README.md", "AGENTS.md", "pyproject.toml", ".gitignore"):
        add(require_file(root / name), name)
    for directory in ("src", "tests", "scripts", "config", "docs"):
        for path in sorted((root / directory).rglob("*")):
            if (path.is_file() and path.suffix in CODE_EXTENSIONS
                    and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)):
                add(path, path.relative_to(root).as_posix())
    for path in public_artifacts(root):
        add(path, path.relative_to(root).as_posix())
    add(require_file(run_dir / "data/snapshot.json"), archive_run_prefix + "/data/snapshot.json")
    for relative in ("data/raw", "data/processed", "data/splits", "outputs/private", "outputs/models",
                     "outputs/reports"):
        folder = run_dir / relative
        for path in sorted(folder.glob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".json", ".csv", ".joblib", ".txt", ".md"}:
                raise ValueError(f"Unexpected private-run file type: {path}")
            add(path, archive_run_prefix + "/" + path.relative_to(run_dir).as_posix())
    if sum(p.stat().st_size for p in members.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Private archive exceeds the 512 MiB input limit; inspect the selected run.")
    return members


def package_endterm(run_dir, output=None, root=ROOT):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(root / "work"):
        raise ValueError("Select an isolated experiment inside the repository's ignored work/ directory.")
    output = Path(output or root / "outputs/private/Endterm_Submission.zip").resolve()
    if output.suffix.lower() != ".zip" or not output.is_relative_to(root / "outputs/private"):
        raise ValueError("Private submission ZIP must remain inside ignored outputs/private/.")
    if output.exists():
        raise ValueError(f"Submission archive already exists; choose a new --output path: {output}")
    provenance = validate_run(run_dir)
    for relative in ("notebooks/endterm.ipynb", "outputs/endterm/Endterm_Report.pdf",
                     "outputs/endterm/Endterm_Presentation.pdf", "outputs/endterm/rubric_checklist.md"):
        require_file(root / relative)
    public_provenance = json.loads(require_file(root / "outputs/endterm/reports/provenance.json")
                                   .read_text(encoding="utf-8"))
    if public_provenance.get("dataset_sha256") != provenance["dataset_sha256"]:
        raise ValueError("Public deliverables refer to a different experiment; regenerate before packaging.")
    experiment_id = provenance.get("experiment_id") or provenance.get("config", {}).get("endterm", {}).get(
        "experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Provenance must contain a safe experiment_id for the canonical archive run path.")
    archive_run_prefix = "work/" + experiment_id
    members = submission_files(root, run_dir, archive_run_prefix)
    instructions = (
        "PRIVATE COURSE SUBMISSION - do not publish this ZIP or work/.\n"
        f"{archive_run_prefix}/ preserves raw API responses, prepared records, split IDs and fitted models.\n"
        f"After extraction, run commands from the archive root with --run-dir {archive_run_prefix}.\n"
        "Only provide these files through the instructor's authorized private submission channel.\n"
        "The original October Midterm snapshot is not reconstructed by this later experiment.\n"
        "Review rubric_checklist.md and complete actual human identities/contributions before submission.\n"
        "MANIFEST.json contains SHA-256 for every payload member; it excludes itself to avoid a circular hash.\n"
        "final_source_hashes.json identifies source at packaging time. The run's source_hashes.json is\n"
        "preserved unchanged and identifies the earlier experiment revision; report/packaging edits may differ.\n"
    ).encode()
    manifest = {
        "privacy": "private instructor submission; never public Git",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": provenance["dataset_sha256"],
        "archive_run_path": archive_run_prefix,
        "manifest_self_hash": "excluded because a file cannot contain its own ordinary SHA-256",
        "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in sorted(members.items())},
    }
    archived_sources = {name: sha256(path) for name, path in members.items()
                        if name.startswith(("src/", "tests/", "scripts/", "config/", "docs/"))
                        or name in {"README.md", "AGENTS.md", "pyproject.toml", ".gitignore"}}
    fitting_hashes_file = run_dir / "outputs/reports/source_hashes.json"
    fitting_hashes = json.loads(fitting_hashes_file.read_text(encoding="utf-8"))
    final_source_hashes = {
        "scope": "Final source at packaging time; does not replace the experiment-time source hashes.",
        "experiment_source_hashes_member": archive_run_prefix + "/outputs/reports/source_hashes.json",
        "files": archived_sources,
        "changed_since_experiment": sorted(name for name, digest in fitting_hashes.items()
                                           if archived_sources.get(name) != digest),
    }
    embedded = {"PRIVATE_SUBMISSION_README.txt": instructions, archive_run_prefix + "/.gitignore": b"*\n",
                "final_source_hashes.json": (json.dumps(final_source_hashes, indent=2) + "\n").encode()}
    for name, data in embedded.items():
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    if temporary.exists():
        raise ValueError(f"A previous packaging temporary file exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, name)
            for name, data in embedded.items():
                archive.writestr(name, data)
            archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            for name, spec in manifest["files"].items():
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["bytes"]:
                    raise ValueError(f"Archive checksum verification failed: {name}")
            if set(archive.namelist()) != set(manifest["files"]) | {"MANIFEST.json"}:
                raise ValueError("Archive contains unexpected members.")
        os.replace(temporary, output)
    except Exception:
        # Only remove this one known task-created file within the checked private directory.
        temporary.unlink(missing_ok=True)
        raise
    print(f"Private ZIP created: {output}\nSHA-256: {sha256(output)}\n"
          f"Payload members: {len(manifest['files'])}. No files were staged or uploaded.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new ZIP path inside outputs/private/.")
    args = parser.parse_args()
    package_endterm(args.run_dir, args.output)


if __name__ == "__main__":
    main()
