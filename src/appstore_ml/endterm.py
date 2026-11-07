"""Endterm experiment on a frozen private snapshot; never collects data."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import ROOT, settings, sha256, write_csv, write_json
from .prepare import prepare
from .split import dataset_fingerprint, fixed_split


def source_hashes(root=ROOT):
    paths = list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    paths += [root / "config/settings.json", root / "docs/endterm_protocol.md", root / "pyproject.toml"]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def validate_endterm_config(config):
    """Reject descriptive settings that disagree with the implemented protocol.

    Endterm parameters are prespecified in the training module. Configuration is
    their auditable declaration, not an alternative set of silently ignored knobs.
    """
    from .endterm_training import MLP_EPOCHS, model_specs

    declared = config.get("endterm")
    if not isinstance(declared, dict):
        raise ValueError("Endterm configuration must be recorded before training.")
    specs = model_specs(config["random_state"])
    network = specs["MLP"][0].named_steps["model"]
    pca = specs["SVM + PCA"][0].named_steps["pca"]
    expected = {
        "stage": "endterm", "primary_metric": "f1_class_1",
        "outer_folds": config["cv_folds"], "inner_folds": 3,
        "pca_retained_variance": pca.n_components,
        "mlp_hidden_layers": list(network.hidden_layer_sizes),
        "mlp_epochs": MLP_EPOCHS, "mlp_alpha": network.alpha,
        "mlp_learning_rate": network.learning_rate_init,
        "text_and_app_stage": "final",
    }
    for name, value in expected.items():
        if declared.get(name) != value:
            raise ValueError(f"Endterm configuration disagrees with implemented {name}: "
                             f"expected {value!r}, received {declared.get(name)!r}.")
    if network.max_iter != MLP_EPOCHS:
        raise ValueError("MLP comparison and diagnostic epoch budgets disagree.")


def verify_continuity(frame, config, new_experiment):
    fingerprint = dataset_fingerprint(frame)
    original = config["endterm"]["original_dataset_sha256"]
    if not new_experiment:
        raise ValueError("Original Midterm snapshot is unavailable/different. Restore its saved data and split, "
                         "or explicitly use --new-experiment and disclose the continuity gap.")
    return {"original_dataset_sha256": original, "current_dataset_sha256": fingerprint,
            "same_dataset": fingerprint == original,
            "same_original_split_verified": False,
            "status": "separate_experiment" if fingerprint != original else "original_split_not_verified",
            "limitation": "The original private split manifest is unavailable. A new split cannot establish "
                          "continuity, even if its seed is the same. No instructor exception is claimed."}


def experiment(run_dir, public_dir=None, new_experiment=False, render=True):
    from .endterm_analysis import analyze_unsupervised, evaluate_endterm
    from .endterm_training import train_endterm

    run_dir = Path(run_dir).resolve()
    public_dir = Path(public_dir or ROOT / "outputs/endterm").resolve()
    config = settings()
    validate_endterm_config(config)
    out = run_dir / "outputs"
    raw = run_dir / "data/raw"
    if any(p.parent != raw for p in raw.rglob("*.response.json")):
        raise ValueError("Nested raw response folders are unsupported; restore the flat collector snapshot.")
    frame, audit = prepare(raw, run_dir / "data/processed", config)
    if len(frame) < config["min_samples"]:
        raise ValueError("Real snapshot does not meet the minimum sample requirement.")
    continuity = verify_continuity(frame, config, new_experiment)
    # Dataset and manifest bytes are checked by fixed_split on every invocation.
    train, test, split_spec = fixed_split(frame, run_dir / "data/splits", config)
    if set(train.high_rated) != {0, 1} or set(test.high_rated) != {0, 1}:
        raise ValueError("Both classes must occur in both partitions.")
    raw_hashes = {p.name: sha256(p) for p in sorted(raw.glob("*.response.json"))}
    frozen_path = run_dir / "data/snapshot.json"
    frozen = {"dataset_sha256": split_spec["dataset_sha256"], "raw_checksums": raw_hashes}
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Frozen snapshot changed. Restore it or use a separate experiment directory.")
    write_json(frozen_path, frozen)
    print(f"Frozen snapshot: {len(frame)} apps, {len(train)} train / {len(test)} test", flush=True)
    fitted, cv, selected = train_endterm(train, out, config)
    # Descriptive PCA has no access to held-out objects or outcomes.
    analyze_unsupervised(train, out, config)
    metrics, _, _ = evaluate_endterm(fitted, test, train, selected, out)
    comparison = cv.merge(metrics, on="model", validate="one_to_one")
    write_csv(comparison, out / "reports/comparison.csv")
    provenance = {
        "experiment_id": config["endterm"]["experiment_id"], "data_kind": "real_api",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        **split_spec, "manifest_sha256": sha256(run_dir / "data/splits/split.csv"),
        "train_n": len(train), "test_n": len(test),
        "train_class_counts": {str(k): int(v) for k, v in train.high_rated.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test.high_rated.value_counts().items()},
        "source_response_checksums": raw_hashes, "config": config,
        "selected_model": selected, "continuity_status": continuity["status"],
        "test_used_for_selection": False,
    }
    write_json(out / "reports/provenance.json", provenance)
    write_json(out / "reports/continuity.json", continuity)
    write_json(out / "reports/cleaning_audit.json", audit)
    write_json(out / "reports/source_hashes.json", source_hashes())
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                      if d.metadata["Name"])
    (out / "reports/environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    publish_aggregates(out, public_dir)
    if render:
        from .endterm_report import build_deliverables
        from .endterm_slides import build_slides
        build_deliverables(public_dir)
        build_slides(public_dir)
    return provenance


def publish_aggregates(run_outputs, public_dir):
    """Copy only aggregate directories, never private predictions or models."""
    spec = importlib.util.spec_from_file_location("publication", ROOT / "scripts/check_publication.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    pending = []
    for name in ["reports", "figures"]:
        destination = Path(public_dir) / name
        for path in (Path(run_outputs) / name).iterdir():
            if path.is_file():
                policy.validate_artifact_bytes(f"outputs/endterm/{name}/{path.name}", path.read_bytes())
                pending.append((path, destination / path.name))
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "work/endterm-2026-11-08")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "outputs/endterm")
    parser.add_argument("--new-experiment", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from .endterm_report import build_deliverables
        from .endterm_slides import build_slides
        build_deliverables(args.public_dir)
        build_slides(args.public_dir)
    else:
        experiment(args.run_dir, args.public_dir, args.new_experiment)


if __name__ == "__main__":
    main()
