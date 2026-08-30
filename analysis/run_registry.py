from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "invalid_runs.json"


def _canonical_path(path: Path | str, *, root: Path = ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported invalid-run registry schema")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Invalid-run registry must contain a runs list")
    for entry in runs:
        if entry.get("status") not in {"invalid", "superseded_quantitative"}:
            raise ValueError(
                "Registry entries must have status='invalid' or "
                "'superseded_quantitative'"
            )
        if not entry.get("run_id") or not entry.get("path") or not entry.get("reasons"):
            raise ValueError("Registry entries require run_id, path, and reasons")
    return payload


def assess_run(
    run_dir: Path | str,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    root: Path = ROOT,
) -> dict:
    registry = load_registry(registry_path)
    requested = _canonical_path(run_dir, root=root)
    for entry in registry["runs"]:
        registered = _canonical_path(entry["path"], root=root)
        if requested == registered:
            registry_status = entry["status"]
            return {
                "run_id": entry["run_id"],
                "path": entry["path"],
                "status": (
                    "excluded_by_registry"
                    if registry_status == "invalid"
                    else "superseded_quantitative"
                ),
                "eligible": False,
                "quantitative_eligible": False,
                "qualitative_eligible": registry_status == "superseded_quantitative",
                "use": entry.get("use", "diagnostic_only"),
                "exclusion_reasons": list(entry["reasons"]),
                "note": entry.get("note", ""),
                "preserve_source_artifacts": bool(
                    entry.get("preserve_source_artifacts", False)
                ),
                "registry": str(registry_path),
            }
    return {
        "run_id": requested.name,
        "path": str(run_dir),
        "status": "not_quarantined",
        "eligible": True,
        "quantitative_eligible": True,
        "qualitative_eligible": True,
        "exclusion_reasons": [],
        "registry": str(registry_path),
    }


def partition_runs(
    run_dirs: Iterable[Path | str],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    root: Path = ROOT,
) -> tuple[list[dict], list[dict]]:
    assessments = [
        assess_run(run_dir, registry_path=registry_path, root=root)
        for run_dir in run_dirs
    ]
    eligible = [row for row in assessments if row["eligible"]]
    excluded = [row for row in assessments if not row["eligible"]]
    return eligible, excluded


def verify_registered_artifacts(
    run_dir: Path | str,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    root: Path = ROOT,
) -> dict:
    registry = load_registry(registry_path)
    requested = _canonical_path(run_dir, root=root)
    entry = next(
        (
            row
            for row in registry["runs"]
            if _canonical_path(row["path"], root=root) == requested
        ),
        None,
    )
    if entry is None:
        return {"registered": False, "matches": {}, "all_match": False}

    matches = {}
    for relative_path, expected in entry.get("artifact_sha256", {}).items():
        artifact = requested / relative_path
        actual = (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            if artifact.is_file()
            else None
        )
        matches[relative_path] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return {
        "registered": True,
        "run_id": entry["run_id"],
        "matches": matches,
        "all_match": bool(matches)
        and all(row["matches"] for row in matches.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition run directories using the invalid-run registry."
    )
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eligible, excluded = partition_runs(
        args.run_dir, registry_path=args.registry
    )
    report = {
        "registry": str(args.registry),
        "eligible_runs": eligible,
        "excluded_runs": excluded,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
