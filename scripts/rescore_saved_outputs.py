from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluators import ATTACK_LABELS, DefensiveComplianceEvaluator


DEFAULT_RUNS = (
    ROOT / "outputs" / "main_v16_final",
    ROOT / "outputs" / "main_v18_grammar_exfil",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_evaluation(row: Dict[str, Any]) -> Dict[str, Any]:
    evaluations = row.get("attack_evaluations") or []
    index = int(row.get("output_index", 0) or 0)
    if index < len(evaluations) and isinstance(evaluations[index], dict):
        return evaluations[index]
    if len(evaluations) == 1 and isinstance(evaluations[0], dict):
        return evaluations[0]
    return {"label": "invalid", "score": 0.0, "status": "invalid"}


def rescore_run(run_dir: Path, evaluator=None) -> Dict[str, Any]:
    evaluator = evaluator or DefensiveComplianceEvaluator()
    source = run_dir / "outputs.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Saved output artifact not found: {source}")

    comparisons = []
    old_counts = Counter({label: 0 for label in ATTACK_LABELS})
    new_counts = Counter({label: 0 for label in ATTACK_LABELS})
    transitions = Counter()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("output_text", "")
        old = _old_evaluation(row)
        new = evaluator.evaluate(text)
        old_label = str(old.get("label", "invalid"))
        old_label = old_label if old_label in ATTACK_LABELS else "invalid"
        new_label = new.label if new.label in ATTACK_LABELS else "invalid"
        old_counts[old_label] += 1
        new_counts[new_label] += 1
        transitions[(old_label, new_label)] += 1
        comparisons.append(
            {
                "source_line": line_number,
                "prompt_id": row.get("prompt_id"),
                "prompt_index": row.get("prompt_index"),
                "output_index": row.get("output_index"),
                "response_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                "response_excerpt": " ".join(str(text).split())[:240],
                "old_label": old_label,
                "old_score": float(old.get("score", 0.0) or 0.0),
                "old_evaluator_version": (
                    (old.get("metadata") or {}).get("version")
                    or (row.get("attack_evaluator") or {}).get("version")
                    or "unknown"
                ),
                "new_label": new_label,
                "new_score": float(new.score),
                "changed": old_label != new_label or float(old.get("score", 0.0) or 0.0) != float(new.score),
                "new_signals": new.metadata.get("signals", {}),
            }
        )

    total = len(comparisons)
    return {
        "run_id": run_dir.name,
        "run_path": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "source_artifact": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "source_sha256": _sha256(source),
        "sample_count": total,
        "old_label_counts": dict(old_counts),
        "new_label_counts": dict(new_counts),
        "old_compliance_rate": old_counts["compliant"] / total if total else 0.0,
        "new_compliance_rate": new_counts["compliant"] / total if total else 0.0,
        "changed_label_count": sum(row["old_label"] != row["new_label"] for row in comparisons),
        "transitions": {
            f"{old_label}->{new_label}": count
            for (old_label, new_label), count in sorted(transitions.items())
        },
        "comparisons": comparisons,
    }


def build_report(run_dirs: Iterable[Path]) -> Dict[str, Any]:
    evaluator = DefensiveComplianceEvaluator()
    return {
        "report_schema_version": 1,
        "method": "offline deterministic rescoring; no model generations",
        "evaluator": evaluator.metadata(),
        "runs": [rescore_run(path.resolve(), evaluator) for path in run_dirs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score saved outputs with the current deterministic evaluator."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Run directory containing outputs.jsonl; repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional separate JSON report path. Existing run artifacts are never modified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.run_dir or DEFAULT_RUNS)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output.resolve()
        source_dirs = {path.resolve() for path in (args.run_dir or DEFAULT_RUNS)}
        if any(output == source / name for source in source_dirs for name in ("outputs.jsonl", "summary.json", "manifest.json")):
            raise ValueError("Refusing to overwrite an original experiment artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
