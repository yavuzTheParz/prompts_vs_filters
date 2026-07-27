from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "full_validation_report.txt"
DEFAULT_HISTORY = ROOT / "outputs" / "full_validation_experiment_history.csv"
DEFAULT_RUN_DIR = ROOT / "outputs" / "full_validation_run"


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _append_section(report_path: Path, title: str, body: str) -> None:
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## {title}\n")
        handle.write(body.rstrip())
        handle.write("\n")


def _run_command(command: list[str], report_path: Path, title: str) -> int:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=os.environ.copy(),
    )
    body = [
        f"Command: {_format_command(command)}",
        f"Return code: {completed.returncode}",
        "",
        "[stdout]",
        completed.stdout.rstrip() or "<empty>",
        "",
        "[stderr]",
        completed.stderr.rstrip() or "<empty>",
    ]
    _append_section(report_path, title, "\n".join(body))
    return completed.returncode


def _append_file(report_path: Path, title: str, source_path: Path) -> None:
    if not source_path.exists():
        _append_section(report_path, title, f"Missing file: {source_path}")
        return
    _append_section(report_path, title, source_path.read_text(encoding="utf-8", errors="replace"))


def _history_has_valid_best(history_path: Path) -> bool:
    if not history_path.exists():
        return False
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    return any(float(row.get("best_api_error") or 0.0) < 1.0 for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unit tests, then a short full-pipeline ES experiment, and write one report file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--mr-objective",
        choices=[
            "behavioral_deviation",
            "semantic_recovery",
            "minimize",
            "maximize",
        ],
        default="behavioral_deviation",
    )
    parser.add_argument("--filter-update-every", type=int, default=1)
    parser.add_argument("--top-k-filter", type=int, default=2)
    parser.add_argument("--real-llm", action="store_true", help="Run the experiment against a configured local LLM.")
    parser.add_argument("--base-url", default=os.getenv("LOCAL_LLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("LOCAL_LLM_API_KEY"))
    parser.add_argument("--model", default=os.getenv("FILTER_EVOLUTION_TEST_MODEL", "dphn/Dolphin3.0-Llama3.1-8B"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("FILTER_EVOLUTION_TEST_TIMEOUT", "180")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.output.resolve()
    history_path = args.history_csv.resolve()
    run_dir = args.run_dir.resolve()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    header = "\n".join(
        [
            "Prompts vs Filters full validation report",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Python: {sys.executable}",
            f"Workspace: {ROOT}",
        ]
    )
    report_path.write_text(header + "\n", encoding="utf-8")

    unit_test_cmd = [
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]
    unit_status = _run_command(unit_test_cmd, report_path, "Unit Tests")

    experiment_cmd = [
        sys.executable,
        "-B",
        "run_es.py",
        "--variant",
        "cma_es",
        "--selection-mode",
        "lexicographic",
        "--mr-objective",
        args.mr_objective,
        "--filter-update-every",
        str(args.filter_update_every),
        "--top-k-filter",
        str(args.top_k_filter),
        "--mu",
        str(args.mu),
        "--lambda",
        str(args.lambda_),
        "--generations",
        str(args.generations),
        "--seed",
        str(args.seed),
        "--history-csv",
        str(history_path),
        "--run-dir",
        str(run_dir),
        "--quiet",
    ]
    if args.real_llm:
        experiment_cmd.extend(["--model", args.model, "--timeout", str(args.timeout)])
        if args.base_url:
            experiment_cmd.extend(["--base-url", args.base_url])
        if args.api_key:
            experiment_cmd.extend(["--api-key", args.api_key])
    else:
        experiment_cmd.insert(3, "--dry-run")

    experiment_status = _run_command(experiment_cmd, report_path, "Full-Pipeline Experiment")
    _append_file(report_path, "Experiment History CSV", history_path)
    valid_best_seen = _history_has_valid_best(history_path)
    if experiment_status == 0 and not valid_best_seen:
        experiment_status = 2

    summary = "\n".join(
        [
            f"Unit tests: {'PASS' if unit_status == 0 else 'FAIL'}",
            f"Experiment: {'PASS' if experiment_status == 0 else 'FAIL'}",
            f"Valid best response seen: {'YES' if valid_best_seen else 'NO'}",
            f"Report: {report_path}",
            f"History CSV: {history_path}",
            f"Run dir: {run_dir}",
        ]
    )
    _append_section(report_path, "Summary", summary)
    print(summary)

    return 0 if unit_status == 0 and experiment_status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
