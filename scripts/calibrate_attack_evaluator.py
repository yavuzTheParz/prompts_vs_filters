#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluators import DefensiveComplianceEvaluator, calibration_confusion_matrix


DEFAULT_FIXTURES = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "attack_evaluator_calibration.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the defensive compliance evaluator confusion matrix."
    )
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    report = calibration_confusion_matrix(
        DefensiveComplianceEvaluator(),
        fixtures,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
