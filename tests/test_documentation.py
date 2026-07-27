from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationEvidenceTests(unittest.TestCase):
    def test_required_audit_documents_exist_and_are_linked(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        required = (
            "docs/architecture.md",
            "docs/metrics.md",
            "docs/safety.md",
            "docs/migration.md",
            "outputs/reports/final_implementation_report.md",
        )
        for relative_path in required:
            self.assertTrue((ROOT / relative_path).is_file())
            self.assertIn(relative_path, readme)

    def test_readme_uses_current_cli_options(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("--with-filter-coevolution", readme)
        run_help = subprocess.run(
            [sys.executable, str(ROOT / "run_es.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        ablation_help = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "run_ablation.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for option in (
            "--dry-run",
            "--filter-update-every",
            "--mr-objective",
            "--run-dir",
            "--disable-structural-mutation",
            "--disable-token-mutation",
        ):
            self.assertIn(option, run_help)
        for option in ("--generations", "--seeds", "--output-dir", "--matrix"):
            self.assertIn(option, ablation_help)

    def test_final_evidence_is_aggregate_and_complete(self):
        run_root = ROOT / "outputs" / "runs" / "t10_full"
        with (run_root / "run_summary.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 35)
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertEqual(len({row["initial_pool_sha256"] for row in rows}), 1)
        self.assertFalse({"input_prompt", "direct_output", "filtered_output"} & set(rows[0]))
        plots = list((run_root / "analysis").glob("convergence_*.svg"))
        self.assertEqual(len(plots), 6)


if __name__ == "__main__":
    unittest.main()
