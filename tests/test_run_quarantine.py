from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis.run_registry import (
    DEFAULT_REGISTRY,
    ROOT,
    assess_run,
    load_registry,
    partition_runs,
    verify_registered_artifacts,
)


TRIGGERING_RUN = Path("outputs/coevo_g320_l10_k3_seed13_run")
EXPECTED_REASONS = {
    "internal_marker_leak",
    "marker_corruption",
    "quality_gate_failure",
    "mode_collapse",
    "duplicate_filter_rule",
}


class RunQuarantineTests(unittest.TestCase):
    def test_registry_quarantines_triggering_run_with_required_reasons(self):
        registry = load_registry(DEFAULT_REGISTRY)
        self.assertEqual(registry["schema_version"], 2)
        assessment = assess_run(TRIGGERING_RUN)
        self.assertFalse(assessment["eligible"])
        self.assertEqual(assessment["status"], "excluded_by_registry")
        self.assertEqual(set(assessment["exclusion_reasons"]), EXPECTED_REASONS)
        self.assertTrue(assessment["preserve_source_artifacts"])

    def test_partition_excludes_quarantined_run_automatically(self):
        eligible, excluded = partition_runs(
            [TRIGGERING_RUN, Path("outputs/runs/t10_full")]
        )
        self.assertEqual([row["run_id"] for row in excluded], [
            "coevo_g320_l10_k3_seed13"
        ])
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["status"], "not_quarantined")

    def test_pref_fix_runs_are_quantitatively_superseded_but_retained(self):
        assessment = assess_run(Path("outputs/main_v16_final"))

        self.assertEqual(assessment["status"], "superseded_quantitative")
        self.assertFalse(assessment["quantitative_eligible"])
        self.assertTrue(assessment["qualitative_eligible"])
        self.assertTrue(assessment["preserve_source_artifacts"])

    def test_evaluator_version_policy_covers_unlisted_v4_runs(self):
        assessment = assess_run(Path("outputs/default_filter_v4"))

        self.assertEqual(assessment["status"], "superseded_quantitative")
        self.assertEqual(
            assessment["attack_evaluator_version"], "defensive-compliance-v4"
        )
        self.assertFalse(assessment["quantitative_eligible"])
        self.assertTrue(assessment["qualitative_eligible"])

    def test_ablation_analyzer_writes_exclusion_manifest_before_reading_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "analysis"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analysis" / "analyze_ablation.py"),
                    "--run-dir",
                    str(TRIGGERING_RUN),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 2, process.stderr)
            manifest = json.loads(
                (output_dir / "analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            exclusions = json.loads(
                (output_dir / "exclusions.json").read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["status"], "excluded_by_registry")
        self.assertEqual(set(manifest["exclusion_reasons"]), EXPECTED_REASONS)
        self.assertEqual(exclusions[0]["run_id"], "coevo_g320_l10_k3_seed13")

    def test_artifact_verification_is_read_only(self):
        content = b"unchanged diagnostic artifact\n"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            artifact = run_dir / "summary.json"
            artifact.write_bytes(content)
            registry = root / "invalid_runs.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "runs": [
                            {
                                "run_id": "fixture",
                                "path": "run",
                                "status": "invalid",
                                "reasons": ["controlled_fixture"],
                                "artifact_sha256": {"summary.json": digest},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            before = artifact.read_bytes()
            result = verify_registered_artifacts(
                run_dir,
                registry_path=registry,
                root=root,
            )
            after = artifact.read_bytes()
        self.assertTrue(result["all_match"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
