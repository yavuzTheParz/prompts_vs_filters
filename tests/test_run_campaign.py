from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_campaign import ALL_CONFIGS, parse_args, parse_seeds, plan, run_state


class CampaignPlanTests(unittest.TestCase):
    def args(self, root, *extra):
        return parse_args(["--plan", "--output-root", root, *extra])

    def test_fixed_phase_covers_every_configuration_and_seed(self):
        with tempfile.TemporaryDirectory() as root:
            jobs = plan(self.args(root, "--phase", "fixed"))
        self.assertEqual(len(jobs), len(ALL_CONFIGS) * 20)
        self.assertEqual(len({job["name"] for job in jobs}), len(jobs))
        # Seeds vary slowest: the first six jobs are one complete block.
        self.assertEqual({job["seed"] for job in jobs[:6]}, {1})
        self.assertEqual({job["config"] for job in jobs[:6]}, set(ALL_CONFIGS))
        command = jobs[0]["command"]
        self.assertEqual(command[command.index("--filter-update-every-evaluations") + 1], "0")
        self.assertEqual(command[command.index("--max-evaluations") + 1], "15000")

    def test_coevolution_schedule_scales_with_the_budget(self):
        with tempfile.TemporaryDirectory() as root:
            jobs = plan(self.args(root, "--phase", "coevo", "--configs", "medium_plus",
                                  "--seeds", "1", "--max-evaluations", "30000"))
        command = jobs[0]["command"]
        self.assertEqual(command[command.index("--filter-update-every-evaluations") + 1], "2000")
        self.assertEqual(command[command.index("--filter-warmup-evaluations") + 1], "4000")
        self.assertEqual(command[command.index("--survival") + 1], "(mu+lambda)")

    def test_shards_partition_the_plan_without_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            full = {job["name"] for job in plan(self.args(root, "--phase", "fixed"))}
            parts = [
                {job["name"] for job in plan(self.args(root, "--phase", "fixed", "--shard", f"{i}/3"))}
                for i in (1, 2, 3)
            ]
        self.assertEqual(set().union(*parts), full)
        self.assertEqual(sum(len(part) for part in parts), len(full))

    def test_finished_and_interrupted_runs_are_recognised(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "fixed_small_plus_s01"
            self.assertEqual(run_state(run_dir), "fresh")
            run_dir.mkdir()
            (run_dir / "checkpoint.pkl").write_bytes(b"x")
            self.assertEqual(run_state(run_dir), "resume")
            jobs = plan(self.args(root, "--phase", "fixed", "--configs", "small_plus", "--seeds", "1"))
            self.assertIn("--resume", jobs[0]["command"])
            (run_dir / "summary.json").write_text(json.dumps({"attacker_calls": 15000}))
            self.assertEqual(run_state(run_dir), "done")

    def test_seed_ranges_and_unknown_configs(self):
        self.assertEqual(parse_seeds("1-3,7"), [1, 2, 3, 7])
        with self.assertRaises(ValueError):
            parse_seeds("1,1")
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            plan(self.args(root, "--phase", "fixed", "--configs", "huge_plus"))


if __name__ == "__main__":
    unittest.main()
