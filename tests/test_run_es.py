from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evolutionary_strategy import ESConfig, evolutionary_strategy_run
from run_es import write_run_dir


class RunArtifactMRObjectiveTests(unittest.TestCase):
    def test_config_records_canonical_mode_and_full_formula(self):
        config = ESConfig(
            lambda_=2,
            mu=2,
            generations=1,
            lightweight=True,
            random_seed=23,
            verbose=False,
            mr_objective="semantic_recovery",
        )
        result = evolutionary_strategy_run(config=config, client=None)
        args = SimpleNamespace(
            api_key="must-not-be-written",
            model="fake-model",
            mr_objective="semantic_recovery",
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_run_dir(tmp, args, config, result)
            payload = json.loads((Path(tmp) / "config.json").read_text())

        self.assertEqual(payload["mr_objective"]["mode"], "semantic_recovery")
        self.assertEqual(
            payload["mr_objective"]["formula"],
            "fitness = 0.7 * ASV + 0.3 * MR",
        )
        self.assertNotIn("api_key", payload["args"])


if __name__ == "__main__":
    unittest.main()
