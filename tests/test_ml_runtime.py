from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ml_runtime import configure_transformers_for_pytorch


class MLRuntimeTests(unittest.TestCase):
    def test_transformers_tensorflow_and_flax_backends_are_disabled(self):
        with patch.dict(os.environ, {"USE_TF": "1", "USE_FLAX": "1"}):
            configure_transformers_for_pytorch()
            self.assertEqual(os.environ["USE_TF"], "0")
            self.assertEqual(os.environ["USE_FLAX"], "0")


if __name__ == "__main__":
    unittest.main()
