from __future__ import annotations

import os


def configure_transformers_for_pytorch() -> None:
    """Keep Hugging Face imports on the PyTorch path used by this project."""
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"


configure_transformers_for_pytorch()
