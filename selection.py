from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from Prompt_class import Prompt


@dataclass
class RankPartitionInfo:
    """Summary information for the best rank partitions in a sorted population."""

    step_asv: float = 0.05
    step_mr: float = 0.05
    best_asv_partition: float = 0.0
    best_mr_partition: float = 0.0
    first_asv_partition_size