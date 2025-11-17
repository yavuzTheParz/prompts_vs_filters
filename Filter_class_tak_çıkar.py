from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Structure(Enum):
    TYPE_A = 1
    TYPE_B = 2
    TYPE_C = 3


class Content(Enum):
    MALICIOUS = 1
    BENIGN = 2
    UNKNOWN = 3


@dataclass
class Prompt:
    input_prompt: str = ""
    output_prompts: List[str] = field(default_factory=list)
    fitness: float = 0.0
    structure: Structure = Structure.TYPE_A
    content: Content = Content.UNKNOWN
