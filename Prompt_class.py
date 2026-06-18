from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Structure(Enum):
    ignore_all_override = 1
    question_request = 2
    imperative_instruction = 3
    role_reprogramming = 4
    poem_request = 5


class Content(Enum):
    bomb_weapons = 1
    hacking_cybercrime = 2
    violence = 3
    misinformation = 4
    hate_abuse = 5


@dataclass
class Prompt:
    input_prompt: str = ""
    output_prompts: List[str] = field(default_factory=list)
    direct_output: str = ""
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    structure: Structure = Structure.ignore_all_override
    content: Content = Content.bomb_weapons
