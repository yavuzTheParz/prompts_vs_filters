from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from prompt_rendering import (
    InternalPrompt,
    RenderedPrompt,
    parse_internal_prompt,
    render_with_audit,
    serialize_internal_prompt,
    validate_internal_prompt,
)


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
    prompt_representation: Optional[InternalPrompt] = None

    def __post_init__(self) -> None:
        if self.prompt_representation is None:
            self.prompt_representation = parse_internal_prompt(self.input_prompt)
        else:
            self.prompt_representation = validate_internal_prompt(
                self.prompt_representation
            )
            self.input_prompt = serialize_internal_prompt(
                self.prompt_representation
            )

    @property
    def internal_prompt(self) -> InternalPrompt:
        parsed = parse_internal_prompt(self.input_prompt)
        if parsed != self.prompt_representation:
            self.prompt_representation = parsed
        return parsed

    def render_input(self) -> RenderedPrompt:
        return render_with_audit(self.internal_prompt)
