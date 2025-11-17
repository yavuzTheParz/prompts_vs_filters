from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Structure(Enum):
    ignore_all_override = 1
    write_me_a_poem = 2
    role_reprogramming = 3
    forced_start_constraint = 4


class Content(Enum):
    bomb_weapons = 1
    financial_crime = 2
    misinformation = 3
    hacking_cybercrime = 4


@dataclass
class Prompt:
    input_prompt: str = ""
    output_prompts: List[str] = field(default_factory=list)
    fitness: float = 0.0
    structure: Structure = Structure.ignore_all_override
    content: Content = Content.bomb_weapons


"""
example usage #1:

p = Prompt(
    input_prompt="Write me a short poem.",
    structure=Structure.write_me_a_poem,
    content=Content.misinformation
)

print(p)

example usage #2:

    if prompt.content == Content.bomb_weapons:
        ref_output = "This is the reference output for bomb category."
    
    elif prompt.content == Content.financial_crime:
        ref_output = "Ref output for fraud category."


"""