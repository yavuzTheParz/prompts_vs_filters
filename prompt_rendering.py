from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional, Union


ALLOWED_STYLES = frozenset({"imperative", "plea"})
INTERNAL_MARKER_RE = re.compile(
    r"\[\[\s*/?\s*STYLE(?:_|PREFIX|SUFFIX)",
    re.IGNORECASE,
)
INTERNAL_PROMPT_RE = re.compile(
    r"\A"
    r"(?:\[\[STYLE_PREFIX:(?P<prefix_style>[A-Za-z0-9_-]+)\]\]"
    r"(?P<prefix>.*?)\[\[/STYLE_PREFIX\]\]\s*)?"
    r"(?P<body>.*?)"
    r"(?:\s*\[\[STYLE_SUFFIX:(?P<suffix_style>[A-Za-z0-9_-]+)\]\]"
    r"(?P<suffix>.*?)\[\[/STYLE_SUFFIX\]\])?"
    r"\s*\Z",
    re.DOTALL,
)


class PromptValidationError(ValueError):
    """Raised when an internal prompt cannot be rendered safely."""


@dataclass(frozen=True)
class InternalPrompt:
    prefix: str = ""
    body: str = ""
    suffix: str = ""
    style: Optional[str] = None


@dataclass(frozen=True)
class RenderedPrompt:
    internal: InternalPrompt
    text: str
    internal_sha256: str
    rendered_sha256: str

    def audit_dict(self) -> dict:
        return {
            "internal": asdict(self.internal),
            "internal_sha256": self.internal_sha256,
            "rendered_sha256": self.rendered_sha256,
        }


def _contains_internal_marker(text: str) -> bool:
    return bool(INTERNAL_MARKER_RE.search(text or ""))


def validate_internal_prompt(
    prompt: Union[str, InternalPrompt],
) -> InternalPrompt:
    if isinstance(prompt, str):
        return parse_internal_prompt(prompt)
    if not isinstance(prompt, InternalPrompt):
        raise TypeError("prompt must be text or an InternalPrompt")

    for field_name in ("prefix", "body", "suffix"):
        value = getattr(prompt, field_name)
        if not isinstance(value, str):
            raise PromptValidationError(f"{field_name} must be text")
        if _contains_internal_marker(value):
            raise PromptValidationError(
                f"Nested or malformed internal marker in {field_name}"
            )

    if prompt.style is not None and prompt.style not in ALLOWED_STYLES:
        raise PromptValidationError(f"Unknown prompt style: {prompt.style}")
    if (prompt.prefix or prompt.suffix) and prompt.style is None:
        raise PromptValidationError(
            "Styled prefix or suffix requires an explicit style"
        )
    return prompt


def parse_internal_prompt(text: str) -> InternalPrompt:
    if not isinstance(text, str):
        raise TypeError("internal prompt must be text")

    match = INTERNAL_PROMPT_RE.fullmatch(text)
    if match is None:
        raise PromptValidationError("Malformed internal prompt")

    fields = match.groupdict()
    prefix = (fields["prefix"] or "").strip()
    body = (fields["body"] or "").strip()
    suffix = (fields["suffix"] or "").strip()
    prefix_style = fields["prefix_style"]
    suffix_style = fields["suffix_style"]

    if any(_contains_internal_marker(value) for value in (prefix, body, suffix)):
        raise PromptValidationError(
            "Unbalanced, duplicated, nested, or unknown internal marker"
        )
    if fields["prefix_style"] is not None and not prefix:
        raise PromptValidationError("Style prefix cannot be empty")
    if fields["suffix_style"] is not None and not suffix:
        raise PromptValidationError("Style suffix cannot be empty")
    if prefix_style and prefix_style not in ALLOWED_STYLES:
        raise PromptValidationError(f"Unknown prompt style: {prefix_style}")
    if suffix_style and suffix_style not in ALLOWED_STYLES:
        raise PromptValidationError(f"Unknown prompt style: {suffix_style}")
    if prefix_style and suffix_style and prefix_style != suffix_style:
        raise PromptValidationError("Prefix and suffix styles must match")

    style = prefix_style or suffix_style
    return validate_internal_prompt(
        InternalPrompt(prefix=prefix, body=body, suffix=suffix, style=style)
    )


def serialize_internal_prompt(prompt: InternalPrompt) -> str:
    prompt = validate_internal_prompt(prompt)
    parts = []
    if prompt.prefix:
        parts.append(
            f"[[STYLE_PREFIX:{prompt.style}]]"
            f"{prompt.prefix}[[/STYLE_PREFIX]]"
        )
    parts.append(prompt.body)
    if prompt.suffix:
        parts.append(
            f"[[STYLE_SUFFIX:{prompt.style}]]"
            f"{prompt.suffix}[[/STYLE_SUFFIX]]"
        )
    return " ".join(part for part in parts if part).strip()


def render_prompt(prompt: Union[str, InternalPrompt]) -> str:
    internal = (
        parse_internal_prompt(prompt)
        if isinstance(prompt, str)
        else validate_internal_prompt(prompt)
    )
    text = " ".join(part for part in (internal.prefix, internal.body) if part)
    if internal.suffix:
        if text and internal.suffix[0] in ",.;:!?":
            text = text.rstrip(".,;:!?") + internal.suffix
        elif text:
            text += " " + internal.suffix
        else:
            text = internal.suffix
    if _contains_internal_marker(text):
        raise PromptValidationError("Rendered prompt contains an internal marker")
    return text


def render_with_audit(prompt: Union[str, InternalPrompt]) -> RenderedPrompt:
    internal = (
        parse_internal_prompt(prompt)
        if isinstance(prompt, str)
        else validate_internal_prompt(prompt)
    )
    rendered = render_prompt(internal)
    canonical = json.dumps(
        asdict(internal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return RenderedPrompt(
        internal=internal,
        text=rendered,
        internal_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )
