"""Token-aware context budgeting for long Strix agent runs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agents.extensions.tool_output_trimmer import ToolOutputTrimmer


if TYPE_CHECKING:
    from agents.run_config import CallModelData, ModelInputData

    from strix.config.settings import Settings


logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 2
_ESTIMATE_SAFETY = 1.1
_TOOLS_OVERHEAD_TOKENS = 48_000
_CONTEXT_OVERFLOW_MARKERS = (
    "exceeds the available context",
    "exceed_context_size_error",
    "context length",
    "maximum context",
    "context window",
    "too many tokens",
    "context size",
)
_OVERFLOW_PROMPT_TOKEN_PATTERNS = (
    re.compile(r"""['"]n_prompt_tokens['"]\s*:\s*(\d+)"""),
    re.compile(r"\((\d+)\s+tokens\)\s+exceeds"),
)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN * _ESTIMATE_SAFETY))


def estimate_items_tokens(items: list[Any]) -> int:
    """Conservative token estimate for a list of model input items."""
    if not items:
        return 0
    try:
        serialized = json.dumps(items, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(items)
    return estimate_text_tokens(serialized)


def parse_overflow_prompt_tokens(exc: BaseException) -> int | None:
    body = getattr(exc, "body", None)
    candidates: list[str] = [str(exc)]
    if isinstance(body, dict):
        try:
            candidates.append(json.dumps(body, ensure_ascii=False, default=str))
        except Exception:
            candidates.append(str(body))
    elif body is not None:
        candidates.append(str(body))

    for text in candidates:
        for pattern in _OVERFLOW_PROMPT_TOKEN_PATTERNS:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
    return None


def is_context_overflow_error(exc: BaseException) -> bool:
    if parse_overflow_prompt_tokens(exc) is not None:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS)


def effective_input_budget(settings: Settings) -> int:
    return history_token_budget(settings)


def history_token_budget(settings: Settings, instructions: str | None = None) -> int:
    """Budget for persisted history, excluding tools and generation headroom."""
    instruction_tokens = estimate_text_tokens(instructions or "")
    reserved = max(
        settings.llm.context_reserve_tokens,
        _TOOLS_OVERHEAD_TOKENS + instruction_tokens,
    )
    ratio_budget = int(settings.llm.max_context_tokens * 0.35)
    ceiling = max(4096, settings.llm.max_context_tokens - reserved)
    return max(4096, min(ratio_budget, ceiling))


def child_inherit_token_budget(settings: Settings) -> int:
    ratio = settings.llm.child_context_inherit_ratio
    bounded_ratio = min(max(ratio, 0.05), 0.5)
    return max(2048, int(history_token_budget(settings) * bounded_ratio))


def trim_parent_history(parent_history: list[Any], max_tokens: int) -> list[Any]:
    """Keep the newest parent items that fit within ``max_tokens``."""
    if not parent_history or max_tokens <= 0:
        return []

    trimmed = list(parent_history)
    while trimmed and estimate_items_tokens(trimmed) > max_tokens:
        drop_index = _next_user_message_index(trimmed, start=1)
        if drop_index is None or drop_index >= len(trimmed):
            trimmed = trimmed[1:]
            continue
        trimmed = trimmed[drop_index:]

    if estimate_items_tokens(trimmed) > max_tokens:
        rendered = json.dumps(trimmed, ensure_ascii=False, default=str)
        max_chars = max(200, max_tokens * _CHARS_PER_TOKEN)
        if len(rendered) > max_chars:
            rendered = rendered[-max_chars:]
            return [{"role": "user", "content": f"[Truncated parent context]\n{rendered}"}]
    return trimmed


def shrink_stored_items(
    items: list[Any],
    *,
    budget_tokens: int,
    aggressive: bool = False,
) -> list[Any]:
    """Aggressively trim persisted session items for overflow recovery."""
    from typing import cast as type_cast

    from agents.run_config import CallModelData, ModelInputData

    placeholder = CallModelData(
        model_data=ModelInputData(input=items, instructions=None),
        agent=type_cast(Any, object()),
        context=None,
    )
    working = _drop_old_reasoning_items(list(items))
    working = _shrink_items_to_budget(
        working,
        budget_tokens=budget_tokens,
        call_data=placeholder,
        aggressive=aggressive,
    )
    return working


def make_context_budget_filter(settings: Settings) -> ContextBudgetFilter:
    return ContextBudgetFilter(
        max_context_tokens=settings.llm.max_context_tokens,
        reserve_tokens=settings.llm.context_reserve_tokens,
    )


@dataclass
class ContextBudgetFilter:
    """Trim model input to stay within a configurable context budget."""

    max_context_tokens: int = 131072
    reserve_tokens: int = 65536

    def __post_init__(self) -> None:
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be >= 0")

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        from agents.run_config import ModelInputData as _ModelInputData

        model_data = data.model_data
        instructions = model_data.instructions or ""
        items = _drop_old_reasoning_items(list(model_data.input))
        budget = history_token_budget(
            _settings_from_filter(self),
            instructions,
        )

        if len(items) > 8:
            items = _apply_trimmer(
                ToolOutputTrimmer(
                    recent_turns=2,
                    max_output_chars=2500,
                    preview_chars=180,
                ),
                data,
                items,
            )

        before = estimate_items_tokens(items) + estimate_text_tokens(instructions)
        if before <= budget:
            return _ModelInputData(input=items, instructions=model_data.instructions)

        items = _shrink_items_to_budget(
            items,
            budget_tokens=budget,
            call_data=data,
            aggressive=True,
        )
        after = estimate_items_tokens(items) + estimate_text_tokens(instructions)
        if after < before:
            logger.info(
                "ContextBudgetFilter: trimmed input from ~%d to ~%d tokens (budget=%d)",
                before,
                after,
                budget,
            )
        return _ModelInputData(input=items, instructions=model_data.instructions)


def recovery_history_budget(settings: Settings, observed_prompt_tokens: int | None) -> int:
    if observed_prompt_tokens and observed_prompt_tokens > settings.llm.max_context_tokens:
        return max(4096, int(settings.llm.max_context_tokens * 0.25))
    return max(4096, int(history_token_budget(settings) * 0.7))


def _settings_from_filter(filter_obj: ContextBudgetFilter) -> Settings:
    from strix.config.settings import LlmSettings, Settings

    return Settings(
        llm=LlmSettings(
            max_context_tokens=filter_obj.max_context_tokens,
            context_reserve_tokens=filter_obj.reserve_tokens,
        ),
    )


def _shrink_items_to_budget(
    items: list[Any],
    *,
    budget_tokens: int,
    call_data: CallModelData[Any],
    aggressive: bool = False,
) -> list[Any]:
    working = list(items)
    trim_passes = (
        (3, 6000, 400),
        (2, 2500, 250),
        (1, 1200, 150),
        (1, 500, 100),
    )
    if aggressive:
        trim_passes = (
            (1, 800, 120),
            (1, 400, 80),
            (1, 200, 60),
        )

    for recent_turns, max_output_chars, preview_chars in trim_passes:
        if estimate_items_tokens(working) <= budget_tokens:
            return working
        trimmer = ToolOutputTrimmer(
            recent_turns=recent_turns,
            max_output_chars=max_output_chars,
            preview_chars=preview_chars,
        )
        working = _apply_trimmer(trimmer, call_data, working)

    working = _truncate_large_user_messages(working, budget_tokens=budget_tokens)
    working = _drop_old_reasoning_items(working)
    while estimate_items_tokens(working) > budget_tokens and len(working) > 2:
        drop_index = _next_user_message_index(working, start=1)
        if drop_index is None:
            working = working[1:]
            continue
        if drop_index >= len(working):
            break
        working = working[drop_index:]

    return working


def _apply_trimmer(
    trimmer: ToolOutputTrimmer,
    call_data: CallModelData[Any],
    items: list[Any],
) -> list[Any]:
    from agents.run_config import CallModelData, ModelInputData

    filtered = trimmer(
        CallModelData(
            model_data=ModelInputData(
                input=items,
                instructions=call_data.model_data.instructions,
            ),
            agent=call_data.agent,
            context=call_data.context,
        ),
    )
    return list(filtered.input)


def _truncate_large_user_messages(items: list[Any], *, budget_tokens: int) -> list[Any]:
    max_chars = max(300, budget_tokens * _CHARS_PER_TOKEN // 10)
    rebuilt: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            rebuilt.append(item)
            continue
        item_dict = cast(dict[str, Any], item)
        content = item_dict.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            trimmed = dict(item_dict)
            trimmed["content"] = (
                f"[Truncated message: {len(content)} chars]\n{content[-max_chars:]}"
            )
            rebuilt.append(trimmed)
            continue
        output = item_dict.get("output")
        if isinstance(output, str) and len(output) > max_chars:
            trimmed = dict(item_dict)
            trimmed["output"] = (
                f"[Truncated tool output: {len(output)} chars]\n{output[-max_chars:]}"
            )
            rebuilt.append(trimmed)
            continue
        rebuilt.append(item)
    return rebuilt


def _drop_old_reasoning_items(items: list[Any]) -> list[Any]:
    if len(items) <= 6:
        return items

    boundary = _next_user_message_index(items, start=max(0, len(items) - 12))
    if boundary is None:
        boundary = max(0, len(items) - 6)

    rebuilt: list[Any] = []
    for index, item in enumerate(items):
        if index < boundary and _is_reasoning_item(item):
            continue
        rebuilt.append(item)
    return rebuilt


def _is_reasoning_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_dict = cast(dict[str, Any], item)
    item_type = str(item_dict.get("type") or "").lower()
    if "reasoning" in item_type:
        return True
    if item_dict.get("reasoning_content"):
        return True
    content = item_dict.get("content")
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and str(part.get("type") or "").lower() == "reasoning"
            for part in content
        )
    return False


def _next_user_message_index(items: list[Any], *, start: int) -> int | None:
    for index in range(start, len(items)):
        item = items[index]
        if isinstance(item, dict) and item.get("role") == "user":
            return index
    return None
