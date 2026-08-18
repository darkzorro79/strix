"""Tests for context budgeting helpers."""

from __future__ import annotations

from typing import Any, cast

from agents.run_config import CallModelData, ModelInputData

from strix.config.settings import LlmSettings, Settings
from strix.core.context_budget import (
    ContextBudgetFilter,
    child_inherit_token_budget,
    effective_input_budget,
    estimate_items_tokens,
    is_context_overflow_error,
    shrink_stored_items,
    trim_parent_history,
)


def _settings(
    *,
    max_context_tokens: int = 131072,
    context_reserve_tokens: int = 16384,
    child_context_inherit_ratio: float = 0.25,
) -> Settings:
    return Settings(
        llm=LlmSettings(
            max_context_tokens=max_context_tokens,
            context_reserve_tokens=context_reserve_tokens,
            child_context_inherit_ratio=child_context_inherit_ratio,
        ),
    )


def test_effective_input_budget_respects_reserve() -> None:
    assert effective_input_budget(_settings()) == int(131072 * 0.35)


def test_child_inherit_token_budget_uses_ratio() -> None:
    budget = child_inherit_token_budget(_settings())
    assert budget == int(int(131072 * 0.35) * 0.25)


def test_is_context_overflow_error_detects_llamacpp_message() -> None:
    exc = RuntimeError(
        "request (283736 tokens) exceeds the available context size (131072 tokens)",
    )
    assert is_context_overflow_error(exc)


def test_trim_parent_history_keeps_recent_items() -> None:
    history = [
        {"role": "user", "content": "old " * 5000},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "recent task"},
    ]
    trimmed = trim_parent_history(history, max_tokens=200)
    rendered = str(trimmed)
    assert "recent task" in rendered
    assert estimate_items_tokens(trimmed) <= 200


def test_shrink_stored_items_trims_large_tool_outputs() -> None:
    items = [
        {"role": "user", "content": "run scan"},
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "x" * 20000,
        },
        {"role": "user", "content": "continue"},
    ]
    shrunk = shrink_stored_items(items, budget_tokens=500)
    assert estimate_items_tokens(shrunk) < estimate_items_tokens(items)


def test_context_budget_filter_leaves_small_input_untouched() -> None:
    settings = _settings()
    items = [{"role": "user", "content": "hello"}]
    data = CallModelData(
        model_data=ModelInputData(input=items, instructions=None),
        agent=cast(Any, object()),
        context=None,
    )
    result = ContextBudgetFilter(
        max_context_tokens=settings.llm.max_context_tokens,
        reserve_tokens=settings.llm.context_reserve_tokens,
    )(data)
    assert result.input == items
