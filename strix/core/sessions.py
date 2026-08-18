"""SDK session helpers for Strix agents."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

from agents.memory import SQLiteSession

from strix.core.context_budget import (
    estimate_items_tokens,
    estimate_text_tokens,
    history_token_budget,
    recovery_history_budget,
    shrink_stored_items,
)


if TYPE_CHECKING:
    from pathlib import Path

    from agents.items import TResponseInputItem
    from agents.memory import Session

    from strix.config.settings import Settings


logger = logging.getLogger(__name__)


def open_agent_session(agent_id: str, path: Path) -> SQLiteSession:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(session_id=agent_id, db_path=path)


_IMAGE_REJECTED_TEXT = "[image rejected by the model]"


async def strip_all_images_from_session(session: Session) -> bool:
    items = await session.get_items()
    if not items:
        return False

    rebuilt: list[Any] = []
    changed = False
    for item in items:
        item_dict = cast("dict[str, Any]", item) if isinstance(item, dict) else None
        if (
            item_dict is not None
            and item_dict.get("type") == "function_call_output"
            and isinstance(item_dict.get("output"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "input_image" for b in item_dict["output"]
            )
        ):
            rebuilt.append(
                {
                    "type": "function_call_output",
                    "call_id": item_dict.get("call_id"),
                    "output": [{"type": "input_text", "text": _IMAGE_REJECTED_TEXT}],
                },
            )
            changed = True
        else:
            rebuilt.append(item)

    if not changed:
        return False

    rebuilt_items = cast("list[TResponseInputItem]", rebuilt)
    await session.clear_session()
    try:
        await session.add_items(rebuilt_items)
    except Exception:
        with contextlib.suppress(Exception):
            await session.add_items(rebuilt_items)
        raise
    return True


async def trim_session_for_context_budget(
    session: Session,
    settings: Settings,
    *,
    observed_prompt_tokens: int | None = None,
    force: bool = False,
) -> bool:
    """Shrink persisted session history before or after a context-overflow model error."""
    items = await session.get_items()
    if not items:
        return False

    budget = recovery_history_budget(settings, observed_prompt_tokens)
    before = estimate_items_tokens(list(items))
    if not force and before <= budget:
        return False

    trimmed_items = shrink_stored_items(
        list(items),
        budget_tokens=budget,
        aggressive=True,
    )
    after = estimate_items_tokens(trimmed_items)
    if trimmed_items == list(items) and not force:
        return False

    rebuilt_items = cast("list[TResponseInputItem]", trimmed_items)
    await session.clear_session()
    try:
        await session.add_items(rebuilt_items)
    except Exception:
        with contextlib.suppress(Exception):
            await session.add_items(rebuilt_items)
        raise

    logger.info(
        "Session trim: ~%d -> ~%d estimated tokens (budget=%d, observed_prompt=%s)",
        before,
        after,
        budget,
        observed_prompt_tokens,
    )
    return True


async def proactively_trim_session(
    session: Session,
    settings: Settings,
    *,
    instructions: str | None = None,
) -> bool:
    items = await session.get_items()
    if not items:
        return False

    budget = history_token_budget(settings, instructions)
    before = estimate_items_tokens(list(items)) + estimate_text_tokens(instructions or "")
    if before <= budget:
        return False

    return await trim_session_for_context_budget(
        session,
        settings,
        force=True,
    )
