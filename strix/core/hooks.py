"""SDK run hooks used by Strix orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agents.lifecycle import RunHooks

from strix.config import load_settings
from strix.core.agents import AgentCoordinator
from strix.core.sessions import proactively_trim_session
from strix.report.state import get_global_report_state


if TYPE_CHECKING:
    from agents import RunContextWrapper
    from agents.agent import Agent
    from agents.items import ModelResponse, TResponseInputItem

    from strix.config.settings import Settings


logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when the accumulated LLM cost reaches the configured budget."""


class ComposedRunHooks(RunHooks[dict[str, Any]]):
    """Invoke multiple run hooks in registration order."""

    def __init__(self, *hooks: RunHooks[dict[str, Any]]) -> None:
        self._hooks = hooks

    async def on_llm_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        response: ModelResponse,
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_end(context, agent, response)


class ContextBudgetHooks(RunHooks[dict[str, Any]]):
    """Persistently trim SDK session history before each model call."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings

    def _resolve_settings(self) -> Settings:
        return self._settings or load_settings()

    async def on_llm_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        del agent, input_items
        ctx = context.context if isinstance(context.context, dict) else {}
        coordinator = ctx.get("coordinator")
        agent_id = ctx.get("agent_id")
        if not isinstance(coordinator, AgentCoordinator) or not isinstance(agent_id, str):
            return

        runtime = coordinator.runtimes.get(agent_id)
        if runtime is None or runtime.session is None:
            return

        try:
            await proactively_trim_session(
                runtime.session,
                self._resolve_settings(),
                instructions=system_prompt,
            )
        except Exception:
            logger.exception("proactive session trim failed for agent %s", agent_id)


class ReportUsageHooks(RunHooks[dict[str, Any]]):
    """Persist SDK-native usage after every model response."""

    def __init__(self, *, model: str, max_budget_usd: float | None = None) -> None:
        import math

        if max_budget_usd is not None and (
            not math.isfinite(max_budget_usd) or max_budget_usd <= 0
        ):
            raise ValueError("max_budget_usd must be a finite number greater than 0")
        self._model = model
        self._max_budget_usd = max_budget_usd

    async def on_llm_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        response: ModelResponse,
    ) -> None:
        report_state = get_global_report_state()
        if report_state is None:
            return

        ctx = context.context if isinstance(context.context, dict) else {}
        agent_name = getattr(agent, "name", None)
        if not isinstance(agent_name, str):
            agent_name = None
        agent_id = ctx.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            agent_id = agent_name or "unknown"

        try:
            report_state.record_sdk_usage(
                agent_id=agent_id,
                agent_name=agent_name,
                model=self._model,
                usage=response.usage,
            )
        except Exception:
            logger.exception("failed to record SDK usage for agent %s", agent_id)

        if self._max_budget_usd is not None:
            cost = report_state.get_total_llm_cost()
            if cost >= self._max_budget_usd:
                raise BudgetExceededError(
                    f"Token budget of ${self._max_budget_usd:.2f} exceeded (spent ${cost:.4f})"
                )
