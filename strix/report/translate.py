"""Russian markdown copy written next to each English scan report.

The English file stays the source of record. When a report is saved, the
configured model translates the prose into Russian and that text is stored
as a sibling ``*.ru.md``. Code, identifiers, and fenced blocks are left as
written. A translation failure is logged and does not fail the save.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from strix.config.models import StrixProvider
from strix.core.inputs import make_model_settings


if TYPE_CHECKING:
    from pathlib import Path

    from agents.items import ModelResponse


logger = logging.getLogger(__name__)

_OFF = frozenset({"", "off", "none", "false", "0", "en"})
_MAX_OUTPUT_TOKENS = 16_384

_INSTRUCTIONS = """\
You translate a security-assessment markdown file into Russian.

Output only the translated markdown. No preamble and no wrapping code fence.

Rules:
- Keep the heading hierarchy, lists, bold, and blank lines.
- Leave fenced code blocks byte-for-byte, including the language tag.
- Leave inline code, file paths, function and method names, HTTP methods, \
status codes, CVSS scores, CWE identifiers, and URLs unchanged.
- Translate the surrounding prose. Do not add, drop, or re-rank findings.
- Do not invent remediation that is not in the source.
"""


def translation_language() -> str | None:
    """Language tag for the extra copy, or None when only English is written.

    ``STRIX_REPORT_LANG`` wins over the settings file so tests can turn the
    extra file off without touching the developer's config.
    """
    raw = os.environ.get("STRIX_REPORT_LANG")
    if raw is None:
        from strix.config import load_settings

        raw = load_settings().report.lang
    lang = raw.strip().lower()
    if lang in _OFF:
        return None
    return lang


def russian_markdown_path(path: Path) -> Path:
    """``report.md`` -> ``report.ru.md`` beside the English file."""
    return path.with_name(f"{path.stem}.ru{path.suffix}")


def maybe_write_russian_copy(source: Path, markdown: str) -> None:
    """Write ``source``'s Russian sibling when translation is enabled."""
    if translation_language() != "ru":
        return
    try:
        translated = _run(_translate(markdown))
    except Exception:
        logger.exception("Russian translation failed for %s", source)
        return
    if not translated:
        logger.warning("Russian translation of %s was empty; English file kept", source)
        return
    from strix.report.writer import atomic_write_text

    dest = russian_markdown_path(source)
    atomic_write_text(dest, translated.rstrip() + "\n")
    logger.info("Saved Russian translation to: %s", dest)


def _run(coro: object) -> str | None:
    running = False
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(coro)  # type: ignore[arg-type]
    # A running loop cannot nest asyncio.run. The save path is normally on a
    # worker thread; this branch covers a caller already inside the loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


def _unwrap_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    _first, newline, rest = stripped.partition("\n")
    if not newline or not rest.rstrip().endswith("```"):
        return stripped
    return rest.rstrip()[: rest.rstrip().rfind("```")].strip()


def _extract_text(response: ModelResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        parts.extend(
            chunk.text
            for chunk in item.content
            if isinstance(chunk, ResponseOutputText) and chunk.text
        )
    return "".join(parts)


async def _translate(markdown: str) -> str | None:
    from strix.config import load_settings

    settings = load_settings()
    model = (settings.llm.model or "").strip()
    if not model:
        logger.warning("Russian translation skipped: no LLM model configured")
        return None
    from strix.config.models import configure_sdk_model_defaults

    configure_sdk_model_defaults(settings)
    llm = settings.llm
    model_settings = make_model_settings(
        None,
        model_name=model,
        request_timeout=llm.timeout,
        prompt_cache=False,
        extra_headers=llm.extra_headers,
        extra_body=llm.extra_body,
        has_tools=False,
    ).resolve(ModelSettings(max_tokens=_MAX_OUTPUT_TOKENS))
    response = await (
        StrixProvider()
        .get_model(model)
        .get_response(
            system_instructions=_INSTRUCTIONS,
            input=markdown,
            model_settings=model_settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    )
    text = _unwrap_fence(_extract_text(response))
    if not text.lstrip().startswith("#"):
        logger.warning("Russian translation did not start with a heading")
        return None
    return text
