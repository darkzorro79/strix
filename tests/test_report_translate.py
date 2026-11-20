from __future__ import annotations

from typing import TYPE_CHECKING

from strix.report.translate import (
    _source_digest,
    _translation_digest_path,
    maybe_write_russian_copy,
)
from strix.report.writer import write_executive_report


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _counting_translate(calls: list[str], text: str = "# Translated\n\nDone.\n"):
    async def _fake(markdown: str) -> str:
        calls.append(markdown)
        return text

    return _fake


def test_skips_translation_after_futures_executors_shut_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_REPORT_LANG", "ru")
    monkeypatch.setattr("concurrent.futures.thread._shutdown", True)
    calls: list[str] = []
    monkeypatch.setattr("strix.report.translate._translate", _counting_translate(calls))

    source = tmp_path / "report.md"
    maybe_write_russian_copy(source, "# Report\n\nBody.\n")

    assert calls == []
    assert not (tmp_path / "report.ru.md").exists()


def test_skips_translation_when_digest_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_REPORT_LANG", "ru")
    body = "# Report\n\nBody.\n"
    source = tmp_path / "report.md"
    (tmp_path / "report.ru.md").write_text("# Translated\n", encoding="utf-8")
    _translation_digest_path(source).write_text(_source_digest(body) + "\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr("strix.report.translate._translate", _counting_translate(calls))

    maybe_write_russian_copy(source, body)

    assert calls == []


def test_executive_report_resave_does_not_retranslate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_REPORT_LANG", "ru")
    calls: list[str] = []
    monkeypatch.setattr("strix.report.translate._translate", _counting_translate(calls))

    write_executive_report(tmp_path, "Scan complete.")
    write_executive_report(tmp_path, "Scan complete.")
    assert len(calls) == 1

    write_executive_report(tmp_path, "Scan complete. One more finding.")
    assert len(calls) == 2
