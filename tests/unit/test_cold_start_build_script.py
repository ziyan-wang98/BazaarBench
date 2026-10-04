from __future__ import annotations

import subprocess

import pytest

from scripts.cold_start import build


def test_cold_start_warmup_success_uses_non_throwing_subprocess(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    build._run_warmup_or_exit(["warmup", "--dry-run"])

    assert calls == [(["warmup", "--dry-run"], {"check": False})]


def test_cold_start_warmup_failure_is_readable(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 17)

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        build._run_warmup_or_exit(["warmup", "--dry-run"])

    message = str(exc_info.value)
    assert "[cold-start-build] warmup command failed with exit code 17" in message
    assert "warmup --dry-run" in message
    assert "Traceback" not in message
