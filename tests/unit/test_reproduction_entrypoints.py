from __future__ import annotations

import subprocess
import sys


def _run_ok(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )


def test_paper_facing_script_help_entrypoints_execute() -> None:
    commands = [
        [sys.executable, "scripts/compute_transaction_value.py", "--help"],
        [sys.executable, "scripts/compute_social_metrics.py", "--help"],
    ]

    for cmd in commands:
        result = _run_ok(cmd)
        assert "usage:" in result.stdout
