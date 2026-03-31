"""Print resolved env config for a PULSE task (obs, actions, rewards, terminations, commands)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pulse.runtime.launcher import run_entry


def main() -> int:
    entry_script = Path(__file__).resolve().parent / "_entry.py"
    return run_entry(entry_script, sys.argv[1:], mode="inspect")


if __name__ == "__main__":
    raise SystemExit(main())