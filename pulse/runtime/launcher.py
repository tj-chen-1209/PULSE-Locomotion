"""Helpers for launching scripts with IsaacLab runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pulse.runtime.bootstrap_isaaclab import resolve_isaaclab_root


def _prepare_launcher_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TERM"] = "xterm"
    env["PULSE_ENTRY_MODE"] = mode

    if "CONDA_PREFIX" not in env:
        conda_candidates = [
            Path.home() / "miniconda3" / "envs" / "env_isaaclab",
            Path.home() / "anaconda3" / "envs" / "env_isaaclab",
        ]
        for candidate in conda_candidates:
            if (candidate / "bin" / "python").exists():
                env["CONDA_PREFIX"] = str(candidate)
                env["PATH"] = f"{candidate / 'bin'}:{env.get('PATH', '')}"
                break

    if shutil.which("python", path=env.get("PATH")) is None:
        python3_path = shutil.which("python3", path=env.get("PATH"))
        if python3_path:
            shim_dir = Path(tempfile.mkdtemp(prefix="pulse-python-shim-"))
            os.symlink(python3_path, shim_dir / "python")
            env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"

    return env


def run_entry(entry_script: Path, cli_args: list[str], mode: str) -> int:
    isaaclab_root = resolve_isaaclab_root()
    launcher = isaaclab_root / "isaaclab.sh"
    cmd = [str(launcher), "-p", str(entry_script), *cli_args]
    return subprocess.call(cmd, env=_prepare_launcher_env(mode))

