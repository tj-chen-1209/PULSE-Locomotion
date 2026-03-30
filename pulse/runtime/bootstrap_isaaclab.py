"""Utilities to locate and import IsaacLab packages."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _candidate_roots() -> list[Path]:
    runtime_dir = Path(__file__).resolve().parent
    repo_root = runtime_dir.parent.parent
    _add_path(repo_root)
    env_root = os.environ.get("ISAACLAB_PATH")

    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            repo_root / ".." / "Issac" / "IsaacLab",
            repo_root / ".." / "IsaacLab",
            repo_root / "IsaacLab",
        ]
    )
    return [path.resolve() for path in candidates]


def _inject_source_paths(root: Path) -> None:
    _add_path(root / "source" / "isaaclab")
    _add_path(root / "source" / "isaaclab_tasks")
    _add_path(root / "source" / "isaaclab_rl")
    _add_path(root / "source" / "isaaclab_mimic")


def resolve_isaaclab_root() -> Path:
    for root in _candidate_roots():
        if root.exists() and (root / "source").exists():
            return root
    raise ModuleNotFoundError(
        "Cannot locate IsaacLab root. Set ISAACLAB_PATH to your IsaacLab repository root, "
        "for example: export ISAACLAB_PATH=/home/tingjia/Project/Issac/IsaacLab"
    )


def bootstrap_isaaclab() -> Path:
    """Ensure `isaaclab*` packages are importable from this repo."""
    try:
        import isaaclab  # noqa: F401
        return resolve_isaaclab_root()
    except ModuleNotFoundError:
        pass

    for root in _candidate_roots():
        _inject_source_paths(root)
        try:
            import isaaclab  # noqa: F401
            return root
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "Cannot import 'isaaclab'. Set ISAACLAB_PATH to your IsaacLab repository root, "
        "for example: export ISAACLAB_PATH=/home/tingjia/Project/Issac/IsaacLab"
    )

