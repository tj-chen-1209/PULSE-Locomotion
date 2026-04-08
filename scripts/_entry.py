"""Unified train/play/inspect/eval runtime entry for PULSE."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import logging
import os
import platform
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root is importable when this file is launched by isaaclab.sh.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from pulse.runtime import rsl_rl_cli_args as cli_args
from pulse.runtime.bootstrap_isaaclab import bootstrap_isaaclab


WEEK1_FIXED_COMMANDS: list[dict[str, float | str]] = [
    {"name": "forward", "vx": 0.8, "vy": 0.0, "yaw": 0.0},
    # "turn" is evaluated as a circular-motion turn rather than an in-place spin.
    # This better matches the current policy's supported motion manifold and the
    # payload robustness question we care about for the existing checkpoint.
    {"name": "turn", "vx": 0.4, "vy": 0.0, "yaw": 0.8},
    {"name": "diagonal", "vx": 0.5, "vy": 0.3, "yaw": 0.0},
]

WEEK1_METRIC_DEFINITIONS = {
    "mean_tracking_error": "Mean sqrt((vx-vx_cmd)^2 + (vy-vy_cmd)^2 + (yaw-yaw_cmd)^2) over rollout steps.",
    "mean_abs_roll_deg": "Mean absolute roll angle in degrees over rollout steps.",
    "mean_abs_pitch_deg": "Mean absolute pitch angle in degrees over rollout steps.",
    "pitch_rms_deg": "Root-mean-square pitch angle in degrees over rollout steps.",
    "peak_abs_pitch_deg": "Peak absolute pitch angle in degrees observed during the rollout.",
    "mean_action_rate_l2": "Mean action-rate L2 penalty from Isaac Lab reward source over rollout steps.",
    "mean_joint_acc_l2": "Mean joint-acceleration L2 penalty from Isaac Lab reward source over rollout steps.",
    "mean_torque_l2": "Mean joint-torque L2 penalty from Isaac Lab reward source over rollout steps.",
    "mean_abs_power": "Mean absolute mechanical power sum(abs(torque * joint_vel)) over rollout steps.",
    "success_rate": "Fraction of episodes that finish by time_out instead of a failure term.",
    "fall_rate": "Fraction of episodes terminated by a failure term rather than time_out.",
    "survival_time_s": "Mean episode duration in seconds until termination.",
}

WEEK1_SUCCESS_CRITERIA = {
    "forward": {
        "mean_tracking_error_max": 0.25,
        "mean_abs_roll_deg_max": 6.0,
        "mean_abs_pitch_deg_max": 6.0,
        "fall_rate_max": 0.05,
        "survival_time_s_min": 18.0,
    },
    "turn": {
        "mean_tracking_error_max": 0.35,
        "mean_abs_roll_deg_max": 8.0,
        "mean_abs_pitch_deg_max": 8.0,
        "fall_rate_max": 0.10,
        "survival_time_s_min": 16.0,
    },
    "diagonal": {
        "mean_tracking_error_max": 0.35,
        "mean_abs_roll_deg_max": 8.0,
        "mean_abs_pitch_deg_max": 8.0,
        "fall_rate_max": 0.10,
        "survival_time_s_min": 16.0,
    },
}

PAYLOAD_GRID_DEFAULT_MASS_SCALES = [1.0, 1.1, 1.2]
PAYLOAD_GRID_DEFAULT_COM_X_VALUES = [-0.02, 0.0, 0.02]
PAYLOAD_COM_X_PRESET_NAMES = {
    -0.02: "small_backward",
    0.0: "nominal",
    0.02: "small_forward",
}
PAYLOAD_PLOT_METRICS = (
    "mean_abs_pitch_deg",
    "mean_tracking_error",
    "success_rate",
    "fall_rate",
)


def _parse_and_launch_app(parser: argparse.ArgumentParser):
    bootstrap_isaaclab()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    if getattr(args_cli, "video", False):
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    return args_cli, app_launcher, app_launcher.app


def _ensure_min_rsl_rl_version(min_version: str = "3.0.1") -> str:
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) >= version.parse(min_version):
        return installed_version

    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{min_version}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)


def _callable_path(func: Any) -> str:
    if func is None:
        return "None"
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", type(func).__name__))
    if module:
        return f"{module}:{qualname}"
    return str(qualname)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if is_dataclass(value):
        return {field.name: _to_serializable(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {k: _to_serializable(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)


def _task_slug(task_name: str | None) -> str:
    if not task_name:
        return "pulse"
    return task_name.replace(":", "_").replace("/", "_").replace(".", "_").replace("-", "_").lower()


def _output_stem(output_arg: str | None, default_stem: str) -> Path:
    path = Path(output_arg) if output_arg is not None else Path(default_stem)
    if path.suffix:
        path = path.with_suffix("")
    return path


def _write_output_files(stem: Path, payload: dict[str, Any], *, text: str | None = None, markdown: str | None = None) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OUTPUT] JSON: {json_path.resolve()}")
    if text is not None:
        txt_path = stem.with_suffix(".txt")
        txt_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"[OUTPUT] TXT:  {txt_path.resolve()}")
    if markdown is not None:
        md_path = stem.with_suffix(".md")
        md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        print(f"[OUTPUT] MD:   {md_path.resolve()}")


def _format_metric_value(value: float) -> str:
    return f"{value:.3f}"


def _format_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def _parse_float_list(raw: str) -> list[float]:
    values = [token.strip() for token in raw.split(",")]
    parsed = [float(token) for token in values if token]
    if not parsed:
        raise ValueError(f"Expected a comma-separated float list, got: {raw!r}")
    return parsed


def _format_signed_value(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _payload_preset_name(com_x: float) -> str:
    rounded = round(float(com_x), 4)
    return PAYLOAD_COM_X_PRESET_NAMES.get(rounded, "custom")


def _payload_scenario_label(mass_scale: float, com_x: float) -> str:
    preset = _payload_preset_name(com_x)
    suffix = f" ({preset})" if preset != "custom" else ""
    return f"mass={mass_scale:.1f}, com_x={_format_signed_value(com_x)}{suffix}"


def _payload_scenario_key(mass_scale: float, com_x: float) -> str:
    mass_token = f"{mass_scale:.2f}".replace("-", "neg").replace(".", "p")
    com_token = f"{com_x:+.3f}".replace("+", "pos").replace("-", "neg").replace(".", "p")
    return f"mass_{mass_token}__com_x_{com_token}"


def _payload_grid_index(values: list[float], target: float) -> int:
    rounded_target = round(float(target), 6)
    for idx, value in enumerate(values):
        if round(float(value), 6) == rounded_target:
            return idx
    raise KeyError(f"Value {target} not found in payload grid {values}.")


def _select_payload_reference_scenarios(
    scenarios: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not scenarios:
        raise ValueError("At least one payload scenario is required.")

    nominal = min(
        scenarios,
        key=lambda scenario: (
            abs(float(scenario["mass_scale"]) - 1.0) + abs(float(scenario["com_x"])),
            abs(float(scenario["mass_scale"]) - 1.0),
            abs(float(scenario["com_x"])),
        ),
    )
    disturbed = max(
        scenarios,
        key=lambda scenario: (
            float(scenario["mass_scale"]),
            float(scenario["com_x"]),
            float(scenario["mean_abs_pitch_deg"]),
        ),
    )
    return nominal, disturbed


def _build_payload_metric_grid(
    scenarios: list[dict[str, Any]],
    mass_scales: list[float],
    com_x_values: list[float],
    metric: str,
):
    import numpy as np

    grid = np.full((len(mass_scales), len(com_x_values)), np.nan, dtype=float)
    for scenario in scenarios:
        row = _payload_grid_index(mass_scales, float(scenario["mass_scale"]))
        col = _payload_grid_index(com_x_values, float(scenario["com_x"]))
        grid[row, col] = float(scenario[metric])
    return grid


def _save_payload_heatmap(
    output_path: Path,
    scenarios: list[dict[str, Any]],
    mass_scales: list[float],
    com_x_values: list[float],
    *,
    metric: str,
    title: str,
    nominal_ref: dict[str, Any],
    disturbed_ref: dict[str, Any],
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    metric_grid = _build_payload_metric_grid(scenarios, mass_scales, com_x_values, metric)
    success_grid = _build_payload_metric_grid(scenarios, mass_scales, com_x_values, "success_rate")
    fall_grid = _build_payload_metric_grid(scenarios, mass_scales, com_x_values, "fall_rate")

    fig, ax = plt.subplots(figsize=(8.4, 6.0), constrained_layout=True)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f3f3f3")
    im = ax.imshow(metric_grid, cmap=cmap, aspect="equal")

    for row in range(metric_grid.shape[0]):
        for col in range(metric_grid.shape[1]):
            value = metric_grid[row, col]
            if np.isnan(value):
                continue
            annotation = (
                f"{value:.3f}\n"
                f"succ {success_grid[row, col]:.0%} | fall {fall_grid[row, col]:.0%}"
            )
            text_color = "white" if value >= np.nanmean(metric_grid) else "#222222"
            ax.text(col, row, annotation, ha="center", va="center", fontsize=10, color=text_color)

    ax.set_title(title)
    ax.set_xlabel("CoM x offset (m)")
    ax.set_ylabel("Mass scale")
    ax.set_xticks(range(len(com_x_values)), [_format_signed_value(value) for value in com_x_values])
    ax.set_yticks(range(len(mass_scales)), [f"{value:.1f}" for value in mass_scales])
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(metric)

    nominal_row = _payload_grid_index(mass_scales, float(nominal_ref["mass_scale"]))
    nominal_col = _payload_grid_index(com_x_values, float(nominal_ref["com_x"]))
    disturbed_row = _payload_grid_index(mass_scales, float(disturbed_ref["mass_scale"]))
    disturbed_col = _payload_grid_index(com_x_values, float(disturbed_ref["com_x"]))
    ax.add_patch(Rectangle((nominal_col - 0.5, nominal_row - 0.5), 1, 1, fill=False, lw=2.5, ec="#1f77b4"))
    ax.add_patch(Rectangle((disturbed_col - 0.5, disturbed_row - 0.5), 1, 1, fill=False, lw=2.5, ec="#111111"))
    ax.text(
        nominal_col,
        nominal_row - 0.62,
        "nominal",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f77b4",
        fontweight="bold",
    )
    ax.text(
        disturbed_col,
        disturbed_row + 0.62,
        "disturbed",
        ha="center",
        va="top",
        fontsize=9,
        color="#111111",
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] PNG:  {output_path.resolve()}")


def _save_nominal_vs_disturbed_bar_chart(
    output_path: Path,
    nominal: dict[str, Any],
    disturbed: dict[str, Any],
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nominal_map = {summary["command_name"]: summary for summary in nominal["per_command"]}
    disturbed_map = {summary["command_name"]: summary for summary in disturbed["per_command"]}
    command_names = [command["name"] for command in WEEK1_FIXED_COMMANDS]
    x = np.arange(len(command_names))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    panels = [
        ("mean_abs_pitch_deg", "Mean |pitch| (deg)", axes[0, 0]),
        ("mean_tracking_error", "Mean tracking error", axes[0, 1]),
        ("success_rate", "Success rate", axes[1, 0]),
        ("fall_rate", "Fall rate", axes[1, 1]),
    ]

    nominal_label = _payload_scenario_label(float(nominal["mass_scale"]), float(nominal["com_x"]))
    disturbed_label = _payload_scenario_label(float(disturbed["mass_scale"]), float(disturbed["com_x"]))
    legend_labels = [f"Nominal: {nominal_label}", f"Disturbed: {disturbed_label}"]

    for metric, ylabel, ax in panels:
        nominal_values = [float(nominal_map[name][metric]) for name in command_names]
        disturbed_values = [float(disturbed_map[name][metric]) for name in command_names]
        bars_nominal = ax.bar(x - width / 2, nominal_values, width, label=legend_labels[0], color="#4c78a8")
        bars_disturbed = ax.bar(x + width / 2, disturbed_values, width, label=legend_labels[1], color="#f58518")
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, command_names)
        if metric in {"success_rate", "fall_rate"}:
            ax.set_ylim(0.0, 1.05)
        else:
            upper = max(max(nominal_values), max(disturbed_values))
            ax.set_ylim(0.0, upper * 1.22 if upper > 0 else 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars_nominal, padding=3, fontsize=8, fmt="%.3f")
        ax.bar_label(bars_disturbed, padding=3, fontsize=8, fmt="%.3f")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Nominal vs disturbed payload: per-command breakdown", fontsize=14, y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] PNG:  {output_path.resolve()}")


def _format_payload_scenario_table(scenarios: list[dict[str, Any]]) -> str:
    rows = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario["label"],
                "mass_scale": scenario["mass_scale"],
                "com_x": scenario["com_x"],
                "mean_tracking_error": _format_metric_value(scenario["mean_tracking_error"]),
                "mean_abs_pitch_deg": _format_metric_value(scenario["mean_abs_pitch_deg"]),
                "peak_abs_pitch_deg": _format_metric_value(scenario["peak_abs_pitch_deg"]),
                "mean_action_rate_l2": _format_metric_value(scenario["mean_action_rate_l2"]),
                "mean_torque_l2": _format_metric_value(scenario["mean_torque_l2"]),
                "mean_abs_power": _format_metric_value(scenario["mean_abs_power"]),
                "success_rate": _format_metric_value(scenario["success_rate"]),
                "fall_rate": _format_metric_value(scenario["fall_rate"]),
                "survival_time_s": _format_metric_value(scenario["survival_time_s"]),
            }
        )
    return _format_markdown_table(
        rows,
        [
            ("scenario", "scenario"),
            ("mass_scale", "mass_scale"),
            ("com_x", "com_x"),
            ("mean_tracking_error", "mean_tracking_error"),
            ("mean_abs_pitch_deg", "mean_abs_pitch_deg"),
            ("peak_abs_pitch_deg", "peak_abs_pitch_deg"),
            ("mean_action_rate_l2", "mean_action_rate_l2"),
            ("mean_torque_l2", "mean_torque_l2"),
            ("mean_abs_power", "mean_abs_power"),
            ("success_rate", "success_rate"),
            ("fall_rate", "fall_rate"),
            ("survival_time_s", "survival_time_s"),
        ],
    )


def _format_payload_per_command_sections(scenarios: list[dict[str, Any]]) -> str:
    sections = []
    for scenario in scenarios:
        sections.extend(
            [
                f"## {scenario['label']}",
                "",
                _format_command_suite_table(scenario["per_command"]),
                "",
            ]
        )
    while sections and sections[-1] == "":
        sections.pop()
    return "\n".join(sections)


def _format_payload_audit_table(scenarios: list[dict[str, Any]]) -> str:
    rows = []
    for scenario in scenarios:
        audit = scenario["payload_audit"]
        rows.append(
            {
                "scenario": scenario["label"],
                "nominal_base_mass": _format_metric_value(audit["nominal_base_mass"]),
                "applied_base_mass": _format_metric_value(audit["applied_base_mass"]),
                "nominal_base_com_x": _format_metric_value(float(audit["nominal_base_com"][0])),
                "applied_base_com_x": _format_metric_value(float(audit["applied_base_com"][0])),
                "applied_mass_scale_vs_nominal": _format_metric_value(audit["applied_mass_scale_vs_nominal"]),
                "applied_com_x_delta": _format_metric_value(audit["applied_com_x_delta"]),
            }
        )
    return _format_markdown_table(
        rows,
        [
            ("scenario", "scenario"),
            ("nominal_base_mass", "nominal_base_mass"),
            ("applied_base_mass", "applied_base_mass"),
            ("nominal_base_com_x", "nominal_base_com_x"),
            ("applied_base_com_x", "applied_base_com_x"),
            ("applied_mass_scale_vs_nominal", "applied_mass_scale_vs_nominal"),
            ("applied_com_x_delta", "applied_com_x_delta"),
        ],
    )


def _resolve_resume_path(args_cli, agent_cfg, log_root_path: str) -> str:
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import get_checkpoint_path

    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _make_policy_reset(policy):
    def _reset(dones: torch.Tensor) -> None:
        if hasattr(policy, "reset"):
            policy.reset(dones)

    return _reset


def _collect_env_audit(env, task_name: str | None) -> dict[str, Any]:
    unwrapped = env.unwrapped
    cfg = unwrapped.cfg

    observations = []
    obs_manager = unwrapped.observation_manager
    for group_name, term_names in obs_manager.active_terms.items():
        term_dims = obs_manager.group_obs_term_dim[group_name]
        observations.append(
            {
                "group": group_name,
                "shape": obs_manager.group_obs_dim[group_name],
                "concatenate_terms": obs_manager.group_obs_concatenate[group_name],
                "terms": [
                    {"name": name, "shape": term_dims[idx]}
                    for idx, name in enumerate(term_names)
                ],
            }
        )

    actions = []
    for idx, term_name in enumerate(unwrapped.action_manager.active_terms):
        cfg_term = getattr(cfg.actions, term_name, None)
        actions.append(
            {
                "index": idx,
                "name": term_name,
                "dimension": unwrapped.action_manager.action_term_dim[idx],
                "cfg_type": type(cfg_term).__name__ if cfg_term is not None else "Unknown",
                "scale": getattr(cfg_term, "scale", None),
                "joint_names": getattr(cfg_term, "joint_names", None),
                "clip": getattr(cfg_term, "clip", None),
                "use_default_offset": getattr(cfg_term, "use_default_offset", None),
            }
        )

    rewards = []
    for idx, term_name in enumerate(unwrapped.reward_manager.active_terms):
        term_cfg = unwrapped.reward_manager.get_term_cfg(term_name)
        rewards.append(
            {
                "index": idx,
                "name": term_name,
                "weight": term_cfg.weight,
                "func": _callable_path(term_cfg.func),
                "params": _to_serializable(term_cfg.params),
            }
        )

    terminations = []
    for idx, term_name in enumerate(unwrapped.termination_manager.active_terms):
        term_cfg = unwrapped.termination_manager.get_term_cfg(term_name)
        terminations.append(
            {
                "index": idx,
                "name": term_name,
                "time_out": bool(term_cfg.time_out),
                "failure": not bool(term_cfg.time_out),
                "func": _callable_path(term_cfg.func),
                "params": _to_serializable(term_cfg.params),
            }
        )

    commands = []
    if is_dataclass(cfg.commands):
        for field in fields(cfg.commands):
            cmd_cfg = getattr(cfg.commands, field.name)
            if cmd_cfg is None:
                continue
            commands.append(
                {
                    "name": field.name,
                    "cfg_type": type(cmd_cfg).__name__,
                    "heading_command": getattr(cmd_cfg, "heading_command", None),
                    "rel_standing_envs": getattr(cmd_cfg, "rel_standing_envs", None),
                    "rel_heading_envs": getattr(cmd_cfg, "rel_heading_envs", None),
                    "resampling_time_range": getattr(cmd_cfg, "resampling_time_range", None),
                    "ranges": _to_serializable(getattr(cmd_cfg, "ranges", None)),
                }
            )

    return {
        "task": task_name,
        "env_cfg_type": type(cfg).__name__,
        "num_envs": unwrapped.num_envs,
        "device": str(unwrapped.device),
        "step_dt": getattr(unwrapped, "step_dt", None),
        "max_episode_length": getattr(unwrapped, "max_episode_length", None),
        "max_episode_length_s": getattr(unwrapped, "max_episode_length_s", None),
        "height_scanner_enabled": cfg.scene.height_scanner is not None,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminations": terminations,
        "commands": commands,
    }


def _format_audit_text(audit: dict[str, Any]) -> str:
    lines = [
        "PULSE Env Audit",
        f"task: {audit['task']}",
        f"env_cfg_type: {audit['env_cfg_type']}",
        f"num_envs: {audit['num_envs']}",
        f"device: {audit['device']}",
        f"step_dt: {audit['step_dt']}",
        f"max_episode_length: {audit['max_episode_length']}",
        f"max_episode_length_s: {audit['max_episode_length_s']}",
        f"height_scanner_enabled: {audit['height_scanner_enabled']}",
        "",
        "=== Observations ===",
    ]
    for group in audit["observations"]:
        lines.append(f"[{group['group']}] shape={group['shape']} concatenate_terms={group['concatenate_terms']}")
        for term in group["terms"]:
            lines.append(f"  - {term['name']}: {term['shape']}")
    lines.append("")
    lines.append("=== Actions ===")
    for action in audit["actions"]:
        lines.append(
            f"[{action['name']}] dim={action['dimension']} type={action['cfg_type']} "
            f"scale={action['scale']} joint_names={action['joint_names']}"
        )
    lines.append("")
    lines.append("=== Rewards ===")
    for reward in audit["rewards"]:
        lines.append(f"[{reward['name']}] weight={reward['weight']} func={reward['func']}")
    lines.append("")
    lines.append("=== Terminations ===")
    for term in audit["terminations"]:
        lines.append(
            f"[{term['name']}] time_out={term['time_out']} failure={term['failure']} func={term['func']}"
        )
    lines.append("")
    lines.append("=== Commands ===")
    for cmd in audit["commands"]:
        lines.append(
            f"[{cmd['name']}] heading_command={cmd['heading_command']} "
            f"rel_standing_envs={cmd['rel_standing_envs']} rel_heading_envs={cmd['rel_heading_envs']} "
            f"ranges={cmd['ranges']}"
        )
    return "\n".join(lines)


def _set_fixed_command(unwrapped, command_case: dict[str, float | str]) -> None:
    cmd_cfg = unwrapped.cfg.commands.base_velocity
    cmd_term = unwrapped.command_manager.get_term("base_velocity")

    for target in (cmd_cfg, cmd_term.cfg):
        target.heading_command = False
        target.rel_standing_envs = 0.0
        target.rel_heading_envs = 0.0
        target.resampling_time_range = (1.0e6, 1.0e6)
        target.ranges.lin_vel_x = (float(command_case["vx"]), float(command_case["vx"]))
        target.ranges.lin_vel_y = (float(command_case["vy"]), float(command_case["vy"]))
        target.ranges.ang_vel_z = (float(command_case["yaw"]), float(command_case["yaw"]))
        if hasattr(target.ranges, "heading"):
            target.ranges.heading = (0.0, 0.0)

    cmd_term.vel_command_b[:, 0] = float(command_case["vx"])
    cmd_term.vel_command_b[:, 1] = float(command_case["vy"])
    cmd_term.vel_command_b[:, 2] = float(command_case["yaw"])
    cmd_term.is_heading_env[:] = False
    cmd_term.is_standing_env[:] = False
    cmd_term.time_left[:] = 1.0e6
    cmd_term.command_counter[:] = 0


def _capture_payload_reference(unwrapped) -> dict[str, Any]:
    robot = unwrapped.scene["robot"]
    base_body_ids, base_body_names = robot.find_bodies("base")
    if not base_body_ids:
        raise RuntimeError("Unable to find body named 'base' for payload evaluation.")
    base_body_id = int(base_body_ids[0])
    return {
        "robot": robot,
        "base_body_id": base_body_id,
        "base_body_name": base_body_names[0],
        "default_mass": robot.data.default_mass.clone(),
        "default_inertia": robot.data.default_inertia.clone(),
        "default_coms": robot.root_physx_view.get_coms().clone(),
    }


def _apply_payload_condition(
    unwrapped,
    payload_ref: dict[str, Any],
    *,
    mass_scale: float,
    com_x_offset: float,
) -> None:
    robot = payload_ref["robot"]
    base_body_id = payload_ref["base_body_id"]
    env_ids = torch.arange(unwrapped.num_envs, dtype=torch.int, device="cpu")

    masses = payload_ref["default_mass"].clone()
    masses[:, base_body_id] = payload_ref["default_mass"][:, base_body_id] * mass_scale
    robot.root_physx_view.set_masses(masses, env_ids)

    inertias = payload_ref["default_inertia"].clone()
    inertias[:, base_body_id] = payload_ref["default_inertia"][:, base_body_id] * mass_scale
    robot.root_physx_view.set_inertias(inertias, env_ids)

    coms = payload_ref["default_coms"].clone()
    coms[:, base_body_id, 0] = payload_ref["default_coms"][:, base_body_id, 0] + com_x_offset
    robot.root_physx_view.set_coms(coms, env_ids)


def _collect_payload_audit(unwrapped, payload_ref: dict[str, Any]) -> dict[str, Any]:
    robot = payload_ref["robot"]
    base_body_id = payload_ref["base_body_id"]
    applied_masses = robot.root_physx_view.get_masses()
    applied_coms = robot.root_physx_view.get_coms()
    nominal_mass = float(payload_ref["default_mass"][0, base_body_id].item())
    applied_mass = float(applied_masses[0, base_body_id].item())
    nominal_com = payload_ref["default_coms"][0, base_body_id].detach().cpu().tolist()
    applied_com = applied_coms[0, base_body_id].detach().cpu().tolist()
    return {
        "base_body_name": payload_ref["base_body_name"],
        "nominal_base_mass": nominal_mass,
        "applied_base_mass": applied_mass,
        "nominal_base_com": nominal_com,
        "applied_base_com": applied_com,
        "applied_mass_scale_vs_nominal": applied_mass / nominal_mass if nominal_mass != 0.0 else None,
        "applied_com_x_delta": float(applied_com[0] - nominal_com[0]),
    }


def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("No episodes were collected.")
    num_episodes = len(episodes)
    termination_counts: dict[str, int] = {}
    for episode in episodes:
        reason = episode["termination_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    return {
        "num_episodes": num_episodes,
        "mean_tracking_error": sum(ep["mean_tracking_error"] for ep in episodes) / num_episodes,
        "mean_abs_roll_deg": sum(ep["mean_abs_roll_deg"] for ep in episodes) / num_episodes,
        "mean_abs_pitch_deg": sum(ep["mean_abs_pitch_deg"] for ep in episodes) / num_episodes,
        "pitch_rms_deg": sum(ep["pitch_rms_deg"] for ep in episodes) / num_episodes,
        "peak_abs_pitch_deg": sum(ep["peak_abs_pitch_deg"] for ep in episodes) / num_episodes,
        "mean_action_rate_l2": sum(ep["mean_action_rate_l2"] for ep in episodes) / num_episodes,
        "mean_joint_acc_l2": sum(ep["mean_joint_acc_l2"] for ep in episodes) / num_episodes,
        "mean_torque_l2": sum(ep["mean_torque_l2"] for ep in episodes) / num_episodes,
        "mean_abs_power": sum(ep["mean_abs_power"] for ep in episodes) / num_episodes,
        "fall_rate": sum(1.0 for ep in episodes if ep["fell"]) / num_episodes,
        "survival_time_s": sum(ep["survival_time_s"] for ep in episodes) / num_episodes,
        "success_rate": sum(1.0 for ep in episodes if not ep["fell"]) / num_episodes,
        "termination_counts": termination_counts,
    }


def _run_fixed_command_rollouts(
    env,
    policy,
    policy_reset,
    command_case: dict[str, float | str],
    num_episodes: int,
    *,
    payload_ref: dict[str, Any] | None = None,
    mass_scale: float = 1.0,
    com_x_offset: float = 0.0,
) -> list[dict[str, Any]]:
    from isaaclab.utils.math import euler_xyz_from_quat

    unwrapped = env.unwrapped
    _set_fixed_command(unwrapped, command_case)
    obs, _ = env.reset()
    if payload_ref is not None:
        _apply_payload_condition(unwrapped, payload_ref, mass_scale=mass_scale, com_x_offset=com_x_offset)
        obs = env.get_observations()
    _set_fixed_command(unwrapped, command_case)

    num_envs = unwrapped.num_envs
    accum_tracking_error = torch.zeros(num_envs, device=unwrapped.device)
    accum_roll_deg = torch.zeros(num_envs, device=unwrapped.device)
    accum_pitch_deg = torch.zeros(num_envs, device=unwrapped.device)
    accum_pitch_sq_deg = torch.zeros(num_envs, device=unwrapped.device)
    peak_pitch_deg = torch.zeros(num_envs, device=unwrapped.device)
    accum_action_rate_l2 = torch.zeros(num_envs, device=unwrapped.device)
    accum_joint_acc_l2 = torch.zeros(num_envs, device=unwrapped.device)
    accum_torque_l2 = torch.zeros(num_envs, device=unwrapped.device)
    accum_abs_power = torch.zeros(num_envs, device=unwrapped.device)
    accum_steps = torch.zeros(num_envs, dtype=torch.long, device=unwrapped.device)
    episodes: list[dict[str, Any]] = []
    max_steps = int(max(unwrapped.max_episode_length * max(num_episodes, 1), unwrapped.max_episode_length * 10))
    step_counter = 0

    while len(episodes) < num_episodes and step_counter < max_steps:
        step_counter += 1
        _set_fixed_command(unwrapped, command_case)
        # Isaac Lab reset/step paths perform stateful in-place writes during rollout.
        # `torch.inference_mode()` can leak inference tensors into these reset paths.
        # `torch.no_grad()` avoids that failure mode while still disabling gradients.
        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_reset(dones)

        robot = unwrapped.scene["robot"]
        command = unwrapped.command_manager.get_command("base_velocity")
        tracking_error = torch.sqrt(
            torch.square(robot.data.root_lin_vel_b[:, 0] - command[:, 0])
            + torch.square(robot.data.root_lin_vel_b[:, 1] - command[:, 1])
            + torch.square(robot.data.root_ang_vel_b[:, 2] - command[:, 2])
        )
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        abs_pitch_deg = torch.rad2deg(torch.abs(pitch))
        action_rate_l2 = torch.sum(
            torch.square(unwrapped.action_manager.action - unwrapped.action_manager.prev_action),
            dim=1,
        )
        joint_acc_l2 = torch.sum(torch.square(robot.data.joint_acc), dim=1)
        torque_l2 = torch.sum(torch.square(robot.data.applied_torque), dim=1)
        abs_power = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)

        accum_tracking_error += tracking_error
        accum_roll_deg += torch.rad2deg(torch.abs(roll))
        accum_pitch_deg += abs_pitch_deg
        accum_pitch_sq_deg += torch.square(abs_pitch_deg)
        peak_pitch_deg = torch.maximum(peak_pitch_deg, abs_pitch_deg)
        accum_action_rate_l2 += action_rate_l2
        accum_joint_acc_l2 += joint_acc_l2
        accum_torque_l2 += torque_l2
        accum_abs_power += abs_power
        accum_steps += 1

        done_mask = dones.to(dtype=torch.bool)
        done_ids = done_mask.nonzero(as_tuple=False).flatten()
        if len(done_ids) == 0:
            continue

        term_manager = unwrapped.termination_manager
        for env_id in done_ids.tolist():
            steps = int(accum_steps[env_id].item())
            if steps <= 0:
                continue
            active_terms = [
                term_name
                for term_name in term_manager.active_terms
                if bool(term_manager.get_term(term_name)[env_id].item())
            ]
            failure_terms = [
                term_name
                for term_name in active_terms
                if not term_manager.get_term_cfg(term_name).time_out
            ]
            termination_reason = failure_terms[0] if failure_terms else (active_terms[0] if active_terms else "unknown")
            fell = any(not term_manager.get_term_cfg(term_name).time_out for term_name in active_terms)
            episodes.append(
                {
                    "command_name": str(command_case["name"]),
                    "vx": float(command_case["vx"]),
                    "vy": float(command_case["vy"]),
                    "yaw": float(command_case["yaw"]),
                    "mean_tracking_error": float(accum_tracking_error[env_id].item() / steps),
                    "mean_abs_roll_deg": float(accum_roll_deg[env_id].item() / steps),
                    "mean_abs_pitch_deg": float(accum_pitch_deg[env_id].item() / steps),
                    "pitch_rms_deg": float(torch.sqrt(accum_pitch_sq_deg[env_id] / steps).item()),
                    "peak_abs_pitch_deg": float(peak_pitch_deg[env_id].item()),
                    "mean_action_rate_l2": float(accum_action_rate_l2[env_id].item() / steps),
                    "mean_joint_acc_l2": float(accum_joint_acc_l2[env_id].item() / steps),
                    "mean_torque_l2": float(accum_torque_l2[env_id].item() / steps),
                    "mean_abs_power": float(accum_abs_power[env_id].item() / steps),
                    "survival_time_s": float(steps * unwrapped.step_dt),
                    "termination_reason": termination_reason,
                    "fell": bool(fell),
                }
            )
            accum_tracking_error[env_id] = 0.0
            accum_roll_deg[env_id] = 0.0
            accum_pitch_deg[env_id] = 0.0
            accum_pitch_sq_deg[env_id] = 0.0
            peak_pitch_deg[env_id] = 0.0
            accum_action_rate_l2[env_id] = 0.0
            accum_joint_acc_l2[env_id] = 0.0
            accum_torque_l2[env_id] = 0.0
            accum_abs_power[env_id] = 0.0
            accum_steps[env_id] = 0
            if len(episodes) >= num_episodes:
                break

    if len(episodes) < num_episodes:
        raise RuntimeError(
            f"Collected only {len(episodes)} episodes for command '{command_case['name']}' before hitting safety cap."
        )
    return episodes[:num_episodes]


def _command_passes(command_name: str, summary: dict[str, Any]) -> bool:
    criteria = WEEK1_SUCCESS_CRITERIA[command_name]
    return (
        summary["mean_tracking_error"] <= criteria["mean_tracking_error_max"]
        and summary["mean_abs_roll_deg"] <= criteria["mean_abs_roll_deg_max"]
        and summary["mean_abs_pitch_deg"] <= criteria["mean_abs_pitch_deg_max"]
        and summary["fall_rate"] <= criteria["fall_rate_max"]
        and summary["survival_time_s"] >= criteria["survival_time_s_min"]
    )


def _run_command_suite(
    env,
    policy,
    policy_reset,
    *,
    num_episodes: int,
    payload_ref: dict[str, Any] | None = None,
    mass_scale: float = 1.0,
    com_x_offset: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    all_episodes: list[dict[str, Any]] = []
    for command_case in WEEK1_FIXED_COMMANDS:
        print(
            "[EVAL] command="
            f"{command_case['name']} vx={command_case['vx']} vy={command_case['vy']} yaw={command_case['yaw']}"
        )
        episodes = _run_fixed_command_rollouts(
            env,
            policy,
            policy_reset,
            command_case,
            num_episodes,
            payload_ref=payload_ref,
            mass_scale=mass_scale,
            com_x_offset=com_x_offset,
        )
        summary = _summarize_episodes(episodes)
        summary["command_name"] = command_case["name"]
        summary["vx"] = command_case["vx"]
        summary["vy"] = command_case["vy"]
        summary["yaw"] = command_case["yaw"]
        summary["criteria"] = WEEK1_SUCCESS_CRITERIA[str(command_case["name"])]
        summary["pass"] = _command_passes(str(command_case["name"]), summary)
        summaries.append(summary)
        all_episodes.extend(episodes)
    return summaries, all_episodes


def _format_command_suite_table(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "command": summary["command_name"],
                "vx": summary["vx"],
                "vy": summary["vy"],
                "yaw": summary["yaw"],
                "mean_tracking_error": _format_metric_value(summary["mean_tracking_error"]),
                "mean_abs_roll_deg": _format_metric_value(summary["mean_abs_roll_deg"]),
                "mean_abs_pitch_deg": _format_metric_value(summary["mean_abs_pitch_deg"]),
                "peak_abs_pitch_deg": _format_metric_value(summary["peak_abs_pitch_deg"]),
                "mean_action_rate_l2": _format_metric_value(summary["mean_action_rate_l2"]),
                "mean_torque_l2": _format_metric_value(summary["mean_torque_l2"]),
                "mean_abs_power": _format_metric_value(summary["mean_abs_power"]),
                "fall_rate": _format_metric_value(summary["fall_rate"]),
                "survival_time_s": _format_metric_value(summary["survival_time_s"]),
                "pass": "PASS" if summary["pass"] else "FAIL",
            }
        )
    return _format_markdown_table(
        rows,
        [
            ("command", "command"),
            ("vx", "vx"),
            ("vy", "vy"),
            ("yaw", "yaw"),
            ("mean_tracking_error", "mean_tracking_error"),
            ("mean_abs_roll_deg", "mean_abs_roll_deg"),
            ("mean_abs_pitch_deg", "mean_abs_pitch_deg"),
            ("peak_abs_pitch_deg", "peak_abs_pitch_deg"),
            ("mean_action_rate_l2", "mean_action_rate_l2"),
            ("mean_torque_l2", "mean_torque_l2"),
            ("mean_abs_power", "mean_abs_power"),
            ("fall_rate", "fall_rate"),
            ("survival_time_s", "survival_time_s"),
            ("pass", "pass"),
        ],
    )


def _assess_stable_degradation(nominal: dict[str, Any], disturbed: dict[str, Any]) -> dict[str, Any]:
    has_degradation = (
        disturbed["mean_tracking_error"] > nominal["mean_tracking_error"]
        or disturbed["mean_abs_roll_deg"] > nominal["mean_abs_roll_deg"]
        or disturbed["mean_abs_pitch_deg"] > nominal["mean_abs_pitch_deg"]
        or disturbed["fall_rate"] > nominal["fall_rate"]
        or disturbed["survival_time_s"] < nominal["survival_time_s"]
    )
    no_collapse = disturbed["fall_rate"] <= 0.25 and disturbed["survival_time_s"] >= 0.75 * nominal["survival_time_s"]
    return {
        "has_degradation": has_degradation,
        "no_collapse": no_collapse,
        "stable_degradation": has_degradation and no_collapse,
    }


def _train_main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
    parser.add_argument(
        "--ray-proc-id",
        "-rid",
        type=int,
        default=None,
        help="Automatically configured by Ray integration, otherwise None.",
    )
    cli_args.add_rsl_rl_args(parser)
    args_cli, app_launcher, simulation_app = _parse_and_launch_app(parser)
    installed_version = _ensure_min_rsl_rl_version("3.0.1")

    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    logger = logging.getLogger(__name__)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training is not supported when using CPU device.")
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning("IO descriptors are only supported for manager based RL environments.")

        env_cfg.log_dir = log_dir
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if args_cli.video:
            from isaaclab.utils.dict import print_dict

            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()

    main()
    simulation_app.close()


def _play_main() -> None:
    parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument("--fixed_vx", type=float, default=None, help="Fix linear x velocity command (m/s).")
    parser.add_argument("--fixed_vy", type=float, default=None, help="Fix linear y velocity command (m/s).")
    parser.add_argument("--fixed_yaw", type=float, default=None, help="Fix yaw rate command (rad/s).")
    cli_args.add_rsl_rl_args(parser)
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)
    installed_version = metadata.version("rsl-rl-lib")

    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict
    from isaaclab_rl.rsl_rl import (
        RslRlBaseRunnerCfg,
        RslRlVecEnvWrapper,
        export_policy_as_jit,
        export_policy_as_onnx,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.fixed_vx is not None or args_cli.fixed_vy is not None or args_cli.fixed_yaw is not None:
            cmd = env_cfg.commands.base_velocity
            cmd.heading_command = False
            vx = args_cli.fixed_vx if args_cli.fixed_vx is not None else 0.0
            vy = args_cli.fixed_vy if args_cli.fixed_vy is not None else 0.0
            yaw = args_cli.fixed_yaw if args_cli.fixed_yaw is not None else 0.0
            cmd.ranges.lin_vel_x = (vx, vx)
            cmd.ranges.lin_vel_y = (vy, vy)
            cmd.ranges.ang_vel_z = (yaw, yaw)
            print(f"[PLAY] Fixed command: vx={vx}  vy={vy}  yaw={yaw}")

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None
        else:
            policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                elif policy_nn is not None:
                    policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()

    main()
    simulation_app.close()


def _inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and export auditable PULSE env details.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate for inspection.")
    parser.add_argument("--output", type=str, default=None, help="Output stem for txt/json audit export.")
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            audit = _collect_env_audit(env, args_cli.task)
            text = _format_audit_text(audit)
            print(text)
            stem = _output_stem(args_cli.output, f"results/audit/{_task_slug(args_cli.task)}")
            _write_output_files(stem, audit, text=text)
        finally:
            env.close()

    main()
    simulation_app.close()


def _eval_fixed_commands_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Week 1 fixed-command baseline for PULSE.")
    parser.add_argument("--task", type=str, default=None, help="Registered task name.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point."
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="Completed episodes per command.")
    parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
    parser.add_argument("--output", type=str, default=None, help="Output stem for json/md export.")
    cli_args.add_rsl_rl_args(parser)
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.observations.policy.enable_corruption = False
        for event_name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_resume_path(args_cli, agent_cfg, log_root_path)
        print(f"[EVAL-FIXED] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            policy_reset = _make_policy_reset(policy)

            summaries, all_episodes = _run_command_suite(
                env,
                policy,
                policy_reset,
                num_episodes=args_cli.num_episodes,
            )
            overall_pass = all(summary["pass"] for summary in summaries)
            payload = {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "metric_definitions": WEEK1_METRIC_DEFINITIONS,
                "success_criteria": WEEK1_SUCCESS_CRITERIA,
                "num_episodes_per_command": args_cli.num_episodes,
                "per_command": summaries,
                "episodes": all_episodes,
                "overall_pass": overall_pass,
            }
            markdown = "\n".join(
                [
                    "# Week 1 Fixed Command Evaluation",
                    "",
                    _format_command_suite_table(summaries),
                    "",
                    f"overall_pass: {'PASS' if overall_pass else 'FAIL'}",
                ]
            )
            stem = _output_stem(args_cli.output, f"results/eval_fixed_commands/{_task_slug(args_cli.task)}")
            _write_output_files(stem, payload, markdown=markdown)
        finally:
            env.close()

    main()
    simulation_app.close()


def _eval_payload_main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 1 payload robustness grid evaluation.")
    parser.add_argument("--task", type=str, default=None, help="Registered task name.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point."
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="Completed episodes per command per scenario.")
    parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
    parser.add_argument(
        "--mass_scales",
        type=str,
        default=",".join(str(value) for value in PAYLOAD_GRID_DEFAULT_MASS_SCALES),
        help="Comma-separated base mass scales for the payload grid.",
    )
    parser.add_argument(
        "--com_x_values",
        type=str,
        default=",".join(str(value) for value in PAYLOAD_GRID_DEFAULT_COM_X_VALUES),
        help="Comma-separated base CoM x offsets in meters for the payload grid.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output stem for json/md export.")
    cli_args.add_rsl_rl_args(parser)
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.observations.policy.enable_corruption = False
        for event_name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_resume_path(args_cli, agent_cfg, log_root_path)
        print(f"[EVAL-PAYLOAD] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            policy_reset = _make_policy_reset(policy)
            payload_ref = _capture_payload_reference(env.unwrapped)

            mass_scales = sorted(_parse_float_list(args_cli.mass_scales))
            com_x_values = sorted(_parse_float_list(args_cli.com_x_values))
            scenario_cfgs = []
            for mass_scale in mass_scales:
                for com_x in com_x_values:
                    scenario_cfgs.append(
                        {
                            "name": _payload_scenario_key(mass_scale, com_x),
                            "label": _payload_scenario_label(mass_scale, com_x),
                            "mass_scale": mass_scale,
                            "com_x": com_x,
                            "com_x_preset": _payload_preset_name(com_x),
                        }
                    )

            scenarios = []
            for scenario_cfg in scenario_cfgs:
                print(
                    f"[EVAL-PAYLOAD] scenario={scenario_cfg['label']} "
                    f"mass_scale={scenario_cfg['mass_scale']} com_x={scenario_cfg['com_x']}"
                )
                summaries, episodes = _run_command_suite(
                    env,
                    policy,
                    policy_reset,
                    num_episodes=args_cli.num_episodes,
                    payload_ref=payload_ref,
                    mass_scale=float(scenario_cfg["mass_scale"]),
                    com_x_offset=float(scenario_cfg["com_x"]),
                )
                aggregate = _summarize_episodes(episodes)
                aggregate["scenario"] = scenario_cfg["name"]
                aggregate["label"] = scenario_cfg["label"]
                aggregate["mass_scale"] = scenario_cfg["mass_scale"]
                aggregate["com_x"] = scenario_cfg["com_x"]
                aggregate["com_x_preset"] = scenario_cfg["com_x_preset"]
                aggregate["payload_audit"] = _collect_payload_audit(env.unwrapped, payload_ref)
                aggregate["per_command"] = summaries
                scenarios.append(aggregate)

            nominal_ref, disturbed_ref = _select_payload_reference_scenarios(scenarios)
            degradation = _assess_stable_degradation(nominal_ref, disturbed_ref)
            stem = _output_stem(args_cli.output, f"results/eval_payload/{_task_slug(args_cli.task)}")
            plot_paths = {
                "mean_abs_pitch_deg_heatmap": stem.with_name(f"{stem.name}_mean_abs_pitch_deg_heatmap.png"),
                "mean_tracking_error_heatmap": stem.with_name(f"{stem.name}_mean_tracking_error_heatmap.png"),
                "nominal_vs_disturbed_per_command": stem.with_name(f"{stem.name}_nominal_vs_disturbed_per_command.png"),
            }
            _save_payload_heatmap(
                plot_paths["mean_abs_pitch_deg_heatmap"],
                scenarios,
                mass_scales,
                com_x_values,
                metric="mean_abs_pitch_deg",
                title="Payload grid heatmap: mean absolute pitch",
                nominal_ref=nominal_ref,
                disturbed_ref=disturbed_ref,
            )
            _save_payload_heatmap(
                plot_paths["mean_tracking_error_heatmap"],
                scenarios,
                mass_scales,
                com_x_values,
                metric="mean_tracking_error",
                title="Payload grid heatmap: mean tracking error",
                nominal_ref=nominal_ref,
                disturbed_ref=disturbed_ref,
            )
            _save_nominal_vs_disturbed_bar_chart(
                plot_paths["nominal_vs_disturbed_per_command"],
                nominal_ref,
                disturbed_ref,
            )
            payload = {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "metric_definitions": WEEK1_METRIC_DEFINITIONS,
                "plot_metrics": PAYLOAD_PLOT_METRICS,
                "mass_scales": mass_scales,
                "com_x_values": com_x_values,
                "scenario_configs": scenario_cfgs,
                "scenarios": scenarios,
                "reference_scenarios": {
                    "nominal": {
                        "scenario": nominal_ref["scenario"],
                        "label": nominal_ref["label"],
                        "mass_scale": nominal_ref["mass_scale"],
                        "com_x": nominal_ref["com_x"],
                    },
                    "disturbed": {
                        "scenario": disturbed_ref["scenario"],
                        "label": disturbed_ref["label"],
                        "mass_scale": disturbed_ref["mass_scale"],
                        "com_x": disturbed_ref["com_x"],
                    },
                },
                "artifacts": {name: str(path) for name, path in plot_paths.items()},
                "degradation_assessment": degradation,
            }
            markdown = "\n".join(
                [
                    "# Week 1 Payload Robustness Grid",
                    "",
                    "## Scenario Summary",
                    "",
                    _format_payload_scenario_table(scenarios),
                    "",
                    "## Figures",
                    "",
                    f"![mean_abs_pitch_deg heatmap]({plot_paths['mean_abs_pitch_deg_heatmap'].name})",
                    "",
                    f"![mean_tracking_error heatmap]({plot_paths['mean_tracking_error_heatmap'].name})",
                    "",
                    f"![nominal vs disturbed per-command]({plot_paths['nominal_vs_disturbed_per_command'].name})",
                    "",
                    "## Payload Audit",
                    "",
                    _format_payload_audit_table(scenarios),
                    "",
                    "## Reference Pair",
                    "",
                    f"nominal: {nominal_ref['label']}",
                    f"disturbed: {disturbed_ref['label']}",
                    "",
                    f"stable_degradation: {degradation['stable_degradation']}",
                    f"has_degradation: {degradation['has_degradation']}",
                    f"no_collapse: {degradation['no_collapse']}",
                    "",
                    "## Per-Command Summary By Cell",
                    "",
                    _format_payload_per_command_sections(scenarios),
                ]
            )
            _write_output_files(stem, payload, markdown=markdown)
        finally:
            env.close()

    main()
    simulation_app.close()


if __name__ == "__main__":
    mode = os.environ.get("PULSE_ENTRY_MODE", "train").strip().lower()
    if mode == "train":
        _train_main()
    elif mode == "play":
        _play_main()
    elif mode == "inspect":
        _inspect_main()
    elif mode == "eval_fixed_commands":
        _eval_fixed_commands_main()
    elif mode == "eval_payload":
        _eval_payload_main()
    else:
        raise ValueError(
            "Invalid PULSE_ENTRY_MODE. Expected 'train', 'play', 'inspect', 'eval_fixed_commands', or 'eval_payload'."
        )
