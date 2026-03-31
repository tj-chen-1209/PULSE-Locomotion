"""Unified train/play runtime entry for PULSE."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable when this file is launched by isaaclab.sh.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from pulse.runtime.bootstrap_isaaclab import bootstrap_isaaclab
from pulse.runtime import rsl_rl_cli_args as cli_args


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

    # Delayed imports keep Omniverse module order correct.
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
    # Fixed-command overrides: pin velocity commands to a single value instead of random sampling.
    parser.add_argument("--fixed_vx", type=float, default=None, help="Fix linear x velocity command (m/s).")
    parser.add_argument("--fixed_vy", type=float, default=None, help="Fix linear y velocity command (m/s).")
    parser.add_argument("--fixed_yaw", type=float, default=None, help="Fix yaw rate command (rad/s).")
    cli_args.add_rsl_rl_args(parser)
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)
    installed_version = metadata.version("rsl-rl-lib")

    # Delayed imports keep Omniverse module order correct.
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

        # Pin velocity commands to fixed values when --fixed_vx/vy/yaw are supplied.
        # Collapse each range to a zero-width interval so the sampler always returns the same value.
        # Also disable heading_command so ang_vel_z is used directly as yaw rate.
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

def _print_obs_terms(env_cfg) -> None:
    """打印 policy 观测空间的所有 term。

    关键思路：env_cfg.observations.policy 是一个 configclass 对象（本质是 dataclass）。
    用 dataclasses.fields() 拿到所有字段的描述符，再用 getattr() 读取每个字段的实际值。
    如果值是 None，说明这个 term 被子类覆盖掉了（比如 flat env 禁用了 height_scan）。
    如果值有 .func 属性，说明它是一个 ObsTerm，打印函数名。
    """
    import dataclasses

    print("\n=== Observation Terms (policy) ===")
    policy = env_cfg.observations.policy
    for field in dataclasses.fields(policy):
        name = field.name
        value = getattr(policy, name)
        if value is None:
            print(f"  {name}: DISABLED")
        elif hasattr(value, "func"):
            # ObsTerm 对象：.func 是实际的函数引用，.__name__ 是函数名
            print(f"  {name}: {value.func.__name__}")
        # 跳过 enable_corruption、concatenate_terms 等非 term 的布尔字段


def _print_action_config(env_cfg) -> None:
    """打印 action 空间的配置。

    关键思路：遍历 env_cfg.actions 的字段，每个字段是一个 ActionTermCfg 对象。
    用 type().__name__ 拿到类名，用 hasattr + getattr 安全地读取可能存在的属性。
    """
    import dataclasses

    print("\n=== Action Config ===")
    for field in dataclasses.fields(env_cfg.actions):
        name = field.name
        value = getattr(env_cfg.actions, name)
        print(f"  [{name}]")
        print(f"    type:        {type(value).__name__}")
        print(f"    scale:       {getattr(value, 'scale', 'N/A')}")
        print(f"    joint_names: {getattr(value, 'joint_names', 'N/A')}")


def _print_reward_terms(env_cfg) -> None:
    """打印所有 reward term 和权重。

    关键思路：weight == 0.0 的 term 是"日志探头"，只监控不影响训练。
    特别标注出来，帮助快速区分真实奖励和监控项。
    """
    import dataclasses

    print("\n=== Reward Terms ===")
    for field in dataclasses.fields(env_cfg.rewards):
        name = field.name
        value = getattr(env_cfg.rewards, name)
        if value is None:
            print(f"  {name}: DISABLED")
        elif hasattr(value, "weight"):
            tag = "  [LOG-ONLY]" if value.weight == 0.0 else ""
            print(f"  {name}: weight={value.weight:+.2e}{tag}")


def _print_terminations(env_cfg) -> None:
    """打印所有终止条件。

    关键思路：time_out=True 的 term 是"超时终止"，属于正常结束。
    其他 term（如 base_contact）是"失败终止"，代表机器人摔倒或违规。
    """
    import dataclasses

    print("\n=== Termination Terms ===")
    for field in dataclasses.fields(env_cfg.terminations):
        name = field.name
        value = getattr(env_cfg.terminations, name)
        if value is None:
            print(f"  {name}: DISABLED")
        else:
            is_timeout = getattr(value, "time_out", False)
            tag = "  [timeout]" if is_timeout else "  [failure]"
            print(f"  {name}{tag}")


def _print_command_ranges(env_cfg) -> None:
    """打印速度指令的采样范围。

    关键思路：env_cfg.commands 下有一个 base_velocity 字段，
    它的 .ranges 属性存储了 lin_vel_x、lin_vel_y、ang_vel_z 等的上下界。
    """
    import dataclasses

    print("\n=== Command Ranges ===")
    for field in dataclasses.fields(env_cfg.commands):
        name = field.name
        cmd = getattr(env_cfg.commands, name)
        print(f"  [{name}]  type: {type(cmd).__name__}")
        if hasattr(cmd, "ranges"):
            for r_field in dataclasses.fields(cmd.ranges):
                r_name = r_field.name
                r_value = getattr(cmd.ranges, r_name)
                print(f"    {r_name}: {r_value}")


def _print_height_scanner(env_cfg) -> None:
    """检查 height scanner 是否启用。"""
    print("\n=== Height Scanner ===")
    enabled = env_cfg.scene.height_scanner is not None
    print(f"  enabled: {enabled}")


def _inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PULSE env config.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)

    # 延迟 import：必须在 AppLauncher 初始化之后才能导入 isaaclab 模块
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
        print(f"\n{'='*50}")
        print(f"  PULSE Env Inspector")
        print(f"  Task: {args_cli.task}")
        print(f"  Cfg:  {type(env_cfg).__name__}")
        print(f"{'='*50}")
        _print_obs_terms(env_cfg)
        _print_action_config(env_cfg)
        _print_reward_terms(env_cfg)
        _print_terminations(env_cfg)
        _print_command_ranges(env_cfg)
        _print_height_scanner(env_cfg)
        print()

    main()
    simulation_app.close()


def _run_one_condition(
    env,
    policy,
    mass_scale: float,
    com_x_offset: float,
    num_episodes: int,
) -> dict:
    """Run rollouts for a single (mass_scale, com_x_offset) condition.

    Returns a dict with aggregated metrics across num_episodes.

    TODO (Week 2): implement domain-randomization override for mass and CoM,
    run the rollout loop, collect episode_length and success_flag per episode,
    and return mean / std statistics.
    """
    # Placeholder — real logic goes here in Week 2.
    return {
        "mass_scale": mass_scale,
        "com_x_offset": com_x_offset,
        "mean_episode_length": None,
        "success_rate": None,
    }


def _save_results(results: list[dict], output_path: str) -> None:
    """Serialize eval results to a JSON file.

    TODO (Week 2): write results with json.dump, create parent dirs if needed,
    and print a confirmation line with the absolute path.
    """
    # Placeholder — real logic goes here in Week 2.
    pass


def _eval_payload_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PULSE checkpoint under varying payload conditions."
    )
    parser.add_argument("--task", type=str, default=None, help="Registered task name.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point."
    )
    parser.add_argument(
        "--mass_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Body mass scale factors to evaluate (e.g. 1.0 1.1 1.2).",
    )
    parser.add_argument(
        "--com_x_offsets",
        type=float,
        nargs="+",
        default=[0.0],
        help="CoM x-axis offsets in metres to evaluate (e.g. 0.0 0.01 -0.01).",
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per condition.")
    parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the JSON results file. Auto-generated if omitted.",
    )
    cli_args.add_rsl_rl_args(parser)
    args_cli, _app_launcher, simulation_app = _parse_and_launch_app(parser)

    # Delayed imports: must come after AppLauncher initialises the Omniverse runtime.
    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab_tasks  # noqa: F401
    import pulse.envs  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        # --- configure env ---
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        # --- load policy ---
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[EVAL] Loading checkpoint: {resume_path}")
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # --- determine output path ---
        if args_cli.output is not None:
            output_path = args_cli.output
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"results/eval_payload_{timestamp}.json"

        # --- sweep over all (mass_scale, com_x_offset) combinations ---
        results = []
        total = len(args_cli.mass_scales) * len(args_cli.com_x_offsets)
        done = 0
        for mass_scale in args_cli.mass_scales:
            for com_x_offset in args_cli.com_x_offsets:
                done += 1
                print(f"[EVAL] [{done}/{total}] mass_scale={mass_scale}  com_x_offset={com_x_offset}")
                metrics = _run_one_condition(
                    env=env,
                    policy=policy,
                    mass_scale=mass_scale,
                    com_x_offset=com_x_offset,
                    num_episodes=args_cli.num_episodes,
                )
                results.append(metrics)

        _save_results(results, output_path)
        print(f"[EVAL] Done. {len(results)} conditions evaluated.")
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
    elif mode == "eval_payload":
        _eval_payload_main()
    else:
        raise ValueError("Invalid PULSE_ENTRY_MODE. Expected 'train', 'play', 'inspect', or 'eval_payload'.")

