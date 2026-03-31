"""Evaluate a trained PULSE policy under varying payload (mass + CoM offset) conditions.

Payload Evaluation Logic
========================

CLI 入口
  python scripts/eval_payload.py
    --task         Go2-Pulse-Flat-v0
    --checkpoint   logs/.../model_299.pt
    --mass_scales  1.0  1.1  1.2
    --com_x_offsets 0.0  0.01  -0.01
    --num_episodes 20
    --num_envs     50
    --output       results/eval_payload_TIMESTAMP.json
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  启动层  (eval_payload.py)                           │
  │  run_entry() → isaaclab.sh → _entry.py              │
  └───────────────────────┬─────────────────────────────┘
                          │ mode="eval_payload"
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  初始化层  (_eval_payload_main)                      │
  │  1. AppLauncher 启动 Omniverse 运行时                 │
  │  2. gym.make(task, cfg=env_cfg)  创建仿真环境         │
  │  3. runner.load(checkpoint)      加载权重             │
  │  4. policy = get_inference_policy()                  │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  扫描层  双重 for 循环                                │
  │                                                      │
  │  for mass_scale in [1.0, 1.1, 1.2]:                 │
  │    for com_x_offset in [0.0, 0.01, -0.01]:          │
  │                                                      │
  │      ┌─────────────────────────────────────┐         │
  │      │  _run_one_condition()  [TODO Week2]  │         │
  │      │                                      │         │
  │      │  1. 覆盖物理参数                      │         │
  │      │     robot.base.mass *= mass_scale    │         │
  │      │     robot.base.com_x += com_x_offset │         │
  │      │                                      │         │
  │      │  2. rollout 循环 (num_episodes 次)    │         │
  │      │     obs = env.reset()                │         │
  │      │     while not done:                  │         │
  │      │       action = policy(obs)           │         │
  │      │       obs, rew, done = env.step()    │         │
  │      │                                      │         │
  │      │  3. 收集每个 episode 的指标            │         │
  │      │     - episode_length                 │         │
  │      │     - success (未摔倒)                │         │
  │      │     - mean_vel_tracking_error        │         │
  │      │                                      │         │
  │      │  return {mean, std} per metric       │         │
  │      └─────────────┬───────────────────────┘         │
  │                    │                                  │
  │      results.append(metrics)                         │
  │                                                      │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  输出层  _save_results()  [TODO Week2]               │
  │                                                      │
  │  results (list of dicts)                             │
  │  → JSON 文件  results/eval_payload_TIMESTAMP.json    │
  │                                                      │
  │  每条记录格式：                                        │
  │  {                                                   │
  │    "mass_scale": 1.1,                                │
  │    "com_x_offset": 0.01,                             │
  │    "mean_episode_length": 342.5,                     │
  │    "success_rate": 0.85,                             │
  │  }                                                   │
  │                                                      │
  │  后续可用 pandas + seaborn 画热力图                    │
  └─────────────────────────────────────────────────────┘

扫描网格示意 (3×3 = 9 个条件)：

              com_x_offset
           -0.01   0.00   +0.01
          ┌──────┬──────┬──────┐
mass 1.0  │  #1  │  #2  │  #3  │
     1.1  │  #4  │  #5  │  #6  │  每格跑 num_episodes 个 episode
     1.2  │  #7  │  #8  │  #9  │
          └──────┴──────┴──────┘
               ↑
               nominal baseline (#2, mass=1.0, com=0.0)
               这格必须先稳定，才能评估其他格的鲁棒性下降
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pulse.runtime.launcher import run_entry


def main() -> int:
    entry_script = Path(__file__).resolve().parent / "_entry.py"
    return run_entry(entry_script, sys.argv[1:], mode="eval_payload")


if __name__ == "__main__":
    raise SystemExit(main())
