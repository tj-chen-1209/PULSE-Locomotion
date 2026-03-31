# Training Log

每次实验后填写，作为研究 trail。不要相信记忆，用文字留存每一个决策。

格式：

```
## YYYY-MM-DD — <一句话描述本次改动>

**改了什么**
- ...

**为什么改**
- ...

**结果**
- ...

**下一步**
- ...
```

---

<!-- 从这里开始按时间倒序追加 -->

## 2026-04-01 — 发现 heading_command OOD，修正命令分布，重新训练

**改了什么**
- `Go2PulseFlatEnvCfg.__post_init__` 新增三行命令分布修正：
  - `heading_command = False`（直接采样 yaw rate，不再用 heading 误差换算）
  - `rel_standing_envs = 0.20`（20% 环境训练原地低速/转圈，原来是 0.02）
  - `rel_heading_envs = 0.0`（配套关闭 heading 模式）
- `Go2PULSEFlatPPORunnerCfg` 中 `max_iterations` 从 300 提高至 1500
- `_play_main` 新增 `--fixed_vx / --fixed_vy / --fixed_yaw` 三个 CLI 参数，支持固定命令 rollout

**为什么改**
- 原地转圈 eval 效果极差（摔倒率高、步态混乱）
- 根因分析：两个叠加的 OOD
  1. 训练用 `heading_command=True`，yaw 观测值是"heading 误差 × 0.5"的衰减量，eval 给固定 yaw rate，形状从未在训练中出现
  2. `rel_standing_envs=0.02` 导致原地旋转样本极少，策略几乎没有练过"vx=0, vy=0, yaw≠0"

**结果**
- play 脚本跑出 `unrecognized arguments: --fixed_heading` 错误（不影响训练，是早期测试时命令拼写问题）
- 重新训练已启动（`Go2-Pulse-Flat-v0`，headless，目标 1500 iter）
- 训练使用 RTX 4060 Ti (8 GB)，12th Gen i5-12600KF，32 GB RAM

**下一步**
- 等待 1500 iter 训练完成，观察 TensorBoard `Episode/mean_reward` 是否收敛
- 用最新 checkpoint 跑三段固定命令 rollout：
  - `vx=0.8, vy=0.0, yaw=0.0`（直行）
  - `vx=0.0, vy=0.0, yaw=0.8`（原地转）
  - `vx=0.5, vy=0.3, yaw=0.0`（斜行）
- 三组 rollout 通过后再启动 `eval_payload.py` 扫 payload 网格

---

## 2026-03-31 — 初始 baseline 训练，300 iter，发现 OOD 问题

**改了什么**
- 继承 `UnitreeGo2FlatEnvCfg` 搭建 `Go2PulseFlatEnvCfg`
- 禁用 height scanner 和 terrain curriculum（blind flat locomotion 设置）
- 添加 4 个 weight=0.0 的 reward 探头（base_contact / action_rate / joint_acc / joint_limits）
- 运行两次训练，各 300 iter（checkpoint: `2026-03-30_23-06-03` 和 `2026-03-31_11-22-54`）

**为什么改**
- 复刻 IsaacLab 官方 Go2 flat baseline，作为 payload 鲁棒性实验的 nominal baseline

**结果**
- 训练完成，有可加载的 `model_299.pt`
- 直行（vx=0.8）基本可行
- 原地转圈（vx=0, yaw=0.8）效果很差 → 触发 OOD 分析

**下一步**
- 见 2026-04-01 条目
