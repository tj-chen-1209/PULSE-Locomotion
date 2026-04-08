# Week 1 Baseline: Audit Version

## Scope

本页只做 Week 1 baseline 的可审计硬字段，不重写大段理论。默认对象是 `Go2PulseFlatEnvCfg` / `Go2-Pulse-Flat-v0`。

环境继承链：

`LocomotionVelocityRoughEnvCfg -> UnitreeGo2RoughEnvCfg -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg`

审计入口：

- `python scripts/inspect_env.py --task Go2-Pulse-Flat-v0 --output results/audit/go2_pulse_flat`
- `python scripts/eval_fixed_commands.py --task Go2-Pulse-Flat-Play-v0 --checkpoint logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt --output results/eval_fixed_commands/go2_pulse_flat`
- `python scripts/eval_payload.py --task Go2-Pulse-Flat-Play-v0 --checkpoint logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt --output results/eval_payload/go2_pulse_flat`

Week 1 正式冻结 baseline checkpoint：

- `logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt`
- 本页所有 fixed-command、payload、boundary sweep、video 证据都统一指向这个 checkpoint

---

## 1. Observation Audit

Flat policy obs 总维度：`48`

Rough 对照：若启用 `height_scan`，总维度会变为 `48 + 187 = 235`

| Term | Dims | Flat 启用 | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---:|---|---|---|---|
| `base_lin_vel` | 3 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | `inspect_env.py` 运行态 `observations[].dims` + `func` | 机身坐标系线速度 |
| `base_ang_vel` | 3 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 机身坐标系角速度 |
| `projected_gravity` | 3 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 用 projected gravity 感知姿态，不直接给 Euler 角 |
| `velocity_commands` | 3 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 当前速度命令 `[vx*, vy*, yaw*]` |
| `joint_pos` | 12 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 12 个关节相对默认站立位的偏差 |
| `joint_vel` | 12 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 12 个关节速度 |
| `actions` | 12 | Yes | `LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg` | 同上 | 只包含 last action，不是多步 action history |
| `height_scan` | 187 | No | `LocomotionVelocityRoughEnvCfg -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 中 `height_scanner.scene_enabled=false` 且 `policy_obs_enabled=false` | 当前 baseline 是 blind locomotion，没有外界地形高度感知 |

结论：

- `obs names + dims` 必须以 `scripts/inspect_env.py` 导出的 JSON/TXT 为准。
- 当前 flat baseline 明确包含 `velocity_commands` 和 `last_action`。
- 当前 flat baseline 明确不包含 `height_scan`。

---

## 2. Action Audit

| 字段 | 最终值 | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---|---|---|
| Action term | `joint_pos` | `LocomotionVelocityRoughEnvCfg.ActionsCfg` | `inspect_env.py` 的 `actions[].name` / `term_type` | 单一 action term |
| Action type | `JointPositionAction` | `isaaclab.envs.mdp.actions.joint_actions.JointPositionAction` | `inspect_env.py` 的 `actions[].term_type` | 不是 torque，不是 velocity |
| Action dim | `12` | Go2 articulation joints | `inspect_env.py` 的 `actions[].dim` | 四条腿，每腿三个关节 |
| Scale | `0.25` | `LocomotionVelocityRoughEnvCfg=0.5 -> UnitreeGo2RoughEnvCfg=0.25 -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 的 `actions[].key_fields.scale` | Go2 父类覆盖后，PULSE 没再改 |
| `use_default_offset` | `True` | `LocomotionVelocityRoughEnvCfg.ActionsCfg` | `inspect_env.py` 的 `actions[].key_fields.use_default_offset` | 以默认站立位为 offset |
| `joint_names` | `[".*"]` | `LocomotionVelocityRoughEnvCfg.ActionsCfg` | `inspect_env.py` 的 `actions[].key_fields.joint_names` | 匹配全部 12 个关节 |
| 目标公式 | `q_target = q_default + action_raw * 0.25` | Action cfg + `JointPositionAction` 实现 | 代码确认 + `inspect_env.py` 导出的 scale / offset 配置 | raw action 是残差，发给 PhysX 的是 position target |

结论：

- 当前 baseline 的动作空间是 `12D joint position target residual`。
- `inspect_env.py` 现在会把 action cfg 关键字段一并导出，不再只是打印类名。

---

## 3. Command Audit

PULSE 对 Go2 flat baseline 做了 command 分布修正，必须单独写明。

| 字段 | 最终值 | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---|---|---|
| `heading_command` | `False` | `LocomotionVelocityRoughEnvCfg=True -> Go2PulseFlatEnvCfg=False` | `inspect_env.py` 的 `commands[].heading_command` | yaw 命令直接作为 `ang_vel_z`，不再用 heading 控制衰减到 0 |
| `rel_standing_envs` | `0.20` | `LocomotionVelocityRoughEnvCfg=0.02 -> Go2PulseFlatEnvCfg=0.20` | `inspect_env.py` 的 `commands[].rel_standing_envs` | 明确提升低速/原地任务覆盖率 |
| `rel_heading_envs` | `0.0` | `LocomotionVelocityRoughEnvCfg=1.0 -> Go2PulseFlatEnvCfg=0.0` | `inspect_env.py` 的 `commands[].rel_heading_envs` | 与 `heading_command=False` 一起生效 |
| `lin_vel_x` range | `[-1.0, 1.0]` | `LocomotionVelocityRoughEnvCfg.CommandsCfg` | `inspect_env.py` 的 `commands[].ranges.lin_vel_x` | 训练采样范围 |
| `lin_vel_y` range | `[-1.0, 1.0]` | 同上 | 同上 | 训练采样范围 |
| `ang_vel_z` range | `[-1.0, 1.0]` | 同上 | 同上 | 训练采样范围 |
| `resampling_time_range` | `[10.0, 10.0]` | `LocomotionVelocityRoughEnvCfg.CommandsCfg` | `inspect_env.py` 的 `commands[].resampling_time_range` | 10 秒固定重采样周期 |

结论：

- 当前 PULSE baseline 已经显式关闭 `heading_command`。
- 这意味着后续 fixed-command 评估与训练命令语义一致，不再是 OOD 的 pure yaw 设定。

---

## 4. Reward Audit

`weight=0.0` 的项在这里被当作日志探针，不参与梯度更新，但需要作为审计字段保留。

| Term | Sign | 最终 weight | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---:|---|---|---|
| `track_lin_vel_xy_exp` | + | `1.5` | `LocomotionVelocityRoughEnvCfg=1.0 -> UnitreeGo2RoughEnvCfg=1.5 -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 的 `rewards[].weight` | 主任务：平面线速度跟踪 |
| `track_ang_vel_z_exp` | + | `0.75` | `LocomotionVelocityRoughEnvCfg=0.5 -> UnitreeGo2RoughEnvCfg=0.75 -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | 同上 | 航向角速度跟踪 |
| `feet_air_time` | + | `0.25` | `LocomotionVelocityRoughEnvCfg=0.125 -> UnitreeGo2RoughEnvCfg=0.01 -> UnitreeGo2FlatEnvCfg=0.25 -> Go2PulseFlatEnvCfg` | 同上 | 鼓励迈步，不鼓励纯滑行 |
| `lin_vel_z_l2` | - | `-2.0` | `LocomotionVelocityRoughEnvCfg` | 同上 | 抑制上下弹跳 |
| `flat_orientation_l2` | - | `-2.5` | `LocomotionVelocityRoughEnvCfg=0.0 -> UnitreeGo2FlatEnvCfg=-2.5 -> Go2PulseFlatEnvCfg` | 同上 | 保持 torso 水平 |
| `ang_vel_xy_l2` | - | `-0.05` | `LocomotionVelocityRoughEnvCfg` | 同上 | 抑制 roll/pitch 角速度 |
| `dof_torques_l2` | - | `-2e-4` | `LocomotionVelocityRoughEnvCfg=-1e-5 -> UnitreeGo2RoughEnvCfg=-2e-4 -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | 同上 | 抑制电机用力过大 |
| `dof_acc_l2` | - | `-2.5e-7` | `LocomotionVelocityRoughEnvCfg -> UnitreeGo2RoughEnvCfg -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | 同上 | 抑制关节加速度过大 |
| `action_rate_l2` | - | `-0.01` | `LocomotionVelocityRoughEnvCfg` | 同上 | 抑制动作抖动 |
| `undesired_contacts` | disabled | `None` | `LocomotionVelocityRoughEnvCfg -> UnitreeGo2RoughEnvCfg=None -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 中该 term 不出现在 active rewards | 当前 baseline 不对 thigh contact 施加惩罚 |
| `dof_pos_limits` | log-only | `0.0` | `LocomotionVelocityRoughEnvCfg -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 的 `log_only=true` | 只记录 joint limit 压力，不塑形 |
| `log_base_contact` | log-only | `0.0` | `Go2PulseFlatEnvCfg` | 同上 | 只记录 base contact |
| `log_action_rate` | log-only | `0.0` | `Go2PulseFlatEnvCfg` | 同上 | 只记录动作变化率 |
| `log_joint_acc` | log-only | `0.0` | `Go2PulseFlatEnvCfg` | 同上 | 只记录 joint acc |
| `log_joint_limits` | log-only | `0.0` | `Go2PulseFlatEnvCfg` | 同上 | 只记录 joint limit 压力 |

### Reward-Induced Bias Hypothesis

当前 baseline 的 reward 主导项仍然是：

`track_lin_vel_xy_exp + flat_orientation_l2 + lin_vel_z_l2 + track_ang_vel_z_exp`

Week 1 的明确假设：

1. policy 可以通过“站稳、躯干水平、前向线速度干净”拿到一条看上去不错的 reward 曲线；
2. 但这不等价于它已经学会了 `pure yaw` 和 `diagonal` 两类固定命令；
3. 因为 `track_ang_vel_z_exp` 的权重仍弱于“别抖、别歪、别弹”的存活型约束，reward 可能系统性高估真实 command coverage；
4. 所以 Week 1 不能再只看 reward curve，必须用固定命令 rollout 指标来证伪或证实这个假设。

这就是本周加入 `eval_fixed_commands.py` 的原因。

---

## 5. Termination Audit

### Active terminations

| Term | `time_out` | Failure term | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---|---|---|---|
| `time_out` | `True` | No | `LocomotionVelocityRoughEnvCfg.TerminationsCfg` | `inspect_env.py` 的 `terminations[].time_out=true` | 正常 episode 截断，20 秒 horizon |
| `base_contact` | `False` | Yes | `LocomotionVelocityRoughEnvCfg -> UnitreeGo2RoughEnvCfg(body_names=base) -> UnitreeGo2FlatEnvCfg -> Go2PulseFlatEnvCfg` | `inspect_env.py` 的 `terminations[].failure=true` | 机身触地即失败 |

### Explicit non-active pre-cut terms

| 条件 | 当前是否 active termination | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---|---|---|
| joint limit | No | `dof_pos_limits` 只在 rewards/log probe | `inspect_env.py` 中不存在对应 termination term | 当前失败不是被 joint limit 预裁掉 |
| torque limit | No | 未定义对应 termination term | `inspect_env.py` 中不存在对应 termination term | 当前失败不是被 torque threshold 预裁掉 |
| posture threshold | No | 只有 `flat_orientation_l2` reward，没有 posture termination | `inspect_env.py` 中不存在对应 termination term | 当前失败不是被姿态阈值预裁掉 |

必须明写：

- active termination 只有 `time_out` 和 `base_contact`
- 当前失败不是被 `joint limit / torque / 姿态阈值` 提前裁掉
- 因此如果 rollout 提前结束，Week 1 先看 `base_contact`，不是先怀疑 hidden pre-cut

---

## 6. Baseline Success Criteria (Week 1)

固定命令三组，一个都不能少：

- `vx=0.8, vy=0.0, yaw=0.0`
- `vx=0.4, vy=0.0, yaw=0.8`
- `vx=0.5, vy=0.3, yaw=0.0`

指标定义已经在 `scripts/eval_fixed_commands.py` 里写死，结果以 JSON/Markdown 导出，不再只看 reward 曲线。

| Command | Pass 条件 | 来源层级 | 最终生效值确认方式 | 说明 |
|---|---|---|---|---|
| `forward` | `mean_tracking_error <= 0.25`, `mean_abs_roll_deg <= 6`, `mean_abs_pitch_deg <= 6`, `fall_rate <= 0.05`, `survival_time_s >= 18.0` | Week 1 audit definition in this doc + `scripts/_entry.py::WEEK1_SUCCESS_CRITERIA` | `results/eval_fixed_commands/*.json` 的 `per_command[].pass` | 前向跟踪要求最严 |
| `turn` | `mean_tracking_error <= 0.35`, `mean_abs_roll_deg <= 8`, `mean_abs_pitch_deg <= 8`, `fall_rate <= 0.10`, `survival_time_s >= 16.0` | 同上 | 同上 | 圆周转向允许略宽松，不再要求原地自转 |
| `diagonal` | `mean_tracking_error <= 0.35`, `mean_abs_roll_deg <= 8`, `mean_abs_pitch_deg <= 8`, `fall_rate <= 0.10`, `survival_time_s >= 16.0` | 同上 | 同上 | 对 command coupling 做最低证明 |

24 小时交付物：

- 一张固定命令表，至少包含：
  - `mean_tracking_error`
  - `mean_abs_roll_deg`
  - `mean_abs_pitch_deg`
  - `fall_rate`
  - `survival_time_s`
  - `pass`

### 6.1 Actual Week 1 Result

当前正式 protocol 已写死为 circular-turn：

- `turn = vx=0.4, vy=0.0, yaw=0.8`
- 冻结 baseline checkpoint = `logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt`
- 本节结果来自 2026-04-07 重新运行后的干净输出
- 旧版 pure-yaw turn (`vx=0.0, vy=0.0, yaw=0.8`) 只保留为历史诊断，不再作为当前 Week 1 验收标准

实际结果文件：

- `results/eval_fixed_commands/go2_pulse_flat.json`
- `results/eval_fixed_commands/go2_pulse_flat.md`

| Command | mean_tracking_error | mean_abs_roll_deg | mean_abs_pitch_deg | fall_rate | survival_time_s | pass | 来源层级 | 最终生效值确认方式 | 解释 |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| `forward` | `0.140` | `1.217` | `1.460` | `0.000` | `20.000` | `PASS` | `results/eval_fixed_commands/go2_pulse_flat.json::per_command[forward]` | 运行脚本直接导出的聚合指标 | 前向 tracking 稳定 |
| `turn` | `0.125` | `1.179` | `1.604` | `0.000` | `20.000` | `PASS` | 同上 | 同上 | circular-turn command 已通过，不再要求原地自转 |
| `diagonal` | `0.146` | `2.759` | `1.989` | `0.000` | `20.000` | `PASS` | 同上 | 同上 | 对角耦合运动通过，但姿态裕度最紧 |

结论：

- 三组固定命令全部通过，Week 1 `overall_pass = PASS`
- 所有 episode 都以 `time_out` 结束，`fall_rate=0.0`
- 当前 baseline 已能稳定完成前向、圆周转向和对角运动三类 nominal locomotion 行为
- 从结果看，当前 ckpt 的风险不在“是否会站住”，而在更强扰动下姿态裕度还能保留多少

---

## 7. Payload Robustness Grid (48h)

本节不再使用 A/B smoke test，正式收口为 `3 x 3` payload grid：

- `mass_scale = 1.0 / 1.1 / 1.2`
- `com_x = -0.02 / 0.00 / +0.02`
- `turn` 固定为 circular-turn：`vx=0.4, vy=0.0, yaw=0.8`
- 每个格子保留 `per_command` summary，不只保留总平均
- 当前 frozen checkpoint 仍为 `logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt`

实际结果文件：

- `results/eval_payload/go2_pulse_flat.json`
- `results/eval_payload/go2_pulse_flat.md`
- `results/eval_payload/go2_pulse_flat_mean_abs_pitch_deg_heatmap.png`
- `results/eval_payload/go2_pulse_flat_mean_tracking_error_heatmap.png`
- `results/eval_payload/go2_pulse_flat_nominal_vs_disturbed_per_command.png`

### 7.1 3x3 Aggregate Grid

| mass_scale | com_x | mean_abs_pitch_deg | mean_tracking_error | success_rate | fall_rate | 解释 |
|---|---:|---:|---:|---:|---:|---|
| `1.0` | `-0.02` | `1.086` | `0.139` | `1.000` | `0.000` | backward COM 最稳，pitch 最小 |
| `1.0` | `0.00` | `1.684` | `0.137` | `1.000` | `0.000` | nominal reference |
| `1.0` | `+0.02` | `2.424` | `0.139` | `1.000` | `0.000` | 仅 forward COM 就能显著抬高 pitch |
| `1.1` | `-0.02` | `1.165` | `0.143` | `1.000` | `0.000` | mass 上去后仍主要受 COM 方向影响 |
| `1.1` | `0.00` | `1.856` | `0.140` | `1.000` | `0.000` | tracking 仍基本不动 |
| `1.1` | `+0.02` | `2.662` | `0.140` | `1.000` | `0.000` | forward COM 继续主导退化 |
| `1.2` | `-0.02` | `1.198` | `0.144` | `1.000` | `0.000` | backward COM 仍有明显缓冲 |
| `1.2` | `0.00` | `2.049` | `0.143` | `1.000` | `0.000` | mass 变重会恶化姿态，但弱于 forward COM |
| `1.2` | `+0.02` | `2.793` | `0.139` | `1.000` | `0.000` | 全 9 格最差，作为 disturbed reference |

结论：

- 3x3 grid 清楚表明最敏感轴是 `forward com_x`，`mass` 次之
- 退化主导项是 `pitch/posture`，不是 `fall_rate`
- tracking 只发生轻微变化，`0.137 ~ 0.144` 范围内波动
- 全 9 格均 `success_rate=1.0`、`fall_rate=0.0`、`survival_time_s=20.0`

### 7.2 Convincing Figures

关键图不只做表，直接把证据挂在文档里：

- Pitch heatmap: `../../results/eval_payload/go2_pulse_flat_mean_abs_pitch_deg_heatmap.png`
- Tracking heatmap: `../../results/eval_payload/go2_pulse_flat_mean_tracking_error_heatmap.png`
- Nominal vs disturbed per-command bar: `../../results/eval_payload/go2_pulse_flat_nominal_vs_disturbed_per_command.png`

![3x3 mean_abs_pitch_deg heatmap](../../results/eval_payload/go2_pulse_flat_mean_abs_pitch_deg_heatmap.png)

![3x3 mean_tracking_error heatmap](../../results/eval_payload/go2_pulse_flat_mean_tracking_error_heatmap.png)

![nominal vs disturbed per-command bar](../../results/eval_payload/go2_pulse_flat_nominal_vs_disturbed_per_command.png)

图上的叙事很明确：

- `mean_abs_pitch_deg` 从 nominal `1.684` 升到 disturbed `2.793`，是当前最有解释力的退化信号
- `mean_tracking_error` 只从 `0.137` 升到 `0.139`，说明 tracking 不是第一失效轴
- per-command 图说明最敏感命令是 `turn` 和 `diagonal`
- nominal -> disturbed 的 `mean_abs_pitch_deg` 增量分别为：
  - `forward: 1.458 -> 2.260`
  - `turn: 1.604 -> 3.113`
  - `diagonal: 1.991 -> 3.005`
- nominal -> disturbed 的 `mean_tracking_error` 仍较小：
  - `forward: 0.140 -> 0.134`
  - `turn: 0.125 -> 0.132`
  - `diagonal: 0.146 -> 0.151`

因此这个 baseline 当前讲得清楚的是：

- degradation 是真的
- 但 degradation 主要是 `posture/pitch` 主导，不是 tracking collapse 主导
- 也不是 fall-rate 主导

### 7.3 Payload Audit

`eval_payload.py` 现在额外导出 nominal/applied mass 与 COM sanity record，用来回答“payload 真的打进去了吗”。

| Scenario | nominal_base_mass | applied_base_mass | nominal_base_com_x | applied_base_com_x | applied_mass_scale_vs_nominal | applied_com_x_delta | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| `nominal` | `6.9210` | `6.9210` | `0.021112` | `0.021112` | `1.000` | `0.000` | nominal 场景不改 base mass / COM |
| `disturbed` | `6.9210` | `8.3052` | `0.021112` | `0.041112` | `1.200` | `0.020` | payload 确实生效，且正向前移 `+0.02m` |

对应来源：

- `results/eval_payload/go2_pulse_flat.json::scenarios[].payload_audit`
- `results/eval_payload/go2_pulse_flat.md` 中的 `Payload Audit` 表

### 7.4 Control-Effort And Safety Metrics

仅靠 `fall_rate` 和 `tracking_error` 已不够区分后续方法优劣，所以 payload 评测里补了两类指标：

- 控制代价类：`mean_action_rate_l2`、`mean_torque_l2`、`mean_abs_power`
- 姿态安全类：`pitch_rms_deg`、`peak_abs_pitch_deg`

nominal vs disturbed 的 aggregate 对比：

| Scenario | pitch_rms_deg | peak_abs_pitch_deg | mean_action_rate_l2 | mean_torque_l2 | mean_abs_power |
|---|---:|---:|---:|---:|---:|
| `nominal` | `1.765` | `2.767` | `3.483` | `415.225` | `91.297` |
| `disturbed` | `2.860` | `4.510` | `3.868` | `493.627` | `99.805` |

这说明：

- `pitch RMS / peak` 比 `fall_rate` 更早暴露安全裕度收缩
- `torque / power / action-rate` 比 `tracking_error` 更能体现控制负担增加
- 这组 baseline 后续如果要做 payload-aware method，对比重点应放在这些量，而不是只看是否跌倒

---

## 8. Boundary Sweep

3x3 grid 已经告诉我们最敏感轴是 `forward com_x`，因此下一步不再乱扫，而是沿敏感方向打穿边界。

实际结果文件：

- `results/eval_payload_boundary/go2_pulse_flat_com_x_boundary.json`
- `results/eval_payload_boundary/go2_pulse_flat_com_x_boundary.md`
- `results/eval_payload_boundary/go2_pulse_flat_mass_boundary.json`
- `results/eval_payload_boundary/go2_pulse_flat_mass_boundary.md`
- `results/eval_payload_boundary/go2_pulse_flat_com_x_boundary_extended.json`
- `results/eval_payload_boundary/go2_pulse_flat_com_x_boundary_extended.md`

### 8.1 Fixed mass = 1.2, sweep com_x

第一刀按原计划扫 `com_x = 0.02 / 0.03 / 0.04`：

| mass_scale | com_x | mean_abs_pitch_deg | peak_abs_pitch_deg | mean_tracking_error | mean_action_rate_l2 | mean_torque_l2 | mean_abs_power | success_rate | fall_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.2` | `0.02` | `2.794` | `4.527` | `0.139` | `3.870` | `493.661` | `99.910` | `1.000` | `0.000` |
| `1.2` | `0.03` | `3.130` | `5.170` | `0.137` | `3.872` | `491.257` | `99.651` | `1.000` | `0.000` |
| `1.2` | `0.04` | `3.423` | `5.479` | `0.142` | `3.967` | `489.650` | `98.514` | `1.000` | `0.000` |

为了继续找 failure onset，又向前扩到 `com_x = 0.04 / 0.05 / 0.06`：

| mass_scale | com_x | mean_abs_pitch_deg | peak_abs_pitch_deg | mean_tracking_error | mean_action_rate_l2 | mean_torque_l2 | mean_abs_power | success_rate | fall_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.2` | `0.04` | `3.422` | `5.478` | `0.143` | `3.969` | `489.583` | `98.358` | `1.000` | `0.000` |
| `1.2` | `0.05` | `3.756` | `5.777` | `0.145` | `3.999` | `488.324` | `97.128` | `1.000` | `0.000` |
| `1.2` | `0.06` | `4.070` | `6.280` | `0.143` | `4.026` | `486.884` | `97.706` | `1.000` | `0.000` |

解释：

- `forward com_x` 方向的退化非常稳定且几乎单调
- `mean_abs_pitch_deg` 从 `2.794 -> 4.070`
- `peak_abs_pitch_deg` 从 `4.527 -> 6.280`
- tracking 基本没有塌，仍在 `0.137 ~ 0.145`
- 到 `com_x=+0.06` 仍未出现 failure onset

### 8.2 Fixed com_x = +0.02, sweep mass

第二刀扫 `mass = 1.2 / 1.3 / 1.4`：

| mass_scale | com_x | mean_abs_pitch_deg | peak_abs_pitch_deg | mean_tracking_error | mean_action_rate_l2 | mean_torque_l2 | mean_abs_power | success_rate | fall_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.2` | `0.02` | `2.794` | `4.512` | `0.139` | `3.869` | `493.657` | `99.855` | `1.000` | `0.000` |
| `1.3` | `0.02` | `2.813` | `4.953` | `0.134` | `4.125` | `532.624` | `108.015` | `1.000` | `0.000` |
| `1.4` | `0.02` | `2.590` | `5.363` | `0.144` | `4.219` | `554.735` | `110.033` | `1.000` | `0.000` |

解释：

- `mass` 轴也会恶化性能，但表现形式更像 control effort 上升，而不是 mean pitch 单调上升
- `mean_action_rate_l2` 从 `3.869 -> 4.219`
- `mean_torque_l2` 从 `493.657 -> 554.735`
- `mean_abs_power` 从 `99.855 -> 110.033`
- 因此 `mass` 是次敏感轴，且主要通过控制代价暴露

### 8.3 Boundary Conclusion

当前已能把边界分成三段：

- `mild degradation` 区：`mass=1.2, com_x=+0.02`
- `clear degradation` 区：`mass=1.2, com_x>=+0.03`，以及 `com_x=+0.02, mass>=1.3`
- `failure onset` 区：本轮尚未打到；至少在 `mass=1.2, com_x=+0.06` 和 `mass=1.4, com_x=+0.02` 以内都还没有出现

因此本轮 boundary sweep 的硬结论不是“已经找到失败点”，而是：

- 已确认 `forward com_x` 是第一敏感轴
- 已确认 `mass` 是第二敏感轴
- 已确认 degradation 先表现为 `pitch safety margin` 和 `control effort` 恶化
- 尚未进入真正的 collapse / fall onset 区

---

## 9. Rollout Video Evidence

三段 fixed-command 视频已按 frozen baseline checkpoint 录制并落盘：

- `results/videos/go2_pulse_flat_forward.mp4`
- `results/videos/go2_pulse_flat_turn.mp4`
- `results/videos/go2_pulse_flat_diagonal.mp4`

对应命令分别是：

- `forward = vx=0.8, vy=0.0, yaw=0.0`
- `turn = vx=0.4, vy=0.0, yaw=0.8`
- `diagonal = vx=0.5, vy=0.3, yaw=0.0`

录制方式统一为：

```bash
conda activate env_isaaclab
cd /home/tingjia/Project/PULSE-Locomotion

python scripts/play.py \
  --task Go2-Pulse-Flat-Play-v0 \
  --checkpoint logs/rsl_rl/go2_pulse_flat/2026-03-31_11-22-54/model_299.pt \
  --video \
  --video_length 400 \
  --fixed_vx <vx> \
  --fixed_vy <vy> \
  --fixed_yaw <yaw>
```

这三段视频与：

- `results/eval_fixed_commands/go2_pulse_flat.json`
- `results/eval_fixed_commands/go2_pulse_flat.md`

共同形成 Week 1 nominal baseline 的完整证据链。

---

## 10. Minimal Audit Checklist

- `inspect_env.py` 产出 obs names + dims
- `inspect_env.py` 产出 action cfg 关键字段
- `inspect_env.py` 产出 reward terms + weights
- `inspect_env.py` 产出 termination terms + `time_out/failure`
- `inspect_env.py` 产出 command ranges
- `inspect_env.py` 明写 height scanner 是否启用
- `eval_fixed_commands.py` 产出三组固定命令结果表
- `eval_payload.py` 产出 `3 x 3` payload grid
- `eval_payload.py` 保留每个 payload 格子的 `per_command` summary
- `eval_payload.py` 导出 heatmap、per-command bar、payload audit
- `eval_payload_boundary/*.json|md` 产出 boundary sweep 结果
- `results/videos/*.mp4` 提供 nominal rollout 视频证据

如果这些文件已经落到 `results/`，Week 1 baseline 才算进入“可审计版”。
