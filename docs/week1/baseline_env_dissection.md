# Week 1: Baseline 庖丁解牛

**环境继承链：**
`Go2PulseFlatEnvCfg` → `UnitreeGo2FlatEnvCfg` → `UnitreeGo2RoughEnvCfg` → `LocomotionVelocityRoughEnvCfg`

**权威配置文件路径：**
- 爷爷类（通用 MDP 定义）：`source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`
- 父类（Go2 机器人参数）：`.../config/go2/rough_env_cfg.py`
- 子类（平地覆盖）：`.../config/go2/flat_env_cfg.py`
- 本项目入口：`pulse/envs/go2/flat_env_cfg.py`

---

## 1. Observation Table

> **obs 向量总维度：** Flat 环境 = **48 维**（Rough 环境含 height_scan 则为 48 + 187 = **235 维**）
>
> **源码路径：** `source/isaaclab/isaaclab/envs/mdp/observations.py`
>
> **观测向量按如下顺序拼接（`concatenate_terms=True`）**

| # | Term 名称 | 源码函数 | 输出维度 | 本体感知？ | Flat 中启用？ | 备注 |
|---|-----------|----------|----------|------------|--------------|------|
| 1 | `base_lin_vel` | `observations.base_lin_vel` | `[N, 3]` | ✅ 是 | ✅ 是 | 机身坐标系（Body Frame）下的线速度 `[vx, vy, vz]`，单位 m/s；加均匀噪声 ±0.1 |
| 2 | `base_ang_vel` | `observations.base_ang_vel` | `[N, 3]` | ✅ 是 | ✅ 是 | 机身坐标系下的角速度 `[ωx, ωy, ωz]`，单位 rad/s；加均匀噪声 ±0.2 |
| 3 | `projected_gravity` | `observations.projected_gravity` | `[N, 3]` | ✅ 是 | ✅ 是 | 重力向量投影到机身坐标系：正常直立时 ≈ `[0, 0, -9.81]`；倾斜时 xy 分量非零，用于感知姿态而非 Euler 角（避免万向节死锁）；加均匀噪声 ±0.05 |
| 4 | `velocity_commands` | `observations.generated_commands` | `[N, 3]` | ✅ 是 | ✅ 是 | 当前帧系统下发的速度目标 `[vx*, vy*, ωz*]`，无噪声；范围来自 `CommandsCfg`：`vx ∈ [-1, 1]`，`vy ∈ [-1, 1]`，`ωz ∈ [-1, 1]`（rad/s）|
| 5 | `joint_pos` | `observations.joint_pos_rel` | `[N, 12]` | ✅ 是 | ✅ 是 | 12 个关节的**相对位置** = `当前角度 - 默认站立角度`，单位 rad；加均匀噪声 ±0.01 |
| 6 | `joint_vel` | `observations.joint_vel_rel` | `[N, 12]` | ✅ 是 | ✅ 是 | 12 个关节的**相对速度** = `当前速度 - 默认速度`，单位 rad/s；加均匀噪声 ±1.5 |
| 7 | `actions` | `observations.last_action` | `[N, 12]` | ✅ 是 | ✅ 是 | **上一帧**网络输出的原始（raw）动作向量（即 `env.action_manager.action`），无噪声；注意这是 scaled 之前的 raw action，不是发给电机的目标角度 |
| 8 | `height_scan` | `observations.height_scan` | `[N, 187]` | ❌ 否（外界感知） | ❌ **否（已禁用）** | 17×11 射线网格高度扫描，分辨率 0.1m，覆盖范围 1.6m×1.0m；Flat 环境中 `self.scene.height_scanner = None` 且 `self.observations.policy.height_scan = None` — **这是 Blind Locomotion 的关键** |

**关于 `last_action` 的重要说明：**
- 调用 `env.action_manager.action`，返回当前帧 policy 输出的 **raw actions**（即网络直接输出的 12 维向量，值域通常在 `[-1, 1]` 附近）。
- 这不等于发给电机的目标角度。发给电机的是 `processed_actions = raw_actions * scale + offset`。
- 为什么要把动作放进观测？让网络知道"我上一帧命令腿去哪了"，有助于输出连贯平滑的动作序列。

---

## 2. Action Table

> **源码路径：** `source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py` — `JointPositionAction`

| 字段 | 值 | 说明 |
|------|----|------|
| **Action Type** | Joint Position Target | 不是 Torque，不是 Velocity；网络输出关节**位置目标** |
| **Action 维度** | 12 维 | Go2 四条腿 × 每腿 3 个关节（髋 Hip, 大腿 Thigh, 小腿 Calf） |
| **`use_default_offset`** | `True` | 以 URDF 中的默认站立姿态角度为偏置（offset），网络输出的是在此之上的**残差** |
| **Scale** | **0.25**（Go2 专属） | 爷爷类默认 0.5；Go2 父类 `rough_env_cfg.py` 中覆盖为 `self.actions.joint_pos.scale = 0.25`；Flat 子类继承此值，未再覆盖 |
| **最终目标角度公式** | `θ_target = θ_default + action_raw × 0.25` | 即 `processed_actions = raw_actions * scale + offset` |
| **送到哪一级控制器** | PhysX 内置 PD 控制器 | 调用 `asset.set_joint_position_target(processed_actions)`，由底层 PD 控制器将关节驱动至目标角度（非直接 torque） |
| **最大偏离默认姿态** | ±0.25 rad（约 ±14.3°） | 当网络输出饱和到 ±1 时的最大关节偏转量 |
| **关节范围** | `joint_names=[".*"]`，全部 12 个关节 | 正则 `.*` 匹配所有关节 |

**动作流水线（每帧）：**
```
PPO Policy Output (dim=12, raw ∈ [-1,1])
        ↓  × scale=0.25
        ↓  + default_joint_pos (offset)
processed_actions (dim=12, 单位: rad)
        ↓
set_joint_position_target(processed_actions)
        ↓
PhysX PD Controller → 关节力矩输出
```

---

## 3. Reward Table

> **源码路径：**
> - 通用函数：`source/isaaclab/isaaclab/envs/mdp/rewards.py`
> - 运动专属函数：`source/isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py`
>
> **最终权重说明：** 以 Flat 配置为准（Go2 父类和 Flat 子类均有覆盖）。`weight=0.0` 的项在训练中**不产生梯度**，相当于未激活。

| Term 名称 | 正/负 | **最终 Weight（Flat）** | 函数公式 | 直觉含义 | 函数位置 |
|-----------|-------|------------------------|----------|----------|----------|
| `track_lin_vel_xy_exp` | ✅ 正 | **+1.5** | `exp(-‖v_cmd_xy - v_actual_xy‖² / std²)`，`std=√0.25` | 核心任务奖励：XY 速度跟踪越准分越高；指数核使得近似即可获得大部分奖励 | `isaaclab/envs/mdp/rewards.py:304` |
| `track_ang_vel_z_exp` | ✅ 正 | **+0.75** | `exp(-(ωz_cmd - ωz_actual)² / std²)`，`std=√0.25` | 航向角速度跟踪：转向越准越好 | `isaaclab/envs/mdp/rewards.py:318` |
| `feet_air_time` | ✅ 正 | **+0.25** | `Σ_feet (t_air - threshold) * first_contact`，`threshold=0.5s`；仅当 `‖v_cmd_xy‖ > 0.1` 时有效 | 鼓励足端腾空：防止狗用"滑行"替代迈步；原始定义 0.125，Flat 子类覆盖为 0.25，Rough 父类 0.01 | `velocity/mdp/rewards.py:27` |
| `lin_vel_z_l2` | ❌ 负 | **-2.0** | `-vz²` | 惩罚机身上下弹跳；鼓励稳定高度运动 | `isaaclab/envs/mdp/rewards.py:77` |
| `flat_orientation_l2` | ❌ 负 | **-2.5** | `-‖g_projected_xy‖²`，即 projected_gravity 的 xy 分量平方和 | **Flat 子类特有的强惩罚**：机身倾斜则重力在 xy 分量变大；鼓励躯干保持水平 | `isaaclab/envs/mdp/rewards.py:91` |
| `ang_vel_xy_l2` | ❌ 负 | **-0.05** | `-（ωx² + ωy²）` | 惩罚机身在俯仰（Pitch）和侧滚（Roll）方向的角速度 | `isaaclab/envs/mdp/rewards.py:84` |
| `dof_torques_l2` | ❌ 负 | **-2×10⁻⁴** | `-Σ τᵢ²`（12 个关节力矩的平方和） | 惩罚电机用力过大；鼓励节能运动；父类覆盖为 -0.0002（爷爷类默认 -1e-5） | `isaaclab/envs/mdp/rewards.py:137` |
| `dof_acc_l2` | ❌ 负 | **-2.5×10⁻⁷** | `-Σ q̈ᵢ²`（12 个关节加速度的平方和） | 惩罚动作抖动；鼓励平滑控制 | `isaaclab/envs/mdp/rewards.py:168` |
| `action_rate_l2` | ❌ 负 | **-0.01** | `-Σ (aₜ - aₜ₋₁)²` | 惩罚相邻两帧动作变化剧烈；促进时序连贯性 | `isaaclab/envs/mdp/rewards.py:252` |
| `undesired_contacts` | ❌ 负（**已禁用**） | 0.0（`=None`） | 大腿（THIGH）接触力超阈值则惩罚 | Go2 父类将其设为 `None`（`self.rewards.undesired_contacts = None`）；在此 baseline 中**不生效** | `isaaclab/envs/mdp/rewards.py:267` |
| `dof_pos_limits` | ❌ 负（**未激活**） | 0.0 | 关节超过软限制时惩罚 | weight=0.0，不影响训练，可在后续实验中激活 | `isaaclab/envs/mdp/rewards.py:189` |

**权重绝对值排序（核心驱动力一览）：**
```
欲望（正）：track_lin_vel_xy_exp(+1.5) > track_ang_vel_z_exp(+0.75) > feet_air_time(+0.25)
恐惧（负）：flat_orientation_l2(-2.5) > lin_vel_z_l2(-2.0) > dof_torques_l2(-2e-4) > ang_vel_xy_l2(-0.05) > action_rate_l2(-0.01)
```

---

## 4. Termination Table

> **源码路径：** `source/isaaclab/isaaclab/envs/mdp/terminations.py`
>
> **当前 Baseline 中只有 2 个 active termination term**，不存在关节超限终止或其他死法。

| Term 名称 | 类型 | 触发条件 | `time_out` 标记 | 对 PPO 的含义 | 函数位置 |
|-----------|------|----------|-----------------|--------------|----------|
| `time_out` | 超时（Truncation） | `episode_length_buf >= max_episode_length`；默认 episode 上限 **20 秒**（= 4000 步，步长 0.005s，decimation=4） | **`True`** | 标记为 `truncated`，PPO 在 GAE 计算时会 bootstrap（不算真正失败）；不惩罚 | `terminations.py:31` |
| `base_contact` | 摔倒（Termination） | 机身（`body_names="base"`）所受合力任意历史帧超过 **1.0 N** | `False` | 标记为 `terminated`（真正死亡），PPO 将该轨迹末尾 value 置零；隐性惩罚 | `terminations.py:154` |

**明确不存在的终止条件（与其他论文实现对比）：**

| 条件 | 是否存在 | 说明 |
|------|----------|------|
| 关节超出位置限制 (`joint_pos_limits`) | ❌ 否 | 只有 `weight=0.0` 的 Reward 项，不是 Termination |
| 电机力矩超限 | ❌ 否 | 无此 Termination |
| 大腿/脚以外身体部位触地 | ❌ 否 | `undesired_contacts` 已被 Go2 父类设为 `None` |
| 高度过低（机身贴地） | ❌ 否 | 无此 Termination |

> **后续实验提示：** 如需增加关节超限终止，可在 `Go2PulseFlatEnvCfg.__post_init__` 中添加：
> ```python
> from isaaclab.managers import TerminationTermCfg as DoneTerm
> import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
> self.terminations.joint_pos_limits = DoneTerm(func=mdp.joint_pos_out_of_limit)
> ```

---

*文档生成时间：Week 1 | 验证方式：逐行追溯源码，所有数值均经代码确认*
