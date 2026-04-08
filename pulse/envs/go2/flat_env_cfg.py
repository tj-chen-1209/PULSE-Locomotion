from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY
)


@configclass
class Go2PulseFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 砍掉不需要的地形和感知
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_heading_envs = 1.0

        # ==========================================
        # 📊 监控中心 (Logging & Stats)
        # 核心思想：利用权重为 0.0 的 Reward 充当日志探头
        # 这样它们的数据会原封不动地出现在 Tensorboard 里，但不会干扰梯度更新！
        # ==========================================
        from isaaclab.managers import RewardTermCfg, SceneEntityCfg
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

        # 1. 监控：每回合摔倒的次数 (Termination Reason)
        self.rewards.log_base_contact = RewardTermCfg(
            func=mdp.illegal_contact, 
            weight=0.0, 
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0}
        )

        # 2. 监控：动作剧烈程度 (Action Stats)
        self.rewards.log_action_rate = RewardTermCfg(
            func=mdp.action_rate_l2, 
            weight=0.0, 
        )

        # 3. 监控：实际产生的关节加速度大小 (Joint Acc)
        self.rewards.log_joint_acc = RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=0.0,
        )
        
        # 4. 监控：关节是否逼近极限位置
        self.rewards.log_joint_limits = RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=0.0,
        )

class Go2PulseFlatEnvCfg_PLAY(Go2PulseFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
