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

        # ==========================================
        # 命令分布修正
        #
        # 原因 1：heading_command=True 让 yaw 命令随时间衰减到 0，
        #         策略永远没见过"持续恒定 yaw rate"，导致 eval OOD。
        #         改为 False 后，yaw rate 直接从 range 采样，与 eval 一致。
        #
        # 原因 2：rel_standing_envs=0.02 太低，原地转圈几乎没训练过。
        #         提高到 0.20，让 20% 的环境专门训练低速 / 原地旋转。
        # ==========================================
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.rel_heading_envs = 0.0

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

class Go2PulseFlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # FIXME:PLAY CFG