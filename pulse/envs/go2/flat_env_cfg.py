from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY
)

@configclass
class Go2PulseFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # FIXME:TRAIN CFG

        # 这里写你自己的修改

        # 奖励可以按需改
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5


class Go2PulseFlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # FIXME:PLAY CFG