from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Go2PULSEFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_pulse_flat"