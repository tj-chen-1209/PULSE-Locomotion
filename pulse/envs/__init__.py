# pulse/envs/__init__.py
import gymnasium as gym

# 注册训练环境
#! 这里是task后面的名字，相当于注册
gym.register(
    id="Go2-Pulse-Flat-v0",  # 这个名字就是你跑 train.py 时 --task 后面跟的名字
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 告诉系统，我们的环境也是基于 ManagerBasedRLEnv
    disable_env_checker=True,
    kwargs={
        # 指定刚刚你写的配置类
        "env_cfg_entry_point": "pulse.envs.go2.flat_env_cfg:Go2PulseFlatEnvCfg",
        # 使用 PULSE 的 PPO runner 配置
        "rsl_rl_cfg_entry_point": "pulse.agents.go2_ppo_cfg:Go2PULSEFlatPPORunnerCfg",
    },
)

# 注册测试(Play)环境
gym.register(
    id="Go2-Pulse-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 注意这里用的是带有 _PLAY 后缀的配置
        "env_cfg_entry_point": "pulse.envs.go2.flat_env_cfg:Go2PulseFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "pulse.agents.go2_ppo_cfg:Go2PULSEFlatPPORunnerCfg",
    },
)