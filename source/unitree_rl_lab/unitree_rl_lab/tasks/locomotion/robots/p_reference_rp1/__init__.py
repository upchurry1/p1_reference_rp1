"""P1 reference experiment; registration never validates unfilled motor data."""

import gymnasium as gym

gym.register(
    id="UESTC-P1-Reference_RP1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:ReferenceRP1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:ReferenceRP1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ReferenceRP1RunnerCfg",
    },
)

