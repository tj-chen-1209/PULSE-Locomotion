from __future__ import annotations

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class PulseBucketVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with an explicit pure-yaw bucket.

    The default Isaac Lab command samples (vx, vy, yaw) independently and uses
    `standing_envs` for the exact-zero command. That gives us plenty of mixed
    turning motions, but almost no dedicated `vx=0, vy=0, yaw!=0` samples.

    This class keeps the original uniform sampler for most envs, then overrides a
    configurable fraction with exact pure-spin commands.
    """

    cfg: "PulseBucketVelocityCommandCfg"

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0 or self.cfg.rel_yaw_envs <= 0.0:
            return

        yaw_bucket_mask = torch.rand(env_ids.numel(), device=self.device) <= self.cfg.rel_yaw_envs
        yaw_env_ids = env_ids[yaw_bucket_mask]
        if yaw_env_ids.numel() == 0:
            return

        # Pure-yaw bucket: no translation, only spin.
        self.vel_command_b[yaw_env_ids, 0] = 0.0
        self.vel_command_b[yaw_env_ids, 1] = 0.0

        yaw_mag = torch.empty(yaw_env_ids.numel(), device=self.device).uniform_(*self.cfg.yaw_only_ang_vel_z_range)
        if self.cfg.yaw_only_bidirectional:
            yaw_sign = torch.where(torch.rand(yaw_env_ids.numel(), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[yaw_env_ids, 2] = yaw_mag * yaw_sign
        else:
            self.vel_command_b[yaw_env_ids, 2] = yaw_mag

        # A yaw-bucket env should not also be treated as "standing" or "heading".
        self.is_standing_env[yaw_env_ids] = False
        if hasattr(self, "is_heading_env"):
            self.is_heading_env[yaw_env_ids] = False


@configclass
class PulseBucketVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`PulseBucketVelocityCommand`.

    `rel_yaw_envs` is interpreted as the probability that a resampled env becomes a
    pure-yaw env. The remaining envs keep the original uniform velocity sampling.
    """

    class_type: type = PulseBucketVelocityCommand

    rel_yaw_envs: float = 0.0
    """Probability of sampling a pure-yaw command on resample."""

    yaw_only_ang_vel_z_range: tuple[float, float] = (0.0, 1.0)
    """Magnitude range for pure-yaw commands in rad/s.

    When ``yaw_only_bidirectional=True``, setting this to ``(0.0, 1.0)``
    produces a uniform signed yaw bucket over ``[-1.0, 1.0]``.
    """

    yaw_only_bidirectional: bool = True
    """Whether to sample both clockwise and counter-clockwise pure-yaw commands."""
