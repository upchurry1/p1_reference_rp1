"""Command sampling and bounded start-step guidance for low-speed P1 fine-tuning."""

import math

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from ...mdp.p_robot.foot_trajectory_rewards import _sole_clearance


class LowSpeedVelocityCommand(UniformVelocityCommand):
    """Keep standing/full-range commands, and oversample straight slow starts."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        edges = cfg.low_speed_edges
        if not 0 <= cfg.low_speed_probability <= 1:
            raise ValueError("low_speed_probability must be in [0, 1]")
        if len(edges) < 2 or not all(math.isfinite(x) for x in edges):
            raise ValueError("Low-speed bin edges must be finite")
        if edges[0] <= 0 or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("Low-speed bin edges must be positive and increasing")
        if edges[-1] > cfg.ranges.lin_vel_x[1]:
            raise ValueError("Low-speed bins exceed the task's forward limit")
        self._low_speed_edges = torch.tensor(edges, device=self.device)
        self.is_low_speed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.is_low_speed_env[env_ids] = False
        selected = (torch.rand(len(env_ids), device=self.device) < self.cfg.low_speed_probability)
        selected &= ~self.is_standing_env[env_ids]
        slow_ids = env_ids[selected]
        bins = torch.randint(len(self._low_speed_edges) - 1, (len(slow_ids),), device=self.device)
        low, high = self._low_speed_edges[bins], self._low_speed_edges[bins + 1]
        self.vel_command_b[slow_ids, 0] = low + (high - low) * torch.rand(len(slow_ids), device=self.device)
        self.vel_command_b[slow_ids, 1:] = 0.0
        # The parent heading controller must not turn a straight-start sample.
        self.is_heading_env[slow_ids] = False
        self.is_low_speed_env[slow_ids] = True


@configclass
class LowSpeedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = LowSpeedVelocityCommand
    # 80% moving * 75% slow = 60% of all commands; 20% remain full-range.
    low_speed_probability: float = 0.75
    low_speed_edges: tuple[float, ...] = (0.05, 0.15, 0.25, 0.40)


def track_low_speed_velocity(
    env, std: float = 0.5, low_speed_std: float = 0.15,
    tight_speed: float = 0.3, full_speed: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Tight low-speed tracking, smoothly returning to the original high-speed width."""
    if not 0 < low_speed_std <= std or not 0 <= tight_speed < full_speed:
        raise ValueError("Invalid low-speed tracking widths or transition speeds")
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")[:, :2]
    velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, :2]
    blend = ((command.norm(dim=1) - tight_speed) / (full_speed - tight_speed)).clamp(0.0, 1.0)
    width = low_speed_std + (std - low_speed_std) * blend
    reward = torch.exp(-(command - velocity).square().sum(dim=1) / width.square())
    return reward * (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7


class LowSpeedStepReward(ManagerTermBase):
    """Finite unloading/lift budgets, renewed only by alternating advancing landings.

    A planted crouch has zero lift credit. Holding a shifted load or a raised
    foot cannot pay repeatedly. Completed steps require real single support,
    sole clearance, foot advance and body progress; flight invalidates a swing.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg, sensor_cfg = cfg.params["asset_cfg"], cfg.params["sensor_cfg"]
        if len(asset_cfg.body_ids) != 2 or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Low-speed steps require ordered left/right feet")
        self._robot = env.scene[asset_cfg.name]
        self._rows = torch.arange(env.num_envs, device=env.device)
        self._weight = self._robot.data.default_mass.sum(dim=1).to(env.device) * 9.81
        self.target_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.best_unload = torch.zeros(env.num_envs, device=env.device)
        self.best_lift = torch.zeros_like(self.best_unload)
        self.air_time = torch.zeros_like(self.best_unload)
        self.ground_time = torch.zeros_like(self.best_unload)
        self.peak_height = torch.zeros_like(self.best_unload)
        self.since_step = torch.zeros_like(self.best_unload)
        self.invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.start_foot = torch.zeros(env.num_envs, 3, device=env.device)
        self.start_root = torch.zeros_like(self.start_foot)
        self.completed_steps = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.target_foot[ids] = -1
        for value in (self.best_unload, self.best_lift, self.air_time, self.ground_time,
                      self.peak_height, self.since_step, self.invalid, self.start_foot, self.start_root, self.completed_steps):
            value[ids] = 0

    def __call__(
        self, env, asset_cfg, sensor_cfg, command_name="base_velocity",
        max_speed=0.45, contact_force=5.0, target_clearance=0.02,
        min_clearance=0.01, min_air_time=0.06, max_air_time=1.2,
        landing_time=0.04, min_step_interval=0.25,
        unload_credit=0.06, lift_credit=0.12, step_credit=0.25,
    ):
        command = env.command_manager.get_command(command_name)
        speed = command[:, :2].norm(dim=1)
        moving = (speed > 0.03) & (speed <= max_speed) & (command[:, 2].abs() <= 0.2)
        heights = _sole_clearance(env, asset_cfg)
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp_min(0.0)
        contact = forces > contact_force
        planted = contact & (heights <= 0.005)
        # Start with the lighter foot: no hidden, randomly assigned phase target.
        initial = moving & (self.target_foot < 0) & planted.all(dim=1)
        self.target_foot = torch.where(initial, forces.argmin(dim=1), self.target_foot)
        foot = self.target_foot.clamp_min(0)
        foot_pos = self._robot.data.body_pos_w[:, asset_cfg.body_ids][self._rows, foot]
        root_pos = self._robot.data.root_pos_w
        self.start_foot = torch.where(initial[:, None], foot_pos, self.start_foot)
        self.start_root = torch.where(initial[:, None], root_pos, self.start_root)
        force = forces[self._rows, foot]
        support = forces[self._rows, 1 - foot]
        height = heights[self._rows, foot]
        supported = (support > 0.45 * self._weight) & (forces.sum(dim=1) > 0.6 * self._weight)
        upright = -self._robot.data.projected_gravity_b[:, 2] > 0.85
        active = moving & (self.target_foot >= 0)
        self.since_step = torch.where(moving, self.since_step + env.step_dt, 0.0)
        airborne = (force <= contact_force) & (height > 0.003) & active
        self.invalid |= active & ~contact.any(dim=1)
        self.invalid |= airborne & ~supported
        valid = active & supported & upright & ~self.invalid

        share = force / forces.sum(dim=1).clamp_min(1.0)
        unload = ((0.5 - share) / 0.35).clamp(0.0, 1.0)
        lift = (height.clamp_min(0.0) / target_clearance).clamp(0.0, 1.0) * unload
        new_unload = torch.where(valid, (unload - self.best_unload).clamp_min(0.0), 0.0)
        new_lift = torch.where(valid, (lift - self.best_lift).clamp_min(0.0), 0.0)
        self.best_unload += new_unload
        self.best_lift += new_lift

        self.air_time += airborne * env.step_dt
        self.peak_height = torch.where(airborne, torch.maximum(self.peak_height, height), self.peak_height)
        grounded = planted[self._rows, foot] & active
        self.ground_time = torch.where(grounded, self.ground_time + env.step_dt, 0.0)
        ended = grounded & (self.ground_time >= landing_time - 1e-6) & (self.air_time > 0)
        direction_b = torch.cat((command[:, :2] / speed.clamp_min(1e-6)[:, None], torch.zeros_like(speed[:, None])), dim=1)
        direction_w = quat_apply(yaw_quat(self._robot.data.root_quat_w), direction_b)
        foot_progress = ((foot_pos - self.start_foot) * direction_w).sum(dim=1)
        root_progress = ((root_pos - self.start_root) * direction_w).sum(dim=1)
        minimum_progress = (speed * 0.1).clamp_min(0.005)
        completed = ended & valid & (self.air_time >= min_air_time - 1e-6) & (self.air_time <= max_air_time)
        completed &= (self.peak_height >= min_clearance) & (foot_progress >= minimum_progress)
        completed &= root_progress >= 0.25 * minimum_progress
        completed &= self.since_step >= min_step_interval - 1e-6
        self.completed_steps[self._rows, foot] += completed
        reward = (unload_credit * new_unload + lift_credit * new_lift + step_credit * completed) / env.step_dt

        # A failed retry retains its spent credit; only a valid opposite-foot
        # transition renews the preparation budgets.
        self.target_foot = torch.where(completed, 1 - foot, self.target_foot)
        self.since_step.masked_fill_(completed, 0.0)
        next_pos = self._robot.data.body_pos_w[:, asset_cfg.body_ids][self._rows, self.target_foot.clamp_min(0)]
        self.start_foot = torch.where(completed[:, None], next_pos, self.start_foot)
        self.start_root = torch.where(completed[:, None], root_pos, self.start_root)
        for value in (self.best_unload, self.best_lift):
            value.masked_fill_(completed | ~moving, 0.0)
        for value in (self.air_time, self.ground_time, self.peak_height):
            value.masked_fill_(ended | completed | ~moving, 0.0)
        self.invalid &= ~(ended | (planted.all(dim=1) & (self.air_time == 0)) | ~moving)
        self.target_foot.masked_fill_(~moving, -1)
        return reward
