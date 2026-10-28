# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RP1 delayed PD/envelope adapted to the installed Isaac Lab actuator API.

Source: robolab/assets/robots/rp1_actuators.py. No encoder-bias correction.
"""

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, IdealPDActuator
from isaaclab.utils import configclass


def dc_motor_envelope(joint_vel, saturation_effort, velocity_limit, max_motor_effort):
    corner = velocity_limit * (1.0 + max_motor_effort / saturation_effort)
    velocity = torch.clamp(joint_vel, min=-corner, max=corner)
    upper = torch.minimum(saturation_effort * (1.0 - velocity / velocity_limit), max_motor_effort)
    lower = torch.maximum(saturation_effort * (-1.0 - velocity / velocity_limit), -max_motor_effort)
    return lower, upper


class ReferenceRP1Actuator(DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.keep_velocity = self._parse_joint_parameter(cfg.keep_velocity, 0.0)
        self.rated_effort = self._parse_joint_parameter(cfg.rated_effort, 0.0)
        self.rated_velocity = self._parse_joint_parameter(cfg.rated_velocity, 0.0)
        if not torch.all((self.keep_velocity > 0) & (self.keep_velocity < self.velocity_limit)):
            raise ValueError("keep_velocity must lie within (0, velocity_limit)")
        if not torch.all((self.rated_effort > 0) & (self.rated_effort <= self.effort_limit)):
            raise ValueError("rated_effort must lie within (0, effort_limit]")
        if not torch.all((self.rated_velocity > 0) & (self.rated_velocity <= self.velocity_limit)):
            raise ValueError("rated_velocity must lie within (0, velocity_limit]")
        self._saturation_effort = self.effort_limit / (1.0 - self.keep_velocity / self.velocity_limit)
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def compute(self, control_action, joint_pos, joint_vel):
        self._joint_vel[:] = joint_vel
        # With zero delay the three delay buffers return the current targets.
        # Skip their gathers/copies while retaining the identical PD and motor
        # envelope, and retain the normal reset/randomization behavior.
        if self.cfg.max_delay == 0:
            return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort):
        lower, upper = dc_motor_envelope(
            self._joint_vel, self._saturation_effort, self.velocity_limit, self.effort_limit
        )
        return torch.clamp(effort, min=lower, max=upper)


@configclass
class ReferenceRP1ActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = ReferenceRP1Actuator
    keep_velocity: dict[str, float] | float | None = None
    rated_effort: dict[str, float] | float | None = None
    rated_velocity: dict[str, float] | float | None = None

