"""Build the unbiased P1 asset from the reference motor parameter table."""

import os
from pathlib import Path

from .p import P_CFG
from .p_reference_parameters import P1_REFERENCE_PD
from .reference_rp1_actuator import ReferenceRP1ActuatorCfg


def build_reference_asset(parameters):
    robot = P_CFG.copy()
    default_urdf = Path(__file__).resolve().parents[6] / "urdf_803_nobattery/urdf/p1.urdf"
    robot.spawn.asset_path = str(Path(os.environ.get("P_ROBOT_URDF", default_urdf)).expanduser().resolve())
    robot.actuators = {}
    for group, values in parameters.items():
        kp, kd = P1_REFERENCE_PD[group]
        robot.actuators[group] = ReferenceRP1ActuatorCfg(
            joint_names_expr=[f".*{group}.*"], stiffness=kp, damping=kd,
            armature=values["armature"], effort_limit=values["effort_limit"],
            effort_limit_sim=values["effort_limit"], velocity_limit=values["velocity_limit"],
            velocity_limit_sim=values["velocity_limit"], keep_velocity=values["keep_velocity"],
            rated_effort=values["rated_effort"], rated_velocity=values["rated_velocity"],
            min_delay=0, max_delay=0, friction=0.1, dynamic_friction=0.1, viscous_friction=0.01,
        )
    return robot

