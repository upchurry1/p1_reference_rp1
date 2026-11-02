# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RP1RewardCfg weights with P1's 12 joints and physical foot geometry."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from . import rewards as mdp
from .low_speed import LowSpeedStepReward, track_low_speed_velocity

# The reference imports these unchanged Isaac Lab formulas through its mdp module.
P1_BODY_NAMES = ('pelvis_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_hip_yaw_link', 'left_knee_pitch_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_hip_yaw_link', 'right_knee_pitch_link', 'right_ankle_pitch_link', 'right_ankle_roll_link')


@configclass
class ReferenceRP1RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=track_low_speed_velocity, weight=2.0,
        params={"std": 0.5, "low_speed_std": 0.15, "tight_speed": 0.3, "full_speed": 0.6},
    )
    low_speed_steps = RewTerm(
        func=LowSpeedStepReward, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True),
        },
    )
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2, params={})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={})
    energy = RewTerm(func=mdp.energy, weight=-0.0001, params={})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-05, params={})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-07, params={})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2, params={"zero_on_reset": True})
    action_smoothness_l2 = RewTerm(func=mdp.action_jerk_l2, weight=-0.02)
    joint_rated_velocity = RewTerm(func=mdp.joint_rated_velocity_limit, weight=-0.05)
    joint_rated_effort = RewTerm(func=mdp.joint_rated_effort_limit, weight=-0.04)
    angular_momentum_rate_l2 = RewTerm(func=mdp.angular_momentum_rate_l2, weight=-5e-5)
    yaw_angular_momentum_l2 = RewTerm(func=mdp.yaw_angular_momentum_l2, weight=-1e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_self_contacts,
        weight=-0.5,
        params={
            "ground_sensor_names": tuple("self_contact_" + body for body in P1_BODY_NAMES),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces"
            )
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0, params={})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "velocity_reference": "link",
            "contact_history_length": 3,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.003,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "min": 0.1,
            "max": 0.5,
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*"]), "min": 0.1, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={})
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*hip_roll.*"])},
    )
    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch.*", ".*knee.*"])},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    upward = RewTerm(func=mdp.upward, weight=0.4, params={})
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.2,
        params={
            "pos_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={
            "replace_invalid_hits": True,
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "ankle_height": 0.0408,
            "threshold": 0.02,
        },
    )
