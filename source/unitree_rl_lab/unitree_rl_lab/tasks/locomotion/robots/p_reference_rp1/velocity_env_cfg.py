"""UESTC P1 with RP1-Flat rewards, terrain, commands and pushes.

Motor data must be supplied in p_reference_parameters.py before training.
Observations and action arithmetic are shared with KneeLift.
"""

from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.p import P_CFG
from unitree_rl_lab.assets.robots.p_reference import build_reference_asset
from unitree_rl_lab.assets.robots.p_reference_parameters import P1_REFERENCE_MOTORS
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.actions import DeploymentLatencyJointPositionAction
from ..p_pace.velocity_env_cfg import EventCfg, KneeLiftActionsCfg, RobotEnvCfg, RobotSceneCfg
from .contact import configure_contacts
from .rewards_cfg import P1_BODY_NAMES, ReferenceRP1RewardsCfg
from .terrain import ReferencePlaneImporter


@configclass
class ReferenceCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.2,
        rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ReferenceActionsCfg(KneeLiftActionsCfg):
    # Same arithmetic and default offsets, without PACE encoder-bias subtraction.
    JointPositionAction = KneeLiftActionsCfg().JointPositionAction.replace(
        class_type=DeploymentLatencyJointPositionAction
    )


@configclass
class ReferenceEventsCfg(EventCfg):
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0),
        params={"velocity_range": {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
        }},
    )


@configclass
class ReferenceCurriculumCfg:
    """RP1-Flat has no terrain, command or push progression."""


@configclass
class ReferenceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        class_type=ReferencePlaneImporter,
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True, texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = P_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=0.005,
    )
    # Reuse the existing illumination, without introducing scanners or observations.
    sky_light = RobotSceneCfg().sky_light.copy()


@configclass
class ReferenceRP1EnvCfg(RobotEnvCfg):
    motor_parameters: dict = deepcopy(P1_REFERENCE_MOTORS)
    scene: ReferenceSceneCfg = ReferenceSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ReferenceActionsCfg = ReferenceActionsCfg()
    commands: ReferenceCommandsCfg = ReferenceCommandsCfg()
    events: ReferenceEventsCfg = ReferenceEventsCfg()
    rewards: ReferenceRP1RewardsCfg = ReferenceRP1RewardsCfg()
    curriculum: ReferenceCurriculumCfg = ReferenceCurriculumCfg()

    def __post_init__(self):
        # Motor values are filled directly in p_reference_parameters.py.
        self.scene.robot = build_reference_asset(self.motor_parameters).replace(prim_path="{ENV_REGEX_NS}/Robot")
        path = Path(self.scene.robot.spawn.asset_path)
        if not path.is_file():
            raise FileNotFoundError(f"P1 reference URDF is missing: {path}")
        root = ET.parse(path).getroot()
        for mesh in root.iter("mesh"):
            mesh_path = path.parent / mesh.attrib["filename"]
            if not mesh_path.is_file():
                raise FileNotFoundError(f"P1 reference mesh is missing: {mesh_path}")
        bodies = tuple(link.attrib["name"] for link in root.findall("link"))
        if set(bodies) != set(P1_BODY_NAMES):
            raise ValueError(f"P1 reference body mapping differs from URDF: {bodies}")
        configure_contacts(self.scene, bodies)
        super().__post_init__()


@configclass
class ReferenceRP1PlayEnvCfg(ReferenceRP1EnvCfg):
    """One robot with the same RP1-Flat commands, pushes and fixed distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
