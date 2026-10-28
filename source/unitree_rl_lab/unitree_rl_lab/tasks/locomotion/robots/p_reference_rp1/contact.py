"""PhysX normal/total force separation, sampled at every physical step.

One filtered sensor per body avoids PhysX's many-body filtering limitation.
Filter 0 is always the ground; foot sensors additionally filter the robot.
"""

from types import SimpleNamespace

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
GROUND_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"


class ReferenceContactSensor(ContactSensor):
    def _initialize_impl(self):
        super()._initialize_impl()
        if self.contact_physx_view.filter_count != len(self.cfg.filter_prim_paths_expr):
            raise RuntimeError(f"Contact filters did not resolve: {self.cfg.filter_prim_paths_expr}")
        data = self._data
        # Aliases used by the newer RoboLab API; the installed net_forces are normal only.
        data.net_normal_forces_w = data.net_forces_w
        data.net_normal_forces_w_history = data.net_forces_w_history
        data.normal_force_matrix_w_history = data.force_matrix_w_history
        if self.cfg.track_friction_forces:
            data.total_forces_w = torch.zeros_like(data.net_forces_w)
            data.total_forces_w_history = torch.zeros_like(data.net_forces_w_history)
        else:
            # These sensors only expose normal forces. Reuse the parent buffers
            # instead of copying and rolling the same history every physics step.
            data.total_forces_w = data.net_forces_w
            data.total_forces_w_history = data.net_forces_w_history

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        if not self.cfg.track_friction_forces:
            return
        # Match the parent's full-batch fast path; tensor indices otherwise
        # launch redundant gather/scatter kernels for every environment.
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        data = self._data
        total = data.net_forces_w[env_ids]
        total = total + torch.nan_to_num(data.friction_forces_w[env_ids], nan=0.0).sum(dim=2)
        data.total_forces_w[env_ids] = total
        data.total_forces_w_history[env_ids] = data.total_forces_w_history[env_ids].roll(1, dims=1)
        data.total_forces_w_history[env_ids, 0] = total

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.cfg.track_friction_forces and hasattr(self._data, "total_forces_w"):
            indices = slice(None) if env_ids is None else env_ids
            self._data.total_forces_w[indices] = 0.0
            self._data.total_forces_w_history[indices] = 0.0


@configclass
class ReferenceContactSensorCfg(ContactSensorCfg):
    class_type: type = ReferenceContactSensor


def configure_contacts(scene, body_names):
    for body in body_names:
        foot = body in FEET
        filters = [GROUND_PATH]
        if foot:
            # PhysX requires each filter to resolve to one body per environment.
            filters.extend("{ENV_REGEX_NS}/Robot/" + other for other in body_names if other != body)
        setattr(scene, "self_contact_" + body, ReferenceContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body,
            filter_prim_paths_expr=filters, history_length=3, update_period=0.005,
            track_friction_forces=foot, max_contact_data_count_per_prim=128 if foot else 32,
        ))


def sensor_view(env, sensor_cfg):
    """Present the newer RP1 sensor fields without changing existing sensor semantics."""
    key = (env.common_step_counter, sensor_cfg.name)
    cached = getattr(env, "_rp1_contact_view", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    sensor = env.scene.sensors[sensor_cfg.name]
    normal = sensor.data
    total = normal.net_forces_w.clone()
    history = normal.net_forces_w_history.clone()
    for foot in FEET:
        index = sensor.body_names.index(foot)
        data = env.scene.sensors["self_contact_" + foot].data
        total[:, index] = data.total_forces_w[:, 0]
        history[:, :, index] = data.total_forces_w_history[:, :, 0]
    view = SimpleNamespace(data=SimpleNamespace(
        net_forces_w=total, net_forces_w_history=history,
        net_normal_forces_w=normal.net_forces_w,
        net_normal_forces_w_history=normal.net_forces_w_history,
        normal_force_matrix_w_history=None,
        current_air_time=normal.current_air_time, current_contact_time=normal.current_contact_time,
    ))
    env._rp1_contact_view = (key, view)
    return view
