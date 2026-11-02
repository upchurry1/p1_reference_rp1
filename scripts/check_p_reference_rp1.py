"""Check the RP1 reference contract, optionally with synthetic-data PhysX smoke.

The motor numbers below are TEST FIXTURES, not identified P1 motor data.
They are never saved into the task's default parameter table.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "source/unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--simulate", action="store_true")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--report", type=Path, default=Path("/tmp/p1_reference_check.json"))
parser.add_argument("--benchmark_actuator", action="store_true", help="Compare zero-delay actuator compute paths only.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
print("Launching reference checks (synthetic motor data only)", flush=True)
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils.types import ArticulationActions

import unitree_rl_lab.tasks
from unitree_rl_lab.assets.robots.p_reference_parameters import (
    P1_REFERENCE_MOTORS, P1_REFERENCE_PD,
)
from unitree_rl_lab.assets.robots.reference_rp1_actuator import (
    ReferenceRP1Actuator, ReferenceRP1ActuatorCfg, dc_motor_envelope,
)
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.actions import DeploymentLatencyJointPositionAction
from unitree_rl_lab.tasks.locomotion.robots.p_pace.velocity_env_cfg import KneeLiftEnvCfg
from unitree_rl_lab.tasks.locomotion.robots.p_reference_rp1.agents import ReferenceRP1RunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.p_reference_rp1.contact import FEET
from unitree_rl_lab.tasks.locomotion.robots.p_reference_rp1 import rewards
from unitree_rl_lab.tasks.locomotion.robots.p_reference_rp1.low_speed import (
    LowSpeedStepReward, LowSpeedVelocityCommand, track_low_speed_velocity,
)
from unitree_rl_lab.tasks.locomotion.robots.p_reference_rp1.velocity_env_cfg import (
    ReferenceRP1EnvCfg,
)
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

TASK = "UESTC-P1-Reference_RP1"
RESULTS = []
BENCHMARK = {}
ORIGINAL_MOTOR_PARAMETERS = deepcopy(P1_REFERENCE_MOTORS)


def passed(message):
    RESULTS.append(message)
    print("PASS:", message, flush=True)


def synthetic_parameters():
    return {group: dict(armature=0.01, effort_limit=100.0, velocity_limit=20.0,
                        keep_velocity=10.0, rated_effort=40.0, rated_velocity=8.0)
            for group in P1_REFERENCE_MOTORS}


def check_configuration():
    assert gym.spec(TASK).kwargs["env_cfg_entry_point"].endswith(":ReferenceRP1EnvCfg")

    cfg = ReferenceRP1EnvCfg(motor_parameters=synthetic_parameters())
    old = KneeLiftEnvCfg()
    assert cfg.observations.to_dict() == old.observations.to_dict()
    assert cfg.observations.policy.history_length == cfg.observations.critic.history_length == 15
    current = cfg.actions.JointPositionAction.to_dict()
    previous = old.actions.JointPositionAction.to_dict()
    current.pop("class_type")
    previous.pop("class_type")
    assert current == previous
    assert cfg.actions.JointPositionAction.class_type is DeploymentLatencyJointPositionAction
    assert vars(cfg.curriculum) == {}
    assert cfg.events.to_dict().keys() == old.events.to_dict().keys()
    events_new, events_old = cfg.events.to_dict(), old.events.to_dict()
    events_new.pop("push_robot")
    events_old.pop("push_robot")
    assert events_new == events_old
    command = cfg.commands.base_velocity
    assert command.resampling_time_range == (4.0, 8.0) and command.rel_standing_envs == 0.2
    assert command.class_type is LowSpeedVelocityCommand and command.low_speed_probability == 0.75
    assert command.heading_command and command.rel_heading_envs == 1.0 and command.heading_control_stiffness == 0.5
    assert command.ranges.lin_vel_x == (-0.6, 1.0) and command.ranges.lin_vel_y == (-0.5, 0.5)
    assert command.ranges.ang_vel_z == (-1.57, 1.57)
    assert cfg.events.push_robot.interval_range_s == (5.0, 8.0)
    assert len(cfg.events.push_robot.params["velocity_range"]) == 6
    terrain = cfg.scene.terrain
    assert terrain.terrain_type == "plane" and terrain.terrain_generator is None
    assert terrain.physics_material.static_friction == terrain.physics_material.dynamic_friction == 1.0
    assert terrain.physics_material.restitution == 0.0
    assert terrain.visual_material.texture_scale == (0.25, 0.25) and terrain.visual_material.project_uvw
    assert not any("scanner" in name for name in vars(cfg.scene))
    assert len([key for key in vars(cfg.scene) if key.startswith("self_contact_")]) == 13
    for group, motor in cfg.scene.robot.actuators.items():
        assert (motor.stiffness, motor.damping) == P1_REFERENCE_PD[group]
        assert motor.friction == motor.dynamic_friction == 0.1 and motor.viscous_friction == 0.01
        assert motor.min_delay == motor.max_delay == 0
        assert not hasattr(motor, "encoder_bias")
    agent = ReferenceRP1RunnerCfg()
    assert agent.clip_actions == 1.0 and agent.policy.init_noise_std == 0.35
    assert agent.min_noise_std == 0.05 and agent.max_noise_std == 0.35
    assert agent.experiment_name == "uestc_p1_reference_rp1"
    passed("observations/actions unchanged; low-speed command/push settings and physical PD match configuration")
    return cfg


def check_low_speed_start(cfg):
    sample_env = SimpleNamespace(num_envs=12000, device="cpu", step_dt=.02,
        scene={"robot": SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(12000)))})
    command = LowSpeedVelocityCommand(cfg.commands.base_velocity.replace(debug_vis=False), sample_env)
    command._resample_command(torch.arange(sample_env.num_envs))
    command._update_command()
    standing, slow = command.is_standing_env, command.is_low_speed_env
    assert abs(float(standing.float().mean()) - .2) < .025
    assert abs(float(slow.float().mean()) - .6) < .025
    assert not command.command[standing].any()
    assert not command.command[slow, 1:].any() and not command.is_heading_env[slow].any()
    for lo, hi in zip((.05, .15, .25), (.15, .25, .4)):
        fraction = ((command.command[:, 0] >= lo) & (command.command[:, 0] < hi) & slow).float().mean()
        assert abs(float(fraction) - .2) < .025
    untouched = command.command[10:].clone()
    command._resample_command(torch.arange(10))
    torch.testing.assert_close(command.command[10:], untouched)
    passed("20/60/20 command mixture, balanced low-speed bins, zero standing/side/yaw, partial resampling")

    class Scene(dict):
        pass

    n = 2
    positions = torch.zeros(n, 2, 3)
    positions[:, :, 2] = .0408
    positions[:, 0, 1], positions[:, 1, 1] = .1, -.1
    force = torch.zeros(n, 2, 3)
    force[:, :, 2] = 100.
    data = SimpleNamespace(
        body_pos_w=positions, body_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(n, 2, 1),
        root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(n, 1),
        root_lin_vel_w=torch.zeros(n, 3), projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(n, 1),
        default_mass=torch.full((n, 2), 10.),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(n, 3)
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=force))}
    target = torch.zeros(n, 3)
    env = SimpleNamespace(scene=scene, num_envs=n, device="cpu", step_dt=.02,
                          command_manager=SimpleNamespace(get_command=lambda name: target))
    params = dict(asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
                  sensor_cfg=SceneEntityCfg("contact_forces", body_ids=[0, 1]))
    term = LowSpeedStepReward(SimpleNamespace(params=params), env)

    def tick():
        result = term(env, **params)
        assert torch.isfinite(result).all()
        return result

    assert not tick().any()  # Standing.
    target[:, 0] = .1
    assert not tick().any()  # Planted knees/body motion earns no lift credit.
    still = track_low_speed_velocity(env)
    torch.testing.assert_close(still, torch.full((n,), .641180388), rtol=1e-5, atol=1e-6)
    data.root_lin_vel_w[:, 0] = .1
    torch.testing.assert_close(track_low_speed_velocity(env), torch.ones(n))
    data.root_lin_vel_w.zero_()

    force[:, 0, 2], force[:, 1, 2] = 20., 180.
    assert (tick() > 0).all()  # Pre-takeoff unloading credit.
    for _ in range(10):
        assert not tick().any()  # No repeated reward for holding a lean.
    force[:, 0, 2], force[:, 1, 2] = 0., 200.
    positions[:, 0, 2] = .0558  # Real 15 mm sole clearance.
    assert (tick() > 0).all()
    for _ in range(4):
        assert not tick().any()
    # In-place landing must not count as a completed advancing step.
    positions[:, 0, 2] = .0408
    force[:, :, 2] = 100.
    for _ in range(3):
        assert not tick().any()
    assert not term.completed_steps.any()
    # A repeated failed attempt cannot refill its unloading/lift budget.
    force[:, 0, 2], force[:, 1, 2] = 0., 200.
    positions[:, 0, 2] = .0558
    for _ in range(5):
        assert not tick().any()
    positions[:, 0, 0] += .02
    data.root_pos_w[:, 0] += .01
    positions[:, 0, 2] = .0408
    force[:, :, 2] = 100.
    landed = tick() + tick()
    assert (landed > 0).all() and (term.completed_steps[:, 0] == 1).all()
    assert (term.target_foot == 1).all()
    # Lifting the same foot again cannot earn the next alternating-step credit.
    force[:, 0, 2], force[:, 1, 2] = 0., 200.
    positions[:, 0, 2] = .0558
    for _ in range(5):
        assert not tick().any()
    before = term.completed_steps[1].clone()
    term.reset(torch.tensor([0]))
    assert not term.completed_steps[0].any()
    torch.testing.assert_close(term.completed_steps[1], before)
    term.reset()
    force[:, :, 2] = 100.
    positions[:, :, 2] = .0408
    tick()
    force[:, :, 2] = 0.
    positions[:, :, 2] = .0708
    for _ in range(5):
        assert not tick().any()  # Two-foot flight is invalid.
    force[:, :, 2] = 100.
    positions[:, :, 2] = .0408
    positions[:, :, 0] += .1
    data.root_pos_w[:, 0] += .1
    for _ in range(3):
        tick()
    assert not term.completed_steps.any()
    # An airborne swing with only a weak floor graze by the other foot is not support.
    term.reset()
    tick()
    force[:, 0, 2], force[:, 1, 2] = 0., 20.
    positions[:, 0, 2] = .0558
    for _ in range(5):
        assert not tick().any()
    force[:, :, 2] = 100.
    positions[:, 0, 2] = .0408
    positions[:, 0, 0] += .1
    data.root_pos_w[:, 0] += .1
    for _ in range(3):
        tick()
    assert not term.completed_steps.any()
    # Holding a properly unloaded leg indefinitely neither pays nor counts at landing.
    term.reset()
    tick()
    force[:, 0, 2], force[:, 1, 2] = 0., 200.
    positions[:, 0, 2] = .0558
    assert (tick() > 0).all()
    for _ in range(65):
        assert not tick().any()
    force[:, :, 2] = 100.
    positions[:, 0, 2] = .0408
    positions[:, 0, 0] += .1
    data.root_pos_w[:, 0] += .1
    for _ in range(3):
        tick()
    assert not term.completed_steps.any()
    target.zero_()
    assert not tick().any() and (term.target_foot == -1).all()
    target[:, 0] = .5
    assert not tick().any()  # Additional start guidance is restricted to low speed.
    passed("tight tracking; bounded unload/lift credit; no crouch/hold/in-place/repeated-foot/flight exploit; advancing landing, partial reset and zero-command gates")


def check_actions_and_motor():
    cfg = ReferenceRP1EnvCfg(motor_parameters=synthetic_parameters()).actions.JointPositionAction
    expected = {
        "hip_roll": (0.0, -0.16, 0.16), "hip_pitch": (-0.2, -0.48, 0.12),
        "hip_yaw": (0.0, -0.1, 0.1), "knee_pitch": (0.2, -0.02, 0.75),
        "ankle_pitch": (-0.05, -0.19, 0.13), "ankle_roll": (0.0, -0.1, 0.1),
    }
    for group, (home, lo, hi) in expected.items():
        pattern = f".*{group}.*"
        term = SimpleNamespace(
            cfg=cfg, _raw_actions=torch.zeros(2, 1), _scale=cfg.scale[pattern],
            _clip=torch.tensor([[cfg.clip[pattern]]]), _offset=home,
            _pending_actions=torch.zeros(2, 1), _inference_elapsed=torch.ones(2),
        )
        DeploymentLatencyJointPositionAction.process_actions(term, torch.tensor([[-2.0], [2.0]]))
        torch.testing.assert_close(term._pending_actions[:, 0], torch.tensor([lo, hi]))
    motor_cfg = ReferenceRP1ActuatorCfg(
        joint_names_expr=["joint"], stiffness=180.0, damping=10.0,
        effort_limit=100.0, velocity_limit=20.0, keep_velocity=10.0,
        rated_effort=40.0, rated_velocity=8.0, min_delay=0, max_delay=0,
    )
    motor = ReferenceRP1Actuator(motor_cfg, joint_names=["joint"], joint_ids=[0], num_envs=2, device="cpu")
    motor.reset(torch.arange(2))
    result = motor.compute(ArticulationActions(torch.full((2, 1), 0.2), torch.zeros(2, 1), torch.zeros(2, 1)),
                           torch.full((2, 1), 0.1), torch.full((2, 1), 0.5))
    torch.testing.assert_close(result.joint_efforts, torch.full((2, 1), 13.0))
    speed = torch.tensor([-40., -20., -10., 0., 10., 20., 40.])
    lower, upper = dc_motor_envelope(speed, torch.tensor(200.), torch.tensor(20.), torch.tensor(100.))
    torch.testing.assert_close(upper, torch.tensor([100., 100., 100., 100., 100., 0., -100.]))
    torch.testing.assert_close(lower, -upper.flip(0))
    motor.reset(torch.tensor([0]))
    passed("all action extrema, unbiased immediate PD, partial reset and four-quadrant plateau/braking bounds")


def check_reward_math():
    normal = torch.zeros(2, 3, 2, 3)
    ground = torch.zeros(2, 3, 2, 1, 3)
    normal[0, :, 0, 2] = ground[0, :, 0, 0, 2] = 100.0
    normal[1, :, 1, 0] = 10.0
    torch.testing.assert_close(rewards._self_contact_count(normal, ground), torch.tensor([0., 1.]))
    env = SimpleNamespace(num_envs=2, device="cpu", action_manager=SimpleNamespace(
        action=torch.zeros(2, 2), prev_action=torch.zeros(2, 2)))
    jerk = rewards.action_jerk_l2(SimpleNamespace(params={}), env)
    assert not jerk(env).any()
    env.action_manager.action[:] = 1
    assert not jerk(env).any()
    env.action_manager.action[:] = 2
    assert not jerk(env).any()
    env.action_manager.action[0] = 4
    torch.testing.assert_close(jerk(env), torch.tensor([2., 2.]))
    jerk.reset(torch.tensor([0]))
    env.action_manager.action[:] = 0
    assert jerk(env)[0] == 0
    env.action_manager.prev_action[:] = 1
    torch.testing.assert_close(rewards.action_rate_l2(env, True), torch.full((2,), 2.0))
    env.reset_terminated = torch.tensor([True, False])
    assert torch.equal(rewards.is_terminated(env), env.reset_terminated)
    passed("normal-ground self-contact separation; action jerk warmup/partial reset; action rate and timeout exclusion")

    class Scene(dict):
        pass

    n = 3
    command = torch.tensor([[1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    force = torch.zeros(n, 3, 2, 3)
    force[0, :, 0, 2] = 100
    force[1, :, :, 2] = 100
    normal_force = force.clone()
    sensor = SimpleNamespace(data=SimpleNamespace(
        net_forces_w=force[:, 0], net_forces_w_history=force,
        net_normal_forces_w=normal_force[:, 0], net_normal_forces_w_history=normal_force,
        current_air_time=torch.tensor([[0., .3], [0., 0.], [.2, .2]]),
        current_contact_time=torch.tensor([[.2, 0.], [.2, .2], [0., 0.]]),
    ))
    positions = torch.tensor([[[-1., 0., .0408], [1., 0., .0708]]]).repeat(n, 1, 1)
    velocity = torch.tensor([[[0., -1., 0.], [0., 1., 0.]]]).repeat(n, 1, 1)
    data = SimpleNamespace(
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        body_pos_w=positions, body_link_lin_vel_w=torch.full((n, 2, 3), .2),
        body_com_pos_w=torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]]).repeat(n, 1, 1),
        body_com_lin_vel_w=velocity,
        body_com_quat_w=torch.tensor([[[1., 0., 0., 0.]]]).repeat(n, 2, 1),
        body_com_ang_vel_w=torch.zeros(n, 2, 3), joint_vel=torch.tensor([[8., 16.]]).repeat(n, 1),
    )
    robot = SimpleNamespace(data=data, device="cpu", root_physx_view=SimpleNamespace(
        get_masses=lambda: torch.ones(n, 2),
        get_inertias=lambda: torch.eye(3).reshape(1, 1, 9).repeat(n, 2, 1),
    ), actuators={"test": SimpleNamespace(joint_indices=[0, 1], rated_velocity=8., rated_effort=40.,
                                        applied_effort=torch.tensor([[40., 80.]]).repeat(n, 1))})
    scene = Scene(robot=robot)
    mock = SimpleNamespace(scene=scene, num_envs=n, device="cpu", step_dt=.02, common_step_counter=0,
                           command_manager=SimpleNamespace(get_command=lambda name: command))
    # Supply the same newer-API view that the physical sensors construct.
    mock._rp1_contact_view = ((0, "test"), sensor)
    selector = SceneEntityCfg("test", body_ids=[0, 1])
    asset = SceneEntityCfg("robot", body_ids=[0, 1])
    torch.testing.assert_close(rewards.feet_air_time_positive_biped(mock, .4, selector, "total"),
                               torch.tensor([.2, 0., 0.]))
    torch.testing.assert_close(rewards.feet_contact_without_cmd(mock, selector, "total"), torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(rewards.feet_height(mock, selector, asset, ankle_height=.0408,
                                                 threshold=.02, force_type="total"), torch.tensor([1., 0., 0.]))
    force[0, :, 0, :2] = torch.tensor([3., 4.])
    assert rewards.body_force(mock, selector, threshold=100., force_type="total")[0] > 0
    assert rewards.body_force(mock, selector, threshold=100., force_type="normal")[0] == 0
    force[0, :, 0, 0] = 400
    assert rewards.feet_stumble(mock, selector, "total")[0]
    torch.testing.assert_close(rewards.joint_rated_velocity_limit(mock), torch.ones(n))
    torch.testing.assert_close(rewards.joint_rated_effort_limit(mock), torch.ones(n))
    torch.testing.assert_close(rewards._angular_momentum(mock), torch.tensor([[0., 0., 2.]]).repeat(n, 1))
    rate = rewards.angular_momentum_rate_l2(SimpleNamespace(params={}), mock)
    assert not rate(mock).any()
    velocity.mul_(2)
    mock.common_step_counter += 1
    torch.testing.assert_close(rate(mock), torch.full((n,), 1e4))
    rate.reset(torch.tensor([0]))
    velocity.mul_(2)
    mock.common_step_counter += 1
    assert rate(mock)[0] == 0
    passed("single support/zero commands/airborne feet, normal versus friction-inclusive total forces, rated limits and analytic angular momentum")
    first = rewards._angular_momentum(mock)
    assert rewards._angular_momentum(mock) is first
    velocity.mul_(2)
    mock.common_step_counter += 1
    torch.testing.assert_close(rewards._angular_momentum(mock), first * 2)
    # Episode randomization may change masses: the next step must reread them.
    robot.root_physx_view.get_masses = lambda: torch.full((n, 2), 2.0)
    mock.common_step_counter += 1
    torch.testing.assert_close(rewards._angular_momentum(mock), first * 4)
    passed("momentum reuse within a control step, invalidation on the next step and changed masses")


def check_actuator_fast_path():
    cfg = ReferenceRP1ActuatorCfg(
        joint_names_expr=["joint"], stiffness=180.0, damping=10.0,
        effort_limit=100.0, velocity_limit=20.0, keep_velocity=10.0,
        rated_effort=40.0, rated_velocity=8.0, min_delay=0, max_delay=0,
    )

    def make_motor(config, count=4, device="cpu"):
        return ReferenceRP1Actuator(config, joint_names=["joint"], joint_ids=[0], num_envs=count, device=device)

    for delay in (0, 2):
        config = cfg.replace(min_delay=delay, max_delay=delay)
        fast, reference = make_motor(config), make_motor(config)
        ids = torch.arange(4)
        fast.reset(ids)
        reference.reset(ids)
        for step in range(12):
            if step == 6:
                fast.reset(ids[:2])
                reference.reset(ids[:2])
            position = torch.full((4, 1), step * 0.1)
            velocity = torch.tensor([[-40.], [-10.], [10.], [40.]])
            targets = (position, torch.zeros_like(position), torch.ones_like(position))
            current = fast.compute(ArticulationActions(*targets), torch.zeros_like(position), velocity)
            reference._joint_vel[:] = velocity
            expected = DelayedPDActuator.compute(
                reference, ArticulationActions(*targets), torch.zeros_like(position), velocity
            )
            torch.testing.assert_close(current.joint_efforts, expected.joint_efforts, rtol=0, atol=0)
            torch.testing.assert_close(fast.computed_effort, reference.computed_effort, rtol=0, atol=0)
    passed("zero/nonzero actuator delay matches the original PD path bit-for-bit, including partial resets and saturation")

    if args.benchmark_actuator:
        fast, reference = make_motor(cfg, args.num_envs, args.device), make_motor(cfg, args.num_envs, args.device)
        ids = torch.arange(args.num_envs, device=args.device)
        fast.reset(ids)
        reference.reset(ids)
        position = torch.full((args.num_envs, 1), .2, device=args.device)
        zero = torch.zeros_like(position)
        velocity = torch.full_like(position, .5)

        def run(motor, original):
            action = ArticulationActions(position, zero, zero)
            if original:
                motor._joint_vel[:] = velocity
                return DelayedPDActuator.compute(motor, action, zero, velocity)
            return motor.compute(action, zero, velocity)

        def sync():
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)

        with torch.inference_mode():
            for _ in range(20):
                run(reference, True)
                run(fast, False)
            samples = {"original_ms": [], "optimized_ms": []}
            for trial in range(3):
                order = [("original_ms", reference, True), ("optimized_ms", fast, False)]
                for label, motor, original in order[::1 if trial % 2 == 0 else -1]:
                    sync()
                    start = time.perf_counter()
                    for _ in range(200):
                        run(motor, original)
                    sync()
                    samples[label].append((time.perf_counter() - start) * 1000 / 200)
        BENCHMARK.update(device=args.device, num_envs=args.num_envs, scope="one actuator compute, not collection", **samples)
        print("ACTUATOR BENCHMARK:", json.dumps(BENCHMARK), flush=True)


def check_simulation(cfg):
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # All conversion/cache/export artifacts stay temporary; source URDF and defaults are untouched.
    with tempfile.TemporaryDirectory(prefix="p1_reference_smoke_") as directory:
        cfg.scene.robot.spawn.usd_dir = directory
        cfg.scene.robot.spawn.force_usd_conversion = True
        env = gym.make(TASK, cfg=cfg)
        try:
            base = env.unwrapped
            robot = base.scene["robot"]
            action = base.action_manager.get_term("JointPositionAction")
            torch.testing.assert_close(action._offset, robot.data.default_joint_pos)
            assert not base.curriculum_manager.active_terms
            cfg_before = deepcopy(base.cfg.commands.base_velocity.to_dict())
            push_before = deepcopy(base.cfg.events.push_robot.to_dict())
            shader = base.sim.stage.GetPrimAtPath("/World/ground/terrain/rp1_visual_material/Shader")
            assert shader.IsValid() and shader.GetAttribute("inputs:project_uvw").Get()
            env.reset()
            for _ in range(args.steps):
                obs, reward, _, _, _ = env.step(torch.zeros(args.num_envs, 12, device=robot.device))
                assert obs["policy"].shape == (args.num_envs, 675)
                assert obs["critic"].shape == (args.num_envs, 720)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
                assert torch.isfinite(reward).all()
            assert base.cfg.commands.base_velocity.to_dict() == cfg_before
            assert base.cfg.events.push_robot.to_dict() == push_before
            assert "gait" not in base.reward_manager.active_terms
            assert len(base.reward_manager.active_terms) == 33
            command = base.command_manager.get_term("base_velocity")
            command.heading_target[:] = robot.data.heading_w + 1.0
            command.is_heading_env[:] = True
            command.is_standing_env[:] = False
            command._update_command()
            torch.testing.assert_close(command.command[:, 2], torch.full((args.num_envs,), .5, device=robot.device))
            command.is_standing_env[:] = True
            command._update_command()
            assert not command.command.any()
            # Validate both the force reconstruction and newest-first physical-step history.
            for foot in FEET:
                sensor = base.scene.sensors["self_contact_" + foot]
                data = sensor.data
                expected = data.net_forces_w + torch.nan_to_num(data.friction_forces_w).sum(dim=2)
                torch.testing.assert_close(data.total_forces_w, expected)
                torch.testing.assert_close(data.total_forces_w_history[:, 0], expected)
            for name, sensor in base.scene.sensors.items():
                if name.startswith("self_contact_") and not sensor.cfg.track_friction_forces:
                    data = sensor.data
                    torch.testing.assert_close(data.total_forces_w, data.net_forces_w)
                    torch.testing.assert_close(data.total_forces_w_history, data.net_forces_w_history)
            base._reset_idx(torch.tensor([0], device=robot.device))
            for foot in FEET:
                # Access the cleared buffer directly: .data would resample the still-live PhysX view.
                assert not base.scene.sensors["self_contact_" + foot]._data.total_forces_w_history[0].any()
            export_deploy_cfg(base, directory)
            exported = yaml.safe_load((Path(directory) / "params/deploy.yaml").read_text())
            assert "motor_identification" not in exported
            assert exported["motor_reference"]["parameters_by_group"] == synthetic_parameters()
            assert exported["commands"]["base_velocity"]["heading_command"]
            assert P1_REFERENCE_MOTORS == ORIGINAL_MOTOR_PARAMETERS
            assert exported["commands"]["base_velocity"]["class_type"] == "LowSpeedVelocityCommand"
            passed(f"PhysX: {args.num_envs} envs, {args.steps} steps, 675/720 observations, 33 rewards, total forces, reset and export")
        finally:
            env.close()


try:
    cfg = check_configuration()
    check_low_speed_start(cfg)
    check_actions_and_motor()
    check_reward_math()
    check_actuator_fast_path()
    if args.simulate:
        check_simulation(cfg)
    args.report.write_text(json.dumps({"ok": True, "checks": RESULTS, "simulated": args.simulate, "benchmark": BENCHMARK}, indent=2))
except BaseException:
    args.report.write_text(json.dumps({"ok": False, "checks": RESULTS, "error": traceback.format_exc()}, indent=2))
    traceback.print_exc()
    raise
finally:
    app.close()
