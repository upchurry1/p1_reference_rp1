"""Evaluate P1 speed tracking and real foot lift/landing counts at fixed commands."""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/unitree_rl_lab"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--speeds", type=float, nargs="+", default=[0., .1, .2, .3, .4, .5])
parser.add_argument("--duration", type=float, default=12.)
parser.add_argument("--warmup", type=float, default=2.)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--report", type=Path, default=Path("/tmp/p1_speed_eval.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist")
if args.repeats < 1 or not all(math.isfinite(x) for x in [*args.speeds, args.duration, args.warmup]):
    parser.error("Speeds and durations must be finite; repeats must be positive")
if args.warmup < 0 or args.duration <= args.warmup + 1:
    parser.error("Duration must exceed warmup by more than one second")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.foot_trajectory_rewards import _sole_clearance
from unitree_rl_lab.utils.keyboard_velocity_command import KeyboardVelocityCommand
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_runner import make_on_policy_runner
from isaaclab.managers import SceneEntityCfg

env = None
try:
    task = "UESTC-P1-Reference_RP1"
    count = len(args.speeds) * args.repeats
    cfg = parse_env_cfg(task, device=args.device, num_envs=count, entry_point_key="play_env_cfg_entry_point")
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.commands.base_velocity.class_type = KeyboardVelocityCommand
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.heading = None
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(cfg.commands.base_velocity.ranges, axis, (0., 0.))
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.)
    with tempfile.TemporaryDirectory(prefix="p1_speed_eval_") as cache:
        cfg.scene.robot.spawn.usd_dir = cache
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=1.0)
        base = env.unwrapped
        robot = base.scene["robot"]
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        runner = make_on_policy_runner(env, agent.to_dict(), log_dir=None, device=base.device)
        runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        command = base.command_manager.get_term("base_velocity")
        command.enable_manual_control()
        targets = torch.zeros(count, 3, device=base.device)
        targets[:, 0] = torch.tensor(args.speeds, device=base.device).repeat_interleave(args.repeats)
        command.vel_command_b.copy_(targets)
        command.is_standing_env[:] = targets.norm(dim=1) < 1e-6
        obs = env.get_observations()
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        body_ids, _ = robot.find_bodies(feet, preserve_order=True)
        sensor = base.scene.sensors["contact_forces"]
        sensor_ids, _ = sensor.find_bodies(feet, preserve_order=True)
        entity = SceneEntityCfg("robot", body_ids=body_ids)
        sums = torch.zeros(count, 3, device=base.device)
        steps = torch.zeros(count, 2, device=base.device)
        airborne_samples = torch.zeros_like(steps)
        air_time, peak = torch.zeros_like(steps), torch.zeros_like(steps)
        resets = torch.zeros(count, device=base.device)
        with torch.inference_mode():
            for _ in range(math.ceil(args.duration / base.step_dt)):
                command.vel_command_b.copy_(targets)
                command.is_standing_env[:] = targets.norm(dim=1) < 1e-6
                obs, _, done, _ = env.step(policy(obs))
                done = done.bool().flatten()
                resets += done
                height = _sole_clearance(base, entity)
                contact = sensor.data.net_forces_w[:, sensor_ids, 2] > 5.
                air = ~contact & contact.flip(1) & (height > .003)
                landed = contact & (air_time >= .06 - 1e-6) & (peak >= .01)
                # Count only uninterrupted post-warmup samples and foot cycles.
                active = ~done & (base.episode_length_buf * base.step_dt >= args.warmup)
                velocity = robot.data.root_lin_vel_b[:, 0]
                sums[:, 0] += velocity * active
                sums[:, 1] += (velocity - targets[:, 0]).abs() * active
                sums[:, 2] += active
                steps += landed & active[:, None]
                airborne_samples += air & active[:, None]
                air_time = torch.where(air, air_time + base.step_dt, 0.)
                peak = torch.where(air, torch.maximum(peak, height), 0.)
                air_time.masked_fill_(done[:, None], 0.)
                peak.masked_fill_(done[:, None], 0.)
        values = torch.cat((sums, steps, airborne_samples, resets[:, None]), dim=1).cpu().tolist()
        rows = []
        for i, speed in enumerate(args.speeds):
            group = values[i * args.repeats:(i + 1) * args.repeats]
            totals = [sum(column) for column in zip(*group)]
            samples = totals[2]
            rows.append(dict(command_vx=speed, mean_vx=totals[0] / max(samples, 1),
                mean_abs_error=totals[1] / max(samples, 1), valid_samples=int(samples),
                left_landings=int(totals[3]), right_landings=int(totals[4]),
                left_air_fraction=totals[5] / max(samples, 1),
                right_air_fraction=totals[6] / max(samples, 1), resets=int(totals[7])))
        result = dict(ok=True, checkpoint=str(args.checkpoint.resolve()),
                      duration_s=args.duration, repeats=args.repeats, results=rows)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        env.close()
        env = None
except BaseException:
    traceback.print_exc()
    args.report.write_text(json.dumps({"ok": False, "error": traceback.format_exc()}, indent=2))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
