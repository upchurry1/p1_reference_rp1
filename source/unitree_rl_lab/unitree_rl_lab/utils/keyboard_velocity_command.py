"""Manual velocity commands for playback of policies trained with uniform commands."""

import torch
from isaaclab.envs.mdp import UniformVelocityCommand


class KeyboardVelocityCommand(UniformVelocityCommand):
    """Keep keyboard velocities through resets and periodic command resampling."""

    def enable_manual_control(self):
        self.vel_command_b.zero_()
        self.is_heading_env.fill_(False)
        self.is_standing_env.fill_(True)

    def set_manual_target(self, velocity: tuple[float, float, float]):
        self.vel_command_b[:] = self.vel_command_b.new_tensor(velocity)
        self.is_standing_env[:] = torch.linalg.vector_norm(self.vel_command_b, dim=1) < 1.0e-6

    def _resample_command(self, env_ids):
        # CommandManager also calls this after episode resets. Never replace a
        # held key with a randomly sampled speed or heading.
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        # Keyboard input already supplies all three base-frame velocity axes.
        pass
