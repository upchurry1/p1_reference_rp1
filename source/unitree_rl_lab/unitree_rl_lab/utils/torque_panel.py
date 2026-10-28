"""A low-rate joint torque display for graphical policy playback."""

import time

import torch


class JointTorquePanel:
    """Show the first robot's applied and unclipped actuator torques in joint order."""

    def __init__(self, robot):
        import omni.ui as ui

        self.robot = robot
        self.last_update_time = float("-inf")
        self.rows = []
        self.window = ui.Window(
            "Joint Torques (PD raw / Applied)", width=680, height=460,
            flags=ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
        )
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Robot 0 | Joint torques in N m | Refresh: 10 Hz", height=22)
                ui.Label("Applied: after motor limits. PD raw: before motor limits.", height=22)
                with ui.HStack(height=24):
                    ui.Label("Joint", width=260)
                    ui.Label("PD raw (N m)", width=180)
                    ui.Label("Applied (N m)", width=180)
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=2):
                        for name in robot.joint_names:
                            with ui.HStack(height=24):
                                ui.Label(name, width=260)
                                # Explicit widths are needed inside the scrolling
                                # frame: an automatic-width trailing label can
                                # receive no visible space despite updated text.
                                computed_label = ui.Label("--", width=180)
                                applied_label = ui.Label("--", width=180)
                            self.rows.append((applied_label, computed_label))

    def update(self):
        now = time.monotonic()
        if not self.window.visible or now - self.last_update_time < 0.1:
            return
        # Copy only robot 0, once per display refresh rather than every physics
        # step. Keep signs and use the articulation's actual joint ordering.
        data = self.robot.data
        torques = torch.stack((data.applied_torque[0], data.computed_torque[0]), dim=-1)
        values = torques.detach().cpu().tolist()
        for (applied_label, computed_label), (applied, computed) in zip(self.rows, values, strict=True):
            applied_label.text = f"{applied:+.2f}"
            computed_label.text = f"{computed:+.2f}"
        self.last_update_time = now

    def close(self):
        self.window.visible = False
