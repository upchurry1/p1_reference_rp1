"""P1 keyboard speed caps, with an optional manual forward-speed limit."""

from copy import deepcopy
import math


def keyboard_speed_ranges(
    hard_limits, vx: float, vy: float, trained_ranges=None, *,
    yaw: float | None = None, vx_limit: float | None = None,
):
    if vx_limit is not None and (not math.isfinite(vx_limit) or vx_limit < 0):
        raise ValueError("Keyboard vx limit must be finite and nonnegative")
    result = deepcopy(hard_limits)
    axes = [("lin_vel_x", vx), ("lin_vel_y", vy)]
    if yaw is not None:
        axes.append(("ang_vel_z", yaw))
    for axis, speed in axes:
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Keyboard speeds must be finite and nonnegative")
        low, high = getattr(hard_limits, axis)
        if trained_ranges is not None:
            trained_low, trained_high = map(float, trained_ranges[axis])
            if not all(map(math.isfinite, (trained_low, trained_high))) or trained_low > trained_high:
                raise ValueError(f"Invalid trained command range: {axis}")
            low, high = max(low, trained_low), min(high, trained_high)
        if axis == "lin_vel_x" and vx_limit is not None:
            # Manual forward playback may exceed the checkpoint's curriculum
            # stage, while retaining the task's hard cap and backward envelope.
            speed = min(speed, vx_limit)
            high = min(hard_limits.lin_vel_x[1], vx_limit)
        # Standing is always allowed. A forward-only stage disables S rather
        # than mapping S to a small positive velocity; zero lateral ranges disable A/D.
        setattr(result, axis, (min(0.0, max(low, -speed)), max(0.0, min(high, speed))))
    return result
