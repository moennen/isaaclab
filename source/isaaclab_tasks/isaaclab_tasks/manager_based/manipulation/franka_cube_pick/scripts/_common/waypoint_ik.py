"""IK waypoint state machine for scripted sequence generation.

Generates per-env EE target poses and gripper states based on scenario labels.
Does not touch simulation — purely computes targets given simulation time.

Scenario waypoints
------------------
All positions are in the robot root frame (== world frame since robot is at origin).
Quaternions are [w, x, y, z].

  GRASP_QUAT   = [0.0, 1.0, 0.0, 0.0]  — gripper pointing down
  NEUTRAL_QUAT = [1.0, 0.0, 0.0, 0.0]  — gripper neutral (for signal pose)

Reachable + Success (happy path):
  t∈[0, T_APPROACH):  HOME → PRE_GRASP (above cube, gripper open)
  t∈[T_APPROACH, T_DWELL): hold PRE_GRASP (PD converges in XY)
  t∈[T_DWELL, T_GRASP): PRE_GRASP → GRASP (descent, fingers OPEN — no knock)
  t∈[T_GRASP, T_GRIP): hold GRASP height, fingers CLOSE (2.5 s to fully close)
  t∈[T_GRIP, T_LIFT): GRASP → LIFT (ease-in α² — slow lift off)
  t∈[T_LIFT, end): LIFT → SUCCESS (success_ee_pos, gripper closed)

Reachable + Failure (realistic attempts that fail — all stay below lift_height=0.45m
  so the reward stays LOW and classification is unambiguous):

  "approach_no_grip"  (20%): Full correct trajectory (approach → descend → lift pose →
    transit to success_pos) but gripper stays OPEN throughout.  Robot ends at success_pos
    holding nothing.  Simulates: correct motion but gripper hardware failure.
    → reward LOW: cube never lifted (stays on floor)

  "stop_at_pregrasp"  (20%): Approaches correctly above the cube (pre_grasp position),
    then stops and holds there.  Never descends to grasp height.  Robot is in correct
    XY above cube but never contacts it.
    → reward LOW: cube never contacted

  "grip_drop_early"  (20%): Full approach → grip → begin lifting → open gripper at a
    random time T_GRIP + 0.2 to T_GRIP + 1.0s (= 7.2–8.0s), well before the cube can
    reach lift_height.  Cube is lifted a few cm then falls.
    → reward LOW: cube_z never exceeds 0.45m (lift EE target reaches 0.45m only at ~8.2s)

  "wrong_approach_target"  (20%): Robot goes to a random reachable position that is far
    from the cube — never contacts it — then holds that position.
    → reward LOW: cube never contacted

  "descend_open_grip"  (20%): Full correct descent to grasp height, holds there with
    gripper OPEN (touches or nearly touches cube but no friction grip), then retreats
    to a neutral position.  Simulates a mis-timed or mis-sized grasp.
    → reward LOW: cube is at most displaced slightly, never lifted

Unreachable + Success (robot correctly goes to signal position):
  Optional hesitation (50%): first move toward cube for 1–2.5 s, then redirect.
  t∈[0, T_APPROACH):  HOME → SIGNAL_POS (with optional cube-direction detour)
  rest:                hold SIGNAL_POS

Unreachable + Failure (robot tries to reach cube — which it can't get to):
  t∈[0, T_APPROACH):  HOME → attempted approach above unreachable cube
  rest:                hold that position
"""

from __future__ import annotations

import math
import random

import torch


# ---------------------------------------------------------------------------
# Timing (seconds, matches episode_length_s = 10.0)
# ---------------------------------------------------------------------------

T_APPROACH   = 2.0   # end of home→pre-grasp phase
T_DWELL      = 3.0   # end of dwell at pre-grasp (1 s: PD converges in XY before descent)
T_GRASP      = 4.5   # end of pre-grasp→grasp descent (fingers OPEN throughout)
T_GRIP       = 7.0   # end of grasp-height dwell (2.5 s for fingers to fully close against cube)
                     # Data shows ke_finger=100 with coupled arm inertia takes ~2.5 s to close
                     # 40 mm against the cube; starting lift before full closure drops the cube.
T_LIFT       = 8.5   # end of grasp→lift phase (1.5 s to reach lift_ee_target_z=0.65 m)
# T_SUCCESS  = 10.0  # episode end (1.5 s at success/signal position)

# Safety margin: EE target reaches lift_height (0.45m) at ~t=8.19s (from ease-in geometry).
# Drop before this time to guarantee cube never reaches lift_height → reward stays LOW.
_GRIP_DROP_MAX_T = T_GRIP + 1.0   # = 8.0s — safe upper bound

GRIPPER_OPEN  = 0.04   # metres per finger
GRIPPER_CLOSE = 0.0    # metres per finger

GRASP_QUAT   = torch.tensor([0.0, 1.0, 0.0, 0.0])  # pointing down
NEUTRAL_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])  # neutral


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slerp_pos(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear interpolation between two (3,) positions."""
    alpha = max(0.0, min(1.0, alpha))
    return (1.0 - alpha) * a + alpha * b


def _phase_alpha(t: float, t_start: float, t_end: float) -> float:
    """Normalised progress [0, 1] within [t_start, t_end]."""
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    return (t - t_start) / (t_end - t_start)


# ---------------------------------------------------------------------------
# Per-env waypoint state
# ---------------------------------------------------------------------------


class WaypointStateMachine:
    """Per-env IK target generator.

    Args:
        cube_pos_b:     (3,) cube initial position in robot body frame
        label:          dict with keys 'reachable' (bool) and 'success' (bool)
        cfg:            geometry config dict (from sequence_schema.DEFAULT_CONFIG)
        device:         torch device
    """

    def __init__(self, cube_pos_b: torch.Tensor, label: dict, cfg: dict, device: torch.device):
        self.label = label
        self.device = device
        self.cfg = cfg

        self.cube_pos = cube_pos_b.to(device)
        self.success_pos = torch.tensor(cfg["success_ee_position"], device=device, dtype=torch.float32)
        self.signal_pos  = torch.tensor(cfg["signal_ee_position"],  device=device, dtype=torch.float32)
        self.approach_h      = cfg["grasp_approach_height"]
        self.lift_height     = cfg["lift_height"]          # reward threshold z [m]
        self.lift_ee_target_z = cfg.get("lift_ee_target_z", self.lift_height + 0.1)
        self.grasp_quat  = GRASP_QUAT.to(device)
        self.neutral_quat = NEUTRAL_QUAT.to(device)

        # Home position: slightly in front of robot base, mid-height
        self.home_pos = torch.tensor([0.3, 0.0, 0.5], device=device, dtype=torch.float32)

        # Pre-grasp: directly above cube
        self.pre_grasp_pos = self.cube_pos.clone()
        self.pre_grasp_pos[2] = self.cube_pos[2] + self.approach_h

        # Grasp EE target: panda_hand centre stops grasp_ee_height above cube centre
        grasp_ee_height = cfg.get("grasp_ee_height", 0.08)
        self.grasp_pos = self.cube_pos.clone()
        self.grasp_pos[2] = self.cube_pos[2] + grasp_ee_height

        # Lift target
        self._lift_pos = self.cube_pos.clone()
        self._lift_pos[2] = self.lift_ee_target_z

        # ---- Reachable failure mode (chosen randomly at init) ----------------
        # Five modes — all keep cube_z below lift_height=0.45m → reward stays LOW.
        _modes = [
            "approach_no_grip",      # correct motion, gripper never closes
            "stop_at_pregrasp",      # stops above cube, never descends
            "grip_drop_early",       # grips and briefly lifts, drops before 0.45m
            "wrong_approach_target", # goes to random position far from cube
            "descend_open_grip",     # descends to grasp height but gripper stays open
        ]
        self._failure_mode = random.choice(_modes)

        if self._failure_mode == "grip_drop_early":
            # Drop time: T_GRIP + 0.2 to T_GRIP + 1.0s = 7.2–8.0s.
            # EE target reaches 0.45m at ~8.19s (ease-in geometry) so this is safely below.
            self._drop_time = random.uniform(T_GRIP + 0.2, _GRIP_DROP_MAX_T)

        elif self._failure_mode == "wrong_approach_target":
            # Random reachable point not near cube, moderate height
            r_min = cfg.get("reachable_radius_min", 0.22)
            r_max = cfg.get("reachable_radius_max", 0.65)
            angle = random.uniform(0.0, 2.0 * math.pi)
            dist  = random.uniform(r_min, r_max)
            height = random.uniform(0.2, 0.45)
            self._wrong_target = torch.tensor(
                [dist * math.cos(angle), dist * math.sin(angle), height],
                device=device,
                dtype=torch.float32,
            )

        # Neutral retreat target for descend_open_grip (back toward home, mid-height)
        self._retreat_pos = self.home_pos.clone()

        # ---- Unreachable success hesitation (50% of sequences) ---------------
        if random.random() < 0.5:
            self._hesitate_end = random.uniform(1.0, 2.5)
        else:
            self._hesitate_end = 0.0  # no hesitation

    def get_target(self, t: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (ee_pos_b (3,), ee_quat_wxyz (4,), finger_pos (float))."""
        reachable = self.label["reachable"]
        success   = self.label["success"]

        if reachable and success:
            return self._reachable_success(t)
        elif reachable and not success:
            return self._reachable_failure(t)
        elif not reachable and success:
            return self._unreachable_success(t)
        else:
            return self._unreachable_failure(t)

    # ---- shared helper: success-style approach+grasp+lift -------------------

    def _approach_grasp_lift(self, t: float) -> tuple[torch.Tensor, float]:
        """Full pick sub-trajectory: approach → dwell → descent → grip dwell → lift → transit.

        Shared by _reachable_success and failure modes that start with a realistic pick.
        """
        if t < T_APPROACH:
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
            finger = GRIPPER_OPEN
        elif t < T_DWELL:
            pos = self.pre_grasp_pos
            finger = GRIPPER_OPEN
        elif t < T_GRASP:
            alpha = _phase_alpha(t, T_DWELL, T_GRASP)
            pos = _slerp_pos(self.pre_grasp_pos, self.grasp_pos, alpha)
            finger = GRIPPER_OPEN
        elif t < T_GRIP:
            pos = self.grasp_pos
            finger = GRIPPER_CLOSE
        elif t < T_LIFT:
            alpha = _phase_alpha(t, T_GRIP, T_LIFT)
            slow_alpha = alpha * alpha   # ease-in
            pos = _slerp_pos(self.grasp_pos, self._lift_pos, slow_alpha)
            finger = GRIPPER_CLOSE
        else:
            alpha = _phase_alpha(t, T_LIFT, 10.0)
            pos = _slerp_pos(self._lift_pos, self.success_pos, alpha)
            finger = GRIPPER_CLOSE
        return pos, finger

    # ---- scenario implementations -------------------------------------------

    def _reachable_success(self, t: float):
        """Full pick: approach → dwell → descent → grip → lift → success."""
        pos, finger = self._approach_grasp_lift(t)
        return pos, self.grasp_quat, finger

    def _reachable_failure(self, t: float):
        """Realistic failure — cube never lifted above 0.45m → reward stays LOW."""

        if self._failure_mode == "approach_no_grip":
            # Full correct trajectory but gripper always open.
            # Robot ends at success_pos, cube stays on floor.
            pos, _ = self._approach_grasp_lift(t)
            return pos, self.grasp_quat, GRIPPER_OPEN

        elif self._failure_mode == "stop_at_pregrasp":
            # Approaches above the cube, then stops. Never descends.
            if t < T_APPROACH:
                alpha = _phase_alpha(t, 0.0, T_APPROACH)
                pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
            else:
                pos = self.pre_grasp_pos
            return pos, self.grasp_quat, GRIPPER_OPEN

        elif self._failure_mode == "grip_drop_early":
            # Full approach + grip + brief lift, then opens gripper at _drop_time.
            # _drop_time is capped at 8.0s so cube never reaches 0.45m.
            pos, finger = self._approach_grasp_lift(t)
            if t >= self._drop_time and t >= T_GRIP:
                finger = GRIPPER_OPEN
            # After drop, keep EE at current lift position (don't chase success_pos).
            # Override: hold grasp_pos after drop to emphasise this is a failure.
            if t >= self._drop_time and t >= T_LIFT:
                pos = self.grasp_pos   # EE retreats to grasp height
            return pos, self.grasp_quat, finger

        elif self._failure_mode == "wrong_approach_target":
            # Goes to wrong position entirely — never contacts cube.
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, self._wrong_target, alpha)
            return pos, self.neutral_quat, GRIPPER_OPEN

        else:  # "descend_open_grip"
            # Descends to grasp height with gripper open (cube touched but not gripped),
            # holds there until T_GRIP, then retreats to home.
            if t < T_APPROACH:
                alpha = _phase_alpha(t, 0.0, T_APPROACH)
                pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
            elif t < T_DWELL:
                pos = self.pre_grasp_pos
            elif t < T_GRASP:
                alpha = _phase_alpha(t, T_DWELL, T_GRASP)
                pos = _slerp_pos(self.pre_grasp_pos, self.grasp_pos, alpha)
            elif t < T_GRIP:
                pos = self.grasp_pos   # hold at grasp height, gripper open
            else:
                alpha = _phase_alpha(t, T_GRIP, 10.0)
                pos = _slerp_pos(self.grasp_pos, self._retreat_pos, alpha)
            return pos, self.grasp_quat, GRIPPER_OPEN

    def _unreachable_success(self, t: float):
        """Unreachable cube, robot correctly goes to signal position."""
        if self._hesitate_end > 0.0 and t < self._hesitate_end:
            alpha = _phase_alpha(t, 0.0, self._hesitate_end)
            pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
        else:
            t_redirect_start = self._hesitate_end if self._hesitate_end > 0.0 else 0.0
            alpha = _phase_alpha(t, t_redirect_start, t_redirect_start + T_APPROACH)
            if self._hesitate_end > 0.0:
                start_pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, 1.0)
            else:
                start_pos = self.home_pos
            pos = _slerp_pos(start_pos, self.signal_pos, alpha)
        return pos, self.neutral_quat, GRIPPER_OPEN

    def _unreachable_failure(self, t: float):
        """Unreachable cube, robot wrongly tries to approach it."""
        alpha = _phase_alpha(t, 0.0, T_APPROACH)
        pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
        return pos, self.grasp_quat, GRIPPER_OPEN


# ---------------------------------------------------------------------------
# Batch interface: update all envs at once
# ---------------------------------------------------------------------------


def get_ik_commands(
    state_machines: list[WaypointStateMachine],
    t: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IK command tensor and gripper target tensor for all envs.

    Returns:
        ik_commands: (N, 7) — [pos_x, pos_y, pos_z, qw, qx, qy, qz] in robot body frame
        gripper_pos: (N, 2) — finger joint position targets [left, right]
    """
    N = len(state_machines)
    ik_commands = torch.zeros(N, 7, device=device)
    gripper_pos  = torch.zeros(N, 2, device=device)

    for i, sm in enumerate(state_machines):
        pos, quat, finger = sm.get_target(t)
        ik_commands[i, :3]  = pos
        ik_commands[i, 3:]  = quat
        gripper_pos[i, :]   = finger  # both fingers same

    return ik_commands, gripper_pos
