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
  t∈[0, T0):  HOME  → PRE_GRASP  (above cube, gripper open)
  t∈[T0, T1): PRE_GRASP → GRASP  (at grasp height above cube centre, gripper open→close)
  t∈[T1, T2): GRASP → LIFT       (cube_xy + [0, 0, lift_height], gripper closed)
  t∈[T2, end): LIFT → SUCCESS    (success_ee_pos, gripper closed)

Reachable + Failure (robot goes to wrong target — diversified failure modes):
  Three modes chosen randomly at init:
    "stop_short" (40%): approach cube but stop at a random fraction of the way
    "wrong_target" (40%): move to a random reachable point far from the cube
    "signal" (20%): current behavior (go to signal_pos)

Unreachable + Success (robot correctly goes to signal position):
  Optional hesitation (50%): first move toward cube for 1–2.5 s, then redirect to signal_pos.
  t∈[0, T0):  HOME → SIGNAL_POS (with optional cube-direction detour before redirect)
  rest:        hold SIGNAL_POS

Unreachable + Failure (robot tries to reach cube — which it can't get to):
  t∈[0, T0):  HOME → attempted approach above unreachable cube
  rest:        hold that position
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
                     # Data shows ke_finger=20 with coupled arm inertia takes ~2.5 s to close
                     # 40 mm against the cube; starting lift before full closure drops the cube.
T_LIFT       = 8.5   # end of grasp→lift phase (1.5 s to reach lift_ee_target_z=0.65 m)
# T_SUCCESS  = 10.0  # episode end (1.5 s at success/signal position)

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
        # IK targets lift_ee_target_z (> lift_height) to compensate PD tracking lag
        self.grasp_quat  = GRASP_QUAT.to(device)
        self.neutral_quat = NEUTRAL_QUAT.to(device)

        # Home position: slightly in front of robot base, mid-height
        self.home_pos = torch.tensor([0.3, 0.0, 0.5], device=device, dtype=torch.float32)

        # Pre-grasp: directly above cube
        self.pre_grasp_pos = self.cube_pos.clone()
        self.pre_grasp_pos[2] = self.cube_pos[2] + self.approach_h

        # Grasp EE target: panda_hand centre stops grasp_ee_height above cube centre
        # (fingers extend ~10 cm below panda_hand; 8 cm clearance keeps them above ground)
        grasp_ee_height = cfg.get("grasp_ee_height", 0.08)
        self.grasp_pos = self.cube_pos.clone()
        self.grasp_pos[2] = self.cube_pos[2] + grasp_ee_height

        # ---- Reachable failure mode (chosen randomly at init) ----------------
        # Three modes: "stop_short" (40%), "wrong_target" (40%), "signal" (20%)
        _mode_roll = random.random()
        if _mode_roll < 0.40:
            self._failure_mode = "stop_short"
            self._stop_fraction = random.uniform(0.5, 0.85)
        elif _mode_roll < 0.80:
            self._failure_mode = "wrong_target"
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
        else:
            self._failure_mode = "signal"

        # ---- Unreachable success hesitation (50% of sequences) ---------------
        if random.random() < 0.5:
            self._hesitate_end = random.uniform(1.0, 2.5)
        else:
            self._hesitate_end = 0.0  # no hesitation

    def get_target(self, t: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (ee_pos_b (3,), ee_quat_wxyz (4,), finger_pos (float)).

        This is called once per env per simulation step.
        """
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

    # ---- scenario implementations ----------------------------------------

    def _reachable_success(self, t: float):
        """Full pick: approach → pre-grasp dwell → descent → grasp dwell → lift → success.

        Fingers stay OPEN through the entire descent so they don't knock the cube.
        They close only during the grasp-height dwell, after the PD controller has
        had 1 s to centre the arm over the cube before gripping.
        """
        if t < T_APPROACH:
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
            quat = self.grasp_quat
            finger = GRIPPER_OPEN
        elif t < T_DWELL:
            # Hold at pre_grasp — PD arm converges in XY before descent.
            pos = self.pre_grasp_pos
            quat = self.grasp_quat
            finger = GRIPPER_OPEN
        elif t < T_GRASP:
            # Descend to grasp height with fingers OPEN — no cube contact during descent.
            alpha = _phase_alpha(t, T_DWELL, T_GRASP)
            pos = _slerp_pos(self.pre_grasp_pos, self.grasp_pos, alpha)
            quat = self.grasp_quat
            finger = GRIPPER_OPEN
        elif t < T_GRIP:
            # Hold at grasp height — PD arm converges in XY, then fingers close.
            # Closing at grasp height ensures finger inner face straddles the cube.
            pos = self.grasp_pos
            quat = self.grasp_quat
            finger = GRIPPER_CLOSE
        elif t < T_LIFT:
            alpha = _phase_alpha(t, T_GRIP, T_LIFT)
            # Ease-in (α²): very slow start lets the cube detach from the floor
            # gradually rather than receiving a large impulsive lateral force from
            # the abrupt constraint-topology change at T_GRIP.
            # At t=T_GRIP+0.08s: 1.5 mm above grasp (vs 7 mm with linear), so
            # contact forces ramp up slowly until the cube lifts off the ground.
            slow_alpha = alpha * alpha
            lift_pos = self.cube_pos.clone()
            lift_pos[2] = self.lift_ee_target_z   # aim above threshold to compensate lag
            pos = _slerp_pos(self.grasp_pos, lift_pos, slow_alpha)
            quat = self.grasp_quat
            finger = GRIPPER_CLOSE
        else:
            alpha = _phase_alpha(t, T_LIFT, 10.0)
            lift_pos = self.cube_pos.clone()
            lift_pos[2] = self.lift_ee_target_z
            pos = _slerp_pos(lift_pos, self.success_pos, alpha)
            quat = self.grasp_quat
            finger = GRIPPER_CLOSE
        return pos, quat, finger

    def _reachable_failure(self, t: float):
        """Reachable cube but robot goes to wrong target — three failure modes."""
        if self._failure_mode == "stop_short":
            # Approach cube but stop at a random fraction of the way
            intermediate = _slerp_pos(self.home_pos, self.pre_grasp_pos, self._stop_fraction)
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, intermediate, alpha)
        elif self._failure_mode == "wrong_target":
            # Move to a random reachable point far from the cube
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, self._wrong_target, alpha)
        else:
            # "signal" mode: original behavior
            alpha = _phase_alpha(t, 0.0, T_APPROACH)
            pos = _slerp_pos(self.home_pos, self.signal_pos, alpha)
        return pos, self.neutral_quat, GRIPPER_OPEN

    def _unreachable_success(self, t: float):
        """Unreachable cube, robot correctly goes to signal position.

        Optional hesitation: first move toward cube for hesitate_end seconds,
        then redirect to signal_pos.
        """
        if self._hesitate_end > 0.0 and t < self._hesitate_end:
            # Move toward cube direction during hesitation
            alpha = _phase_alpha(t, 0.0, self._hesitate_end)
            pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, alpha)
        else:
            # Redirect to signal_pos
            t_redirect_start = self._hesitate_end if self._hesitate_end > 0.0 else 0.0
            alpha = _phase_alpha(t, t_redirect_start, t_redirect_start + T_APPROACH)
            # Start from wherever we were at hesitate_end
            if self._hesitate_end > 0.0:
                hesitate_alpha = min(1.0, self._hesitate_end / self._hesitate_end)
                start_pos = _slerp_pos(self.home_pos, self.pre_grasp_pos, hesitate_alpha)
            else:
                start_pos = self.home_pos
            pos = _slerp_pos(start_pos, self.signal_pos, alpha)
        return pos, self.neutral_quat, GRIPPER_OPEN

    def _unreachable_failure(self, t: float):
        """Unreachable cube, robot wrongly tries to approach it."""
        alpha = _phase_alpha(t, 0.0, T_APPROACH)
        # Try to go to pre-grasp above the unreachable cube — robot gets stuck
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
