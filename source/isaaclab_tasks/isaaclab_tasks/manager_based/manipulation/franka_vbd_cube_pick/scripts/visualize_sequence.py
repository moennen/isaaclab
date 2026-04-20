# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool 4 — Sequence Visualizer (Newton ViewerGL).

Replays recorded VBD sequences in the Newton interactive viewer with an
interactive sidebar panel that lets you:
  - Filter sequences by label (all / reachable_success / reachable_failure /
    unreachable_success / unreachable_failure)
  - Select any sequence from the filtered list and play it immediately
  - Adjust playback speed with a slider

Viewer displays:
  - The Franka arm following recorded joint angles (via Newton FK)
  - Cube position — orange box (driven by recorded particle CoM)
  - Success EE target — green sphere
  - Signal EE target — blue sphere
  - EE position — yellow sphere
  - Per-frame reward stream printed to stdout

Reward computation uses the same reward_utils.py as the RL environment,
so what you see in the viewer is exactly what the reward function sees.

Usage
-----
python scripts/visualize_sequence.py \\
    --input  data/validation/vbd_sequences.json \\
    [--sequence_id seq_0003]   # optional: jump to this sequence on launch
    [--speed 1.0]              # playback speed multiplier
    [--viewer gl|null]         # gl = interactive, null = headless test
    [--record]                 # dump per-frame FK state to record_<seq_id>.json

Standard Newton viewer controls (ViewerGL):
    Space   — pause / resume
    R       — restart sequence from frame 0
    W/A/S/D / mouse drag — orbit camera
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
import newton.examples as newton_examples

sys.path.insert(0, str(Path(__file__).parent))
from _common.reward_eval import REWARD_WEIGHTS, compute_all_rewards
from _common.sequence_schema import DEFAULT_CONFIG, label_description, load_sequences

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---------------------------------------------------------------------------
# Panda URDF
# ---------------------------------------------------------------------------

_FRANKA_URDF_CANDIDATES = [
    Path(
        "/home/horde/micromamba/envs/env_isaaclab/lib/python3.11/site-packages/"
        "isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/"
        "franka_description/robots/panda_arm_hand.urdf"
    ),
]

_PANDA_HAND_BODY_LABEL = "panda_hand"


def _find_panda_urdf() -> Path:
    for p in _FRANKA_URDF_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Panda URDF not found.")


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

_CUBE_COLOR    = (1.0, 0.5, 0.0)   # orange
_SUCCESS_COLOR = (0.1, 0.9, 0.1)   # green
_SIGNAL_COLOR  = (0.2, 0.4, 1.0)   # blue
_EE_COLOR      = (1.0, 1.0, 0.1)   # yellow

_ALL_LABEL     = "(all labels)"
_LABEL_DISPLAY = {
    "reachable_success":   "reachable + success",
    "reachable_failure":   "reachable + failure",
    "unreachable_success": "unreachable + success",
    "unreachable_failure": "unreachable + failure",
}


def _vec3_array(*xyzs) -> wp.array:
    return wp.array([wp.vec3(*xyz) for xyz in xyzs], dtype=wp.vec3)


def _radii_array(*vals) -> wp.array:
    return wp.array(list(vals), dtype=wp.float32)


# ---------------------------------------------------------------------------
# Example class
# ---------------------------------------------------------------------------


class SequenceVisualizer:
    """Newton example class — implements step() and render() for the main loop."""

    def __init__(self, viewer, sequences: list, cfg: dict, model, hand_idx: int,
                 cube_body_idx: int = -1,
                 speed: float = 1.0, initial_id: str | None = None, record: bool = False,
                 record_dir: Path | None = None):
        self.viewer          = viewer
        self._all            = sequences
        self.cfg             = cfg
        self.model           = model
        self.hand_idx        = hand_idx
        self.cube_body_idx   = cube_body_idx
        self.speed           = speed
        self._record         = record
        self._record_dir     = record_dir or _OUTPUTS_DIR
        self._record_frames: list = []
        self._n_robot_joints = 9

        # Build ordered list of unique label strings
        seen = {}
        for s in sequences:
            key = label_description(s["label"])
            seen[key] = True
        self._unique_labels = list(seen.keys())

        self._filter_items = [_ALL_LABEL] + [
            _LABEL_DISPLAY.get(l, l) for l in self._unique_labels
        ]
        self._filter_idx   = 0
        self._filtered     = sequences[:]

        self._selected_idx = 0
        if initial_id is not None:
            for i, s in enumerate(self._filtered):
                if s["id"] == initial_id:
                    self._selected_idx = i
                    break

        self.state    = model.state()
        self.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)

        self.device      = torch.device("cpu")
        self.success_pos = torch.tensor(cfg["success_ee_position"], dtype=torch.float32).unsqueeze(0)
        self.signal_pos  = torch.tensor(cfg["signal_ee_position"],  dtype=torch.float32).unsqueeze(0)
        self.robot_pos_w = torch.zeros(1, 3)

        self._load_sequence(self._filtered[self._selected_idx])

        viewer.set_model(model)
        viewer.set_camera(
            pos=wp.vec3(0.4, -2.0, 1.2),
            pitch=-15.0,
            yaw=95.0,
        )

        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render_ui, position="side")

    # ------------------------------------------------------------------
    # Sequence management

    def _load_sequence(self, seq: dict):
        if self._record and self._record_frames:
            self._flush_record()

        self.seq       = seq
        self.frames    = seq["frames"]
        self.sim_time  = 0.0
        self.frame_idx = 0
        self._prev_joint_q = np.array(self.frames[0]["joint_pos"], dtype=np.float32)
        self._record_frames = []
        self._apply_frame(0)

        label_str = label_description(seq["label"])
        print(f"\n[visualize] {seq['id']}  ({label_str})")
        dist = seq.get("cube_horizontal_dist", "?")
        print(f"[visualize] dist={dist:.3f}m  {len(self.frames)} frames at 50 Hz  speed={self.speed}x\n")
        print(f"{'t':>6}  {'frame':>5}  {'total':>8}  {'approach':>8}  {'grip':>6}  "
              f"{'lift':>6}  {'success':>8}  {'signal':>8}  {'mask':>4}")
        print("-" * 80)

    def _flush_record(self):
        seq_id = self.seq["id"]
        out_path = self._record_dir / f"record_{seq_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "seq_id": seq_id,
                "label":  self.seq["label"],
                "frames": self._record_frames,
            }, f, indent=2)
        print(f"\n[visualize] Record saved → {out_path}  ({len(self._record_frames)} frames)")
        self._record_frames = []

    def _apply_filter(self):
        if self._filter_idx == 0:
            self._filtered = self._all[:]
        else:
            label = self._unique_labels[self._filter_idx - 1]
            self._filtered = [s for s in self._all if label_description(s["label"]) == label]
        self._selected_idx = 0
        if self._filtered:
            self._load_sequence(self._filtered[0])

    # ------------------------------------------------------------------
    # ImGui sidebar

    def _render_ui(self, imgui):
        imgui.separator_text("Sequences")

        changed, new_idx = imgui.combo("Label", self._filter_idx, self._filter_items)
        if changed:
            self._filter_idx = new_idx
            self._apply_filter()

        imgui.text(f"{len(self._filtered)} sequences")
        imgui.spacing()

        list_height = imgui.ImVec2(-1, 220)
        if imgui.begin_list_box("##seqlist", list_height):
            for i, seq in enumerate(self._filtered):
                is_selected = (i == self._selected_idx)
                clicked, _ = imgui.selectable(seq["id"], is_selected)
                if clicked and not is_selected:
                    self._selected_idx = i
                    self._load_sequence(self._filtered[i])
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_list_box()

        imgui.spacing()
        imgui.separator_text("Playback")

        changed, new_speed = imgui.slider_float("Speed", self.speed, 0.1, 4.0, "%.1fx")
        if changed:
            self.speed = max(0.1, new_speed)

        progress = self.frame_idx / max(len(self.frames) - 1, 1)
        imgui.progress_bar(progress, imgui.ImVec2(-1, 0), f"{self.frame_idx}/{len(self.frames)}")

        imgui.spacing()
        imgui.separator_text("Current frame")
        if self.frame_idx < len(self.frames):
            t = self.frames[self.frame_idx]["t"]
            imgui.text(f"t = {t:.2f} s    frame {self.frame_idx}")
            label_str = label_description(self.seq["label"])
            imgui.text(f"label: {_LABEL_DISPLAY.get(label_str, label_str)}")

    # ------------------------------------------------------------------
    # FK helpers

    def _apply_frame(self, idx: int):
        """Set model joint_q to frame[idx] and run FK.

        joint_q layout: [9 robot joints | 3 cube xyz | 4 cube quat xyzw]
        The cube free-joint coords position the visual box at cube_pos_w (particle CoM).
        """
        if idx >= len(self.frames):
            return
        frame      = self.frames[idx]
        n          = self._n_robot_joints
        joint_q_np = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        joint_q_np[:n] = np.array(frame["joint_pos"], dtype=np.float32)
        cube_pos = frame["cube_pos_w"]
        joint_q_np[n:n+3]   = cube_pos
        joint_q_np[n+3:n+7] = [0.0, 0.0, 0.0, 1.0]   # identity quat xyzw
        joint_q = wp.array(joint_q_np, dtype=wp.float32)
        newton.eval_fk(self.model, joint_q, self.joint_qd, self.state)

    def _compute_reward(self, frame_idx: int) -> dict:
        frame      = self.frames[frame_idx]
        joint_q_np = np.array(frame["joint_pos"], dtype=np.float32)

        body_q = self.state.body_q.numpy()
        ee_pos = body_q[self.hand_idx][:3]

        cube_pos = frame["cube_pos_w"]

        # gripper_width = sum of both finger joint positions (indices 7 and 8)
        gripper_width = float(joint_q_np[7]) + float(joint_q_np[8])

        ee_pos_t      = torch.tensor(ee_pos,    dtype=torch.float32).unsqueeze(0)
        cube_pos_t    = torch.tensor(cube_pos,  dtype=torch.float32).unsqueeze(0)
        gripper_w_t   = torch.tensor([gripper_width], dtype=torch.float32)
        joint_vel_t   = torch.tensor(frame["joint_vel"], dtype=torch.float32).unsqueeze(0)
        curr_t        = torch.tensor(joint_q_np, dtype=torch.float32).unsqueeze(0)
        prev_t        = torch.tensor(self._prev_joint_q, dtype=torch.float32).unsqueeze(0)

        rews = compute_all_rewards(
            ee_pos_w=ee_pos_t, cube_pos_w=cube_pos_t, robot_pos_w=self.robot_pos_w,
            gripper_width=gripper_w_t,
            joint_vel=joint_vel_t, action_curr=curr_t, action_prev=prev_t,
            success_ee_pos=self.success_pos, signal_ee_pos=self.signal_pos,
            r_min=self.cfg["reachable_radius_min"], r_max=self.cfg["reachable_radius_max"],
            lift_height=self.cfg["lift_height"],
        )
        self._prev_joint_q = joint_q_np.copy()
        return {k: float(v[0].item()) for k, v in rews.items()}

    # ------------------------------------------------------------------
    # Newton example interface

    def step(self):
        if self.viewer.is_paused():
            return
        next_idx = self.frame_idx + self.speed
        self.frame_idx = min(int(next_idx), len(self.frames) - 1)
        self.sim_time += (1.0 / 50.0) * self.speed   # VBD sequences run at 50 Hz
        self._apply_frame(self.frame_idx)

    def render(self):
        if self.frame_idx >= len(self.frames):
            return

        frame = self.frames[self.frame_idx]
        rews  = self._compute_reward(self.frame_idx)

        print(
            f"{frame['t']:6.2f}  {self.frame_idx:5d}  "
            f"{rews['total']:+8.2f}  "
            f"{rews['approach_cube_reachable']:+8.3f}  "
            f"{rews['grip_cube_reachable']:+6.3f}  "
            f"{rews['lift_cube_reachable']:+6.2f}  "
            f"{rews['cube_at_success_position']:+8.3f}  "
            f"{rews['go_to_signal_position']:+8.3f}  "
            f"{rews['reachable_mask']:4.0f}"
        )

        if self._record:
            body_q_r  = self.state.body_q.numpy()
            ee_fk_r   = body_q_r[self.hand_idx][:3].tolist()
            ee_json_r = frame.get("ee_pos_w", ee_fk_r)
            self._record_frames.append({
                "frame_idx":      self.frame_idx,
                "t":              frame["t"],
                "joint_pos":      frame["joint_pos"],
                "ee_pos_fk":      [round(v, 6) for v in ee_fk_r],
                "ee_pos_json":    [round(v, 6) for v in ee_json_r],
                "cube_pos_w":     frame["cube_pos_w"],
                "gripper_closed": frame.get("gripper_closed", False),
                "rewards":        {k: round(rews[k], 6) for k in rews if k != "reachable_mask"},
                "reachable_mask": round(rews.get("reachable_mask", 0.0), 1),
            })

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        # Success target sphere
        self.viewer.log_points(
            "success_target",
            _vec3_array(self.cfg["success_ee_position"]),
            radii=_radii_array(0.04),
            colors=_vec3_array(_SUCCESS_COLOR),
        )

        # Signal target sphere
        self.viewer.log_points(
            "signal_target",
            _vec3_array(self.cfg["signal_ee_position"]),
            radii=_radii_array(0.04),
            colors=_vec3_array(_SIGNAL_COLOR),
        )

        # EE marker
        body_q = self.state.body_q.numpy()
        ee_xyz = body_q[self.hand_idx][:3].tolist()
        self.viewer.log_points(
            "ee",
            _vec3_array(ee_xyz),
            radii=_radii_array(0.02),
            colors=_vec3_array(_EE_COLOR),
        )

        self.viewer.end_frame()

    def test_final(self):
        if self._record and self._record_frames:
            self._flush_record()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = newton_examples.create_parser()
    parser.add_argument("--input",       type=str,
                        default=str(_OUTPUTS_DIR / "vbd_sequences.json"),
                        help="Path to sequences JSON (Tool 1 output).")
    parser.add_argument("--sequence_id", type=str, default=None,
                        help="Sequence ID to jump to on launch (optional).")
    parser.add_argument("--speed",       type=float, default=1.0,
                        help="Initial playback speed multiplier (1.0 = real-time at 50 Hz).")
    parser.add_argument("--record",      action="store_true",
                        help="Dump per-frame FK state + rewards to record_<seq_id>.json.")

    viewer, args = newton_examples.init(parser=parser)

    data      = load_sequences(args.input)
    cfg       = data["config"]
    sequences = data["sequences"]

    if not sequences:
        raise ValueError(f"No sequences found in {args.input}")

    # Build visualization-only model: robot + free-floating box for cube CoM.
    # The box tracks recorded cube_pos_w (particle CoM) so it's a rigid proxy
    # for the deformable cube. Actual physics uses VBD particles — not present here.
    urdf_path = _find_panda_urdf()
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    mb.add_urdf(str(urdf_path), floating=False)
    hand_idx = next(i for i, k in enumerate(mb.body_label) if _PANDA_HAND_BODY_LABEL in k)
    cube_body_idx = mb.add_body(mass=0.0, label="cube_vis")
    mb.add_shape_box(
        body=cube_body_idx,
        hx=0.025, hy=0.025, hz=0.025,
        label="cube_vis_shape",
    )
    model = mb.finalize()

    record_dir = Path(args.input).parent if getattr(args, "record", False) else None

    example = SequenceVisualizer(
        viewer,
        sequences,
        cfg,
        model,
        hand_idx,
        cube_body_idx=cube_body_idx,
        speed=args.speed,
        initial_id=args.sequence_id,
        record=getattr(args, "record", False),
        record_dir=record_dir,
    )
    newton_examples.run(example, args)


if __name__ == "__main__":
    main()
