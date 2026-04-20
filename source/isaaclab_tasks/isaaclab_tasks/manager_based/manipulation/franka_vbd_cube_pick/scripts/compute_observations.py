# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool 4 — VBD Observation Extractor.

Unlike the rigid-task version, no re-simulation is needed: generate_sequences.py
already records the full physics state per frame (joint_pos, joint_vel, ee_pos_w,
cube_pos_w). This script simply reshapes that data into the standard 25-dim obs
vector used by analyze_observations.py.

Observation vector (25 dims)
----------------------------
  cube_pos  (3) : cube XYZ in robot root frame (particle CoM) [m]
  ee_pos    (3) : EE position in robot root frame [m]
  joint_pos (7) : arm joint positions [rad]
  finger_pos(2) : finger joint positions [m]
  joint_vel (7) : arm joint velocities [rad/s]
  finger_vel(2) : finger joint velocities [m/s]
  grip      (1) : gripper closed flag (0.0 or 1.0)

Usage
-----
micromamba run -n env_isaaclab python scripts/compute_observations.py \\
    --input  data/validation/vbd_sequences_v3.json \\
    --output data/validation/vbd_observations.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

OBS_NAMES = [
    "cube_x", "cube_y", "cube_z",
    "ee_x", "ee_y", "ee_z",
    "j0", "j1", "j2", "j3", "j4", "j5", "j6",
    "f0", "f1",
    "jv0", "jv1", "jv2", "jv3", "jv4", "jv5", "jv6",
    "fv0", "fv1",
    "grip",
]  # 25 total
assert len(OBS_NAMES) == 25


def main():
    parser = argparse.ArgumentParser(
        description="Extract 25-dim obs from VBD sequences (no simulation needed)."
    )
    parser.add_argument("--input",  type=str,
                        default=str(_OUTPUTS_DIR / "vbd_sequences_v3.json"))
    parser.add_argument("--output", type=str,
                        default=str(_OUTPUTS_DIR / "vbd_observations.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    cfg       = data["config"]
    sequences = data["sequences"]
    n_seq     = len(sequences)
    n_frames  = len(sequences[0]["frames"])
    print(f"[obs-extract] {n_seq} sequences × {n_frames} frames = {n_seq*n_frames:,} total frames")

    out_seqs = []
    for seq in sequences:
        frame_obs = []
        for fr in seq["frames"]:
            jp  = fr["joint_pos"]       # 9: arm(7) + finger(2)
            jv  = fr["joint_vel"]       # 9
            jc  = fr["joint_pos_cmd"]   # 9
            ep  = fr["ee_pos_w"]        # 3
            rp  = fr["robot_pos_w"]     # 3
            cp  = fr["cube_pos_w"]      # 3

            # Express in robot root frame (= world frame; robot fixed at origin)
            cube_b = [cp[0] - rp[0], cp[1] - rp[1], cp[2] - rp[2]]
            ee_b   = [ep[0] - rp[0], ep[1] - rp[1], ep[2] - rp[2]]
            grip   = 1.0 if fr.get("gripper_closed", False) else 0.0

            obs = cube_b + ee_b + list(jp[:7]) + list(jp[7:9]) + list(jv[:7]) + list(jv[7:9]) + [grip]
            assert len(obs) == 25

            frame_obs.append({
                "step": fr["step"],
                "t":    fr["t"],
                "obs":  [round(float(x), 7) for x in obs],
                "action": [round(float(x), 7) for x in jc],
            })

        out_seqs.append({
            "id":                   seq["id"],
            "label":                seq["label"],
            "cube_init_pos_w":      seq["cube_init_pos_w"],
            "cube_horizontal_dist": seq["cube_horizontal_dist"],
            "frames":               frame_obs,
        })

    output = {
        "version":         "1.0",
        "computed_at":     datetime.datetime.now().isoformat(),
        "source":          args.input,
        "config":          cfg,
        "obs_names":       OBS_NAMES,
        "obs_dims":        len(OBS_NAMES),
        "note":            "Extracted from physics simulation; cube_pos = particle CoM",
        "sequences":       out_seqs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[obs-extract] Saved {n_seq} sequences → {out_path}")


if __name__ == "__main__":
    main()
