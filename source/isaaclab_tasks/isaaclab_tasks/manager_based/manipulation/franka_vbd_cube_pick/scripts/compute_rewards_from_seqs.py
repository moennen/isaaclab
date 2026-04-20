# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fast reward computation directly from sequences (no re-simulation needed).

Sequences already record the full physics state per frame (ee_pos_w, cube_pos_w,
joint_pos, joint_vel). This script applies reward_utils.compute_all_rewards() to
those recorded states and produces a replay-format JSON.

Output format is identical to replay_sequences.py so analyze_results.py and
analyze_observations.py work without modification.

Usage
-----
micromamba run -n env_isaaclab python scripts/compute_rewards_from_seqs.py \\
    --input  data/validation/vbd_sequences_v3.json \\
    --output data/validation/vbd_seqrewards.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common.reward_eval import REWARD_WEIGHTS, compute_all_rewards
from _common.sequence_schema import expected_high_reward, label_description, load_sequences
from _common.waypoint_ik import T_LIFT as _T_LIFT  # 8.5 s — success gate

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

_DEVICE = torch.device("cpu")


def compute_sequence(seq: dict, cfg: dict) -> dict:
    """Compute per-frame rewards from recorded state for a single sequence."""
    r_min = cfg["reachable_radius_min"]
    r_max = cfg["reachable_radius_max"]
    lift_height = cfg["lift_height"]
    success_pos = torch.tensor([cfg["success_ee_position"]], dtype=torch.float32, device=_DEVICE)
    signal_pos  = torch.tensor([cfg["signal_ee_position"]],  dtype=torch.float32, device=_DEVICE)

    label = seq["label"]
    frames = seq["frames"]

    episode_rewards = {k: 0.0 for k in list(REWARD_WEIGHTS.keys()) + ["total"]}
    success_frame = -1
    variance_accum = []

    frame_results = []
    prev_cmd = None

    for fr in frames:
        ee_pos  = torch.tensor([fr["ee_pos_w"]],   dtype=torch.float32, device=_DEVICE)
        cube_pos = torch.tensor([fr["cube_pos_w"]], dtype=torch.float32, device=_DEVICE)
        robot_pos = torch.tensor([fr["robot_pos_w"]], dtype=torch.float32, device=_DEVICE)
        jp = fr["joint_pos"]
        jv = fr["joint_vel"]
        jc = fr["joint_pos_cmd"]

        gripper_width = torch.tensor([jp[7] + jp[8]], dtype=torch.float32, device=_DEVICE)
        joint_vel_t   = torch.tensor([jv], dtype=torch.float32, device=_DEVICE)
        action_curr   = torch.tensor([jc], dtype=torch.float32, device=_DEVICE)
        action_prev   = torch.tensor([prev_cmd if prev_cmd is not None else jc],
                                     dtype=torch.float32, device=_DEVICE)

        rews = compute_all_rewards(
            ee_pos_w=ee_pos, cube_pos_w=cube_pos, robot_pos_w=robot_pos,
            gripper_width=gripper_width,
            joint_vel=joint_vel_t, action_curr=action_curr, action_prev=action_prev,
            success_ee_pos=success_pos, signal_ee_pos=signal_pos,
            r_min=r_min, r_max=r_max, lift_height=lift_height,
        )

        frame_rew = {k: float(v[0].item()) for k, v in rews.items() if k in REWARD_WEIGHTS}
        frame_total = sum(v * REWARD_WEIGHTS[k] for k, v in frame_rew.items())
        for k in episode_rewards:
            if k == "total":
                episode_rewards[k] += frame_total
            else:
                episode_rewards[k] += frame_rew.get(k, 0.0)

        t_cur = fr["t"]
        if success_frame == -1:
            if label["reachable"]:
                if t_cur >= _T_LIFT and cube_pos[0, 2].item() > lift_height:
                    success_frame = fr["step"]
            else:
                signal_dist = torch.norm(ee_pos[0] - signal_pos[0]).item()
                if signal_dist < 0.15:
                    success_frame = fr["step"]

        reach_mask = float(rews["reachable_mask"][0].item()) if "reachable_mask" in rews else 0.0

        frame_results.append({
            "step":           fr["step"],
            "t":              t_cur,
            "reachable_mask": reach_mask,
            "robot_pos_w":    fr["robot_pos_w"],
            "ee_pos_w":       fr["ee_pos_w"],
            "cube_pos_w":     fr["cube_pos_w"],
            "rewards":        {k: round(frame_rew.get(k, 0.0), 6) for k in REWARD_WEIGHTS},
        })

        prev_cmd = jc

    return {
        "id":                      seq["id"],
        "label":                   label,
        "expected_high_reward":    expected_high_reward(label),
        "episode_rewards":         {k: round(v, 4) for k, v in episode_rewards.items()},
        "success_frame":           success_frame,
        "joint_pos_variance_mean": 0.0,  # generation is the source — no replay to compare
        "frames":                  frame_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute rewards from recorded sequence state (no simulation)."
    )
    parser.add_argument("--input",  type=str,
                        default=str(_OUTPUTS_DIR / "vbd_sequences_v3.json"))
    parser.add_argument("--output", type=str,
                        default=str(_OUTPUTS_DIR / "vbd_seqrewards.json"))
    args = parser.parse_args()

    data      = load_sequences(args.input)
    cfg       = data["config"]
    sequences = data["sequences"]
    n_seq     = len(sequences)
    n_frames  = len(sequences[0]["frames"])

    print(f"[seqrewards] {n_seq} sequences × {n_frames} frames = {n_seq*n_frames:,} total frames")

    all_results = []
    ok = mismatch = 0
    for i, seq in enumerate(sequences):
        result = compute_sequence(seq, cfg)
        all_results.append(result)

        exp = "SUCCESS" if result["expected_high_reward"] else "FAILURE"
        got = "SUCCESS" if result["success_frame"] >= 0 else "FAILURE"
        match = "[OK]" if exp == got else "[MISMATCH]"
        if exp == got:
            ok += 1
        else:
            mismatch += 1
        ep = result["episode_rewards"]["total"]
        print(f"  [{seq['id']}] ep_reward={ep:+.1f}  expected={exp}  {match}"
              f"  success_frame={result['success_frame']}")

    accuracy = 100.0 * ok / n_seq
    print(f"\n[seqrewards] Accuracy: {ok}/{n_seq} OK = {accuracy:.1f}%  "
          f"({mismatch} MISMATCH)")

    output = {
        "version":       "1.0",
        "computed_at":   datetime.datetime.now().isoformat(),
        "source":        args.input,
        "note":          "Rewards computed from sequences state (no re-simulation). "
                         "joint_pos_variance_mean = 0 (single source).",
        "config":        cfg,
        "sequences":     all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[seqrewards] Saved {n_seq} sequences → {out_path}")


if __name__ == "__main__":
    main()
