---
name: franka-cube-pick-domain
description: Task-specific design decisions for the Franka cube pick task — geometry, reachability, reward function, and observation design. PENDING user validation.
level: 3
status: draft
depends_on: [isaaclab-task-structure]
extends: null
---

## Preconditions

- Task skeleton created and importable (skill 03 complete)
- Physics engine confirmed as Newton (`feature/newton` branch)
- User has validated the proposed positions and reward structure (see Variables below)

## Context

### Task Description

Franka Panda mounted at the world origin [0, 0, 0] on a flat ground plane (z=0).
A rigid cube (~4cm, scaled DexCube asset) is randomly spawned on the ground at the
start of each episode. The cube's XY position covers both reachable and unreachable
zones uniformly.

The robot must learn two behaviours based on the cube's position:
- **Reachable**: pick up the cube and transport it to the success EE position
- **Unreachable**: move the EE to the signal position (communicates "I cannot reach")

The robot is NOT told which branch applies — it must infer reachability from the cube
position observation and its own joint configuration.

### Reachability Model

Reachability is modelled as a horizontal distance check from the robot base:

```
reachable = (R_MIN <= ||cube_xy - robot_xy||₂ <= R_MAX)
```

This is a simplification — the true workspace envelope of the Franka at ground level
is not a perfect annulus, but it is a good first approximation for validation.
Refine the model if validation reveals misclassified cases.

### Reward Architecture

Two branches gated by a float mask (1.0 = reachable, 0.0 = unreachable):

**Reachable branch:**
- `approach_cube_reachable(std=0.1)` × 1.0 — EE approaches cube via tanh kernel
- `lift_cube_reachable(lift_height=0.05)` × 10.0 — binary: cube above 5cm
- `cube_at_success_position(std=0.1, lift_height=0.05)` × 15.0 — cube near success EE pos while lifted

**Unreachable branch:**
- `go_to_signal_position(std=0.1)` × 10.0 — EE moves to signal position via tanh kernel

**Penalties (always):**
- `action_rate_l2` × -1e-4
- `joint_vel_l2` × -1e-4

### Why this reward structure

- Branches are mutually exclusive (gated mask) → gradients never mix across branches
- Lift is binary, not tanh, because the transition from floor to lifted is qualitative
- `cube_at_success` is conditional on `lifted` to prevent the robot earning success reward
  without picking up the cube
- Signal position reward uses the same tanh kernel as approach for consistency
- Penalties are standard from the lift task — will be escalated by curriculum if needed

## Variables — PENDING VALIDATION

These values are embedded in `FrankaCubePickEnvCfg` and read by reward/event functions.
Confirm each before running validation tools.

| Variable | Proposed value | What it controls | Notes |
|---|---|---|---|
| `reachable_radius_min` | 0.22 m | Min horizontal dist for ground-level grasp | Franka can't grasp too close to base |
| `reachable_radius_max` | 0.65 m | Max horizontal dist for ground-level grasp | Conservative — true max is ~0.85m but ground adds constraint |
| `success_ee_position` | [0.5, 0.0, 0.5] | EE + cube target when task complete | In front of robot, mid-height |
| `signal_ee_position` | [0.0, 0.0, 0.8] | EE target for unreachable signal | Arm pointing straight up |
| Cube spawn X | [0.0, 0.8] | Forward reach of spawn zone | Covers full reachable+unreachable range |
| Cube spawn Y | [-0.6, 0.6] | Lateral reach of spawn zone | Covers full lateral range |
| `lift_height` | 0.5 m | Minimum cube Z to count as "lifted" | Mid-Franka height (~1.0m robot) — validated |
| `episode_length_s` | 8.0 s | Episode length | Longer than lift task — needs pick+transport |

## Steps

1. **Review and confirm each value in the Variables table** with the user before running simulations.

2. **Update `FrankaCubePickEnvCfg`** in `franka_cube_pick_env_cfg.py` with confirmed values.

3. **Update `EventCfg.reset_cube_position`** pose_range with confirmed spawn bounds.

4. **Run the reward validation tool** (skill 05) with a scripted success sequence and verify
   reward signals match expected branches before proceeding to RL.

## Verification

After validation tool confirms correct reward signals:
```bash
# Successful reachable sequence total reward should be >> 0
# Successful unreachable sequence total reward should be >> 0
# Failed reachable sequence (no pick) total reward should be ~0
# Failed unreachable sequence (wrong signal pos) total reward should be ~0
```
See skill 05 for the actual validation tool.

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Cube spawns inside robot | Init pos overlaps with robot base at low X | Raise cube spawn X min to 0.15 |
| Reachable cube marked unreachable in rewards | R_MAX too conservative | Increase `reachable_radius_max` and revalidate |
| Signal position causes self-collision | [0,0,0.8] is not collision-free for all joint configs | Test the config in sim; adjust signal position |
| Reward branch flips mid-episode | Cube moves after contact during pick attempt | Add a "cube locked" flag or increase episode check frequency |

### Observation Space (dexsuite-style, Newton-compatible)

47 dimensions total — reads body state directly from articulation, no FrameTransformer:

| Term | Dims | Source |
|---|---|---|
| `cube_pos` | 3 | `cube.data.root_pos_w` → robot root frame |
| `cube_quat` | 4 | `cube.data.root_quat_w` → robot root frame |
| `ee_state` | 13 | `robot.data.body_link_state_w[:, panda_hand, :]` → robot root frame (pos+quat+linvel+angvel) |
| `joint_pos` | 9 | `mdp.joint_pos_rel` (7 arm + 2 finger) |
| `joint_vel` | 9 | `mdp.joint_vel_rel` |
| `actions` | 9 | `mdp.last_action` |

**Why this pattern** (from dexsuite analysis):
- `body_link_state_w` exists on both PhysX and Newton — no sensor required.
- `subtract_frame_transforms` + `quat_rotate_inverse` handle world→robot-frame transformation.
- Velocities are expressed in robot root frame using `quat_rotate_inverse(robot_quat_w, vel_w)`.
- This mirrors the Kuka dexsuite task implementation on the `nicolasm/isaac-dexsuite-3dg-kuka-experimental-tasks` branch.

**EE link**: `panda_hand` (not the fingertip). No offset applied — the link origin is close enough for rewards and observations. Exact EE tip offset (0.1034m along Z) was used in the old FrameTransformer approach; dropped for Newton compatibility.

### Newton Architecture Notes

- Physics engine: `NewtonCfg` + `MJWarpSolverCfg` (replace `PhysxCfg`).
- Quaternion convention: `xyzw` in Newton (vs `wxyz` in PhysX). The IsaacLab math utilities handle this internally — observation code uses standard IsaacLab `subtract_frame_transforms` and `quat_rotate_inverse` which are convention-aware.
- DifferentialIK not available on Newton — validation scripts (generate_sequences.py, replay_sequences.py) are PhysX-only until a Newton-compatible IK is found.
- JointPositionActionCfg and BinaryJointPositionActionCfg work on Newton (proven by `Isaac-Reach-Franka-v0` on feature/newton).

## Changelog

- 2026-04-08: initial version — positions and rewards PROPOSED, awaiting user validation
- 2026-04-08: Added Newton architecture notes, dexsuite-style observation spec, removed FrameTransformer references.
