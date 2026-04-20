# Franka Cube Pick — Product Requirements Document

*Bootstrapped from conversation (no initial PRD was provided). Requirements derived from chat history.*

---

## Problem Statement

Training an RL policy for manipulation tasks is error-prone because bugs in physics setup, reward functions, or observations only surface after expensive training runs. We need a validation framework that can catch these bugs *before* any policy is trained, by generating scripted sequences, replaying them in simulation, and correlating reward signals with known ground-truth labels.

The specific task is a Franka Panda robot that must pick up a rigid cube if it is reachable, or move to a designated signal position if it is not. The two-branch design (pick vs. signal) tests whether the reward function correctly handles both reachable and unreachable cases without gradient leakage between branches.

---

## Goals

1. Define a two-branch manipulation task (pick if reachable, signal if unreachable) as an Isaac Lab manager-based RL environment.
2. Build a three-tool validation toolchain that validates physics, rewards, and observations without training any policy.
3. Achieve > 90% label-reward correlation (sequence labelled "high reward" produces higher total reward than sequences labelled "low reward") when replaying scripted sequences.
4. All four scenario types (reachable+success, reachable+failure, unreachable+success, unreachable+failure) produce meaningfully distinct reward signals.
5. The validation toolchain is fully testable without Isaac Sim (unit tests run in the micromamba env).

---

## Non-Goals

- RL policy training (out of scope until validation passes).
- Table or elevated surfaces — cube rests on the ground plane only.
- Multi-object manipulation.
- Sim-to-real transfer or domain randomization.
- Visual observations (joint-position observations only for now).

---

## Constraints

- **Robot**: Franka Panda (FRANKA_PANDA_HIGH_PD_CFG for scripted sequences, FRANKA_PANDA_CFG for RL env).
- **Simulator**: Isaac Sim 5.1.0, Isaac Lab (current main), Python 3.11 via micromamba.
- **GPU**: L40 via NVIDIA DGX Cloud.
- **Task location**: `source/isaaclab_tasks_experimental/` — follows the existing experimental tasks pattern.
- **No table**: cube rests directly on the ground at z = `cube_half_height`.
- **Robot commands via DifferentialIK** for scripted sequences; joint-position actions for RL env.

---

## Requirements

### Functional Requirements

**F1 — Task environment**
- F1.1: The environment follows the Isaac Lab manager-based pattern (`ManagerBasedRLEnvCfg`).
- F1.2: A two-level config hierarchy: abstract base config + robot-specific derived config (MISSING pattern).
- F1.3: The cube spawns randomly at positions consistent with its reachability label.
- F1.4: The robot must move to `success_ee_position` after lifting the cube (reachable branch).
- F1.5: The robot must move to `signal_ee_position` when the cube is unreachable.
- F1.6: Reachability is determined geometrically (horizontal distance from robot base), not from any observation.

**F2 — Reward function**
- F2.1: Two branches gated by a float reachability mask (1.0 = reachable, 0.0 = unreachable).
- F2.2: Branch A (reachable): approach cube → lift cube → reach success position.
- F2.3: Branch B (unreachable): move EE to signal position.
- F2.4: A standalone implementation (`reward_eval.py`) takes plain tensors — no env object.
- F2.5: The env-integrated version (`mdp/rewards.py`) and standalone version must stay in sync.
- F2.6: Action rate and joint velocity penalties apply to both branches.

**F3 — Validation toolchain — Tool 1 (Generator)**
- F3.1: Generate N labeled sequences via DifferentialIK following scripted waypoints.
- F3.2: Label distribution: 70% reachable / 30% unreachable, 50% success / 50% failure.
- F3.3: All four scenario types: reachable+success, reachable+failure, unreachable+success, unreachable+failure.
- F3.4: Output: JSON file with per-frame joint positions, EE position, cube position, robot base position, joint velocities, gripper state.
- F3.5: Runs in parallel across N environments.

**F4 — Validation toolchain — Tool 2 (Replayer)**
- F4.1: Replay all sequences (or a single specified sequence) in Isaac Sim.
- F4.2: Compute per-frame rewards from live simulation state (not from recorded values).
- F4.3: Two modes: headless (batch) and visualize (renders scene + prints reward stream).
- F4.4: Output: JSON with per-frame rewards for each term, episode totals, expected-high-reward flag.

**F5 — Validation toolchain — Tool 3 (Analyzer)**
- F5.1: Read replay JSON, produce statistical report without Isaac Sim.
- F5.2: Outputs: reward histograms per scenario type, per-term reward curves over time, confusion matrix (expected vs. actual high/low reward), correlation scatter plot.
- F5.3: Report saved as PNG figures + text summary.

**F6 — Unit tests**
- F6.1: Tests run without Isaac Sim (pure Python + PyTorch in micromamba env).
- F6.2: Cover: reward function correctness, waypoint state machine geometry, sequence schema roundtrip, sampling distribution.
- F6.3: No dummy tests — each test validates a realistic failure mode.

### Non-Functional Requirements

- NF1: Unit tests complete in under 10 seconds.
- NF2: Generator supports at least 16 parallel environments.
- NF3: Reward standalone implementation mirrors env implementation exactly (no silent divergence).
- NF4: All world-state inputs to the reward function (robot position, cube position, EE position, joint velocities) come from live simulation state during replay, not from pre-recorded values.

---

## Design Decisions

| Decision | Choice | Rationale | Alternatives considered |
|---|---|---|---|
| Lift height | 0.5 m | Mid-Franka height (~1.0 m total); high enough to be unambiguously "lifted" | Initially proposed 0.05 m (1 cube height); user corrected to mid-Franka |
| Reachable radius | [0.22, 0.65] m | Covers Franka's practical workspace; inner bound avoids self-collision, outer bound is arm reach limit | Validated with cube at [0.45, 0.1, 0.025] (dist≈0.461, reachable) and [0.80, 0.05, 0.025] (dist≈0.802, unreachable) |
| Success EE position | [0.5, 0.0, 0.5] | In reachable zone, at mid-Franka height; reachable by the arm after lifting | — |
| Signal EE position | [0.0, 0.0, 0.8] | Above and in front of robot; unambiguously distinct from any pick position | — |
| Cube spawn zone | x: [0.0, 0.8], y: [-0.6, 0.6] | Covers both reachable and unreachable annulus regions | — |
| Reachable+failure and unreachable+success robot motion | Both go to signal_pos | These scenarios are distinguished by *cube position* (reward mask), not by robot motion. Same EE trajectory, different reward branch active | Could have used different EE trajectories; rejected because it conflates signal motion with the reachability distinction |
| Reward branch gating | Float mask (not boolean) | Enables gradient flow within each active branch; the inactive branch contributes exactly zero | Boolean mask would work for scripted evaluation but blocks gradient in RL training |
| Sampling extraction | `_common/sampling.py` (separate from generate_sequences.py) | Avoids Isaac Sim AppLauncher bootstrap when running unit tests | Initially in generate_sequences.py; broke test collection with INTERNALERROR |
| Frame recording | Includes `robot_pos_w` and `joint_vel` | Enables offline reward recomputation from JSON without running Isaac Sim | Initially omitted; added after noting reward function accesses full world state |
| WaypointStateMachine timing | T_APPROACH=2.0s, T_GRASP=3.5s, T_LIFT=5.5s, total 8s | Gives robot enough time to approach, grasp, and lift at realistic joint velocities | — |

---

## Open Questions

- **Position validation**: The geometry constants (reachable_radius_min/max, success_ee_position, signal_ee_position, cube_spawn_x/y) were proposed by the agent and have not yet been explicitly confirmed by the user. Marked `PENDING VALIDATION` in `04-task-domain.md`.
- **Reward weights**: REWARD_WEIGHTS in reward_eval.py (approach=1.0, lift=10.0, success=15.0, signal=10.0, action_rate=-1e-4, joint_vel=-1e-4) are proposed values; final weights should be tuned after first validation run.
- **Analyzer confusion threshold**: The `10.0` total reward threshold in replay_sequences.py for HIGH/LOW classification is a placeholder; should be set based on observed reward distributions from a real generation run.

---

## Validation Criteria

The project's validation phase is complete when:

1. `pytest tests/validation/franka_cube_pick/` passes with 0 failures (currently: 86/86 ✓).
2. Generator produces 100 sequences without errors; JSON is well-formed and all labels are geometrically consistent.
3. Replayer produces reward JSON where reachable+success episodes have higher total reward than reachable+failure episodes in > 90% of cases.
4. Analyzer confusion matrix shows > 90% accuracy (label matches expected-high-reward classification).
5. Signal reward is zero throughout all reachable episodes; approach/lift/success rewards are zero throughout all unreachable episodes (branch exclusivity).

---

## Changelog

- 2026-04-08: Initial PRD bootstrapped from conversation history (no prior PRD existed). Requirements synthesized retroactively — implementation was already in progress.
- 2026-04-08: Added frame recording of `robot_pos_w` and `joint_vel` after noting reward function accesses full world state.
- 2026-04-08: Updated Design Decisions with sampling extraction decision and frame recording decision.
