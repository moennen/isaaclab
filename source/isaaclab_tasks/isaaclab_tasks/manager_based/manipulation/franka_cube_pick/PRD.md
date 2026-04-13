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
- **Simulator**: Isaac Sim 5.1.0, Isaac Lab (`feature/newton` branch), Python 3.11 via micromamba.
- **Physics engine**: Newton (MJWarp) and PhysX both supported via the preset system (`presets=newton` selects Newton). Single gym ID — no separate Newton task registration.
- **GPU**: L40 via NVIDIA DGX Cloud.
- **Task location**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/franka_cube_pick/` — in the main `isaaclab_tasks` package (not experimental).
- **No table**: cube rests directly on the ground at z = `cube_half_height`.
- **Robot commands via DifferentialIK** for scripted sequences (PhysX only); joint-position actions for RL env (Newton-compatible).

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
- F2.4: Pure-tensor reward kernels live in `reward_utils.py` (task root, no Isaac Lab imports). This is the single source of truth shared by the RL env, tools, and tests.
- F2.5: `mdp/rewards.py` wraps `reward_utils.py` with env data extraction. `reward_eval.py` re-exports from `reward_utils.py`. No reward logic is duplicated.
- F2.6: Action rate and joint velocity penalties apply to both branches.

**F3 — Validation toolchain — Tool 1 (Generator)**
- F3.1: Generate N labeled sequences via DifferentialIK following scripted waypoints.
- F3.2: Label distribution: 70% reachable / 30% unreachable, 50% success / 50% failure.
- F3.3: All four scenario types: reachable+success, reachable+failure, unreachable+success, unreachable+failure.
- F3.4: Output: JSON file with per-frame joint positions, EE position, cube position, robot base position, joint velocities, gripper state.
- F3.5: Runs in parallel across N environments.

**F4 — Validation toolchain — Tool 2 (Replayer)**
- F4.1: Replay all sequences (or a single specified sequence) standalone — no Isaac Sim required.
- F4.2: Re-derive EE position via Newton FK from recorded joint angles (independent of generator).
- F4.3: Compute per-frame rewards using `reward_utils.compute_all_rewards` — same kernels as RL env.
- F4.4: Output: JSON with per-frame rewards for each term, episode totals, expected-high-reward flag.

**F5 — Validation toolchain — Tool 3 (Analyzer)**
- F5.1: Read replay JSON, produce statistical report without Isaac Sim.
- F5.2: Outputs: reward histograms per scenario type, per-term reward curves over time, confusion matrix (expected vs. actual high/low reward), correlation scatter plot.
- F5.3: Report saved as PNG figures + text summary.

**F8 — Validation toolchain — Tool 4 (Visualizer)**
- F8.1: Load all sequences from the JSON and let the user select which one to play via an interactive imgui sidebar panel — no restart required to switch sequences.
- F8.2: Sidebar panel: label filter combo (all / reachable_success / reachable_failure / unreachable_success / unreachable_failure), scrollable sequence listbox (click = play immediately), speed slider (0.1–4.0×), progress bar, current frame info.
- F8.3: Display robot arm (via Newton FK + `viewer.log_state`), cube (orange sphere), success target (green sphere), signal target (blue sphere), EE position (yellow sphere).
- F8.4: Print per-frame reward stream (all terms) to stdout during playback.
- F8.5: Uses `reward_utils.compute_all_rewards` — same kernels as RL env and replayer.
- F8.6: Optional `--sequence_id` to jump to a specific sequence on launch.
- F8.7: No Isaac Sim / AppLauncher required — pure Newton viewer + imgui_bundle (Python only, no build step).

**F6 — Unit tests**
- F6.1: Tests run without Isaac Sim (pure Python + PyTorch in micromamba env).
- F6.2: Cover: reward function correctness, waypoint state machine geometry, sequence schema roundtrip, sampling distribution.
- F6.3: No dummy tests — each test validates a realistic failure mode.

**F7 — Observation space**
- F7.1: State-based policy only (no visual observations).
- F7.2: Single `policy` observation group following the dexsuite pattern — reads body state directly from articulation data, no FrameTransformer.
- F7.3: Observation vector (47 dims):
  - `cube_pos` (3): cube XYZ in robot root frame [m]
  - `cube_quat` (4): cube orientation wxyz in robot root frame
  - `ee_state` (13): EE pos(3) + quat(4) + lin_vel(3) + ang_vel(3) in robot root frame
  - `joint_pos` (9): 7 arm + 2 finger joints [rad or m]
  - `joint_vel` (9): 7 arm + 2 finger velocities [rad/s or m/s]
  - `actions` (9): last joint-position action
- F7.4: EE state is read from `robot.data.body_link_state_w` (panda_hand link) — compatible with Newton.
- F7.5: Observation history may be added in a future iteration; start without it.

### Non-Functional Requirements

- NF1: Unit tests complete in under 10 seconds.
- NF2: Generator supports at least 16 parallel environments.
- NF3: Reward standalone implementation mirrors env implementation exactly (no silent divergence).
- NF4: All world-state inputs to the reward function (robot position, cube position, EE position, joint velocities) come from live simulation state during replay, not from pre-recorded values.
- NF5: All observation and reward functions are Newton-compatible — no FrameTransformer, no PhysX-specific API calls.
- NF6: The sequence generator must run without Isaac Sim (standalone Newton IK), using only the URDF and Newton's IK library.

---

## Design Decisions

| Decision | Choice | Rationale | Alternatives considered |
|---|---|---|---|
| Lift height | 0.5 m | Mid-Franka height (~1.0 m total); high enough to be unambiguously "lifted" | Initially proposed 0.05 m (1 cube height); user corrected to mid-Franka |
| Reachable radius | [0.22, 0.65] m | Covers Franka's practical workspace; inner bound avoids self-collision, outer bound is arm reach limit | Validated with cube at [0.45, 0.1, 0.025] (dist≈0.461, reachable) and [0.80, 0.05, 0.025] (dist≈0.802, unreachable) |
| Success EE position | [0.5, 0.0, 0.5] | In reachable zone, at mid-Franka height; reachable by the arm after lifting | — |
| Signal EE position | [0.0, 0.0, 0.8] | Above and in front of robot; unambiguously distinct from any pick position | — |
| Cube spawn zone | x: [0.0, 0.8], y: [-0.6, 0.6] | Covers both reachable and unreachable annulus regions | — |
| Reachable+failure robot motion | 5 realistic failure modes (approach_no_grip, stop_at_pregrasp, grip_drop_early, wrong_approach_target, descend_open_grip) | Simple "go to signal_pos" was replaced: failures should be *close to real attempts* — gripper hardware failure, mis-timed drop, wrong position, mis-timed grasp — not obviously wrong motions. All modes keep cube_z < 0.45m (reward LOW constraint). Key constraint: grip_drop_early uses _GRIP_DROP_MAX_T = T_GRIP + 1.0 = 8.0s (EE reaches 0.45m only at ~8.19s from ease-in geometry). | Old approach: single mode going to signal_pos — not realistic, would not expose real failure modes to RL training |
| Unreachable+success robot motion | Goes to signal_pos (optionally with 1–2.5s hesitation toward cube) | Distinguished by cube position (reachability mask), not robot motion. Hesitation adds trajectory variety. | Same-motion-as-reachable-failure approach (old) |
| Reward branch gating | Float mask (not boolean) | Enables gradient flow within each active branch; the inactive branch contributes exactly zero | Boolean mask would work for scripted evaluation but blocks gradient in RL training |
| Sampling extraction | `_common/sampling.py` (separate from generate_sequences.py) | Avoids Isaac Sim AppLauncher bootstrap when running unit tests | Initially in generate_sequences.py; broke test collection with INTERNALERROR |
| Frame recording | Includes `robot_pos_w` and `joint_vel` | Enables offline reward recomputation from JSON without running Isaac Sim | Initially omitted; added after noting reward function accesses full world state |
| WaypointStateMachine timing | T_APPROACH=2.0s, T_DWELL=3.0s, T_GRASP=4.5s, T_GRIP=7.0s, T_LIFT=8.5s, episode=10s | Added T_DWELL (1s for PD to converge XY before descent) and T_GRIP (2.5s for fingers to fully close against cube — ke_finger=100 with arm inertia takes ~2.5s). IK commands update at 50Hz; 10 physics substeps per frame (2ms each). | T_APPROACH=2.0, T_GRASP=3.5, T_LIFT=5.5 (old, simpler schedule) |
| Physics engine | Newton (MJWarp) + PhysX via `PresetCfg` | Single gym ID; backend selected at CLI via `presets=newton`. `launch_simulation()` auto-detects `NewtonCfg` in config — no `--experience` flag needed. Newton parameters validated against `generate_sequences.py`. | Separate gym IDs per backend (superseded — led to duplicate registrations and split maintenance) |
| Task package location | `isaaclab_tasks` (main, not experimental) | `isaaclab_tasks_experimental` was removed; all tasks now live in `isaaclab_tasks` | Was initially in experimental; refactored early in session |
| EE tracking — no FrameTransformer | Read from `robot.data.body_link_state_w` (panda_hand) | FrameTransformer sensor is not available on Newton branch | FrameTransformer was used initially; removed when Newton compatibility was required |
| Observation style | dexsuite-style, single Policy group, 47 dims | No FrameTransformer; reads body state directly; proven pattern from dexsuite tasks; Newton-compatible | Initial flat obs with `ee_position_in_robot_root_frame` (FrameTransformer); superseded |
| Observation history | Not included (initial version) | Start simple; history can be added later if policy fails to disambiguate states | Adding history from the start adds complexity before baseline is proven |
| Warp version conflict — root fix | Delete bundled warp 1.8.2 from `omni.warp.core-1.8.2+lx64/warp/`, create symlink to pip-installed warp 1.12.1 | Isaac Sim 5.1.0 ships `omni.warp.core-1.8.2` with a bundled warp 1.8.2 directory. The extension's own `extension.py` was designed to use a symlink to pip-installed warp (developer mode), but falls back to the bundled copy when the real directory exists. Replacing it with the symlink makes Isaac Sim use warp 1.12.1. | Runtime shim on `Array.__class_getitem__`; Newton-only kit (both workarounds, now removed) |
| Sequence generator physics | Newton IK standalone (no AppLauncher) | Standalone is faster, reproducible, and doesn't depend on Isaac Sim startup. Kept even after warp fix. | DifferentialIK (PhysX); running Isaac Sim with Newton |
| Cube lift mechanism | Pure contact friction (Newton MuJoCo solver, no assist forces) | Force-based 3D grip assist was discarded: applies external body forces to the cube, bypassing the contact/friction problem we need to validate. Virtual grasp teleportation was also discarded: physically incorrect. Current approach uses pure Newton contacts — same physics as RL training. Required matching the Newton example (example_ik_cube_stacking.py): PD gains ×7.5, gravity compensation, armature, effort limits, ke=5e4, kf=1e3, impratio=1000, cone=elliptic, 10 substeps of 2ms per 20ms outer step. | Force-based 3D grip assist (k_p=20, k_d=10); virtual grasp (teleportation) |
| Contact timestep stability | 10 substeps of 2ms per 20ms outer step | ke=5e4 with m=0.1kg → contact period T=8.9ms. dt=20ms > T causes numerical instability. 10 substeps → sub_dt=2ms < T → stable. Matches the Newton example's sim_substeps=10 approach. | Single 20ms step (caused premature cube knock at approach phase) |
| Missing pip dependencies | `lazy-loader`, `isaaclab_physx`, `isaaclab_newton`, `pycollada` installed | These were missing from the env despite isaaclab editable install pointing to source 4.5.13 on the dexsuite branch | Not needed on standard main branch |
| Cube spawner | `CuboidCfg(size=(0.05,0.05,0.05))` instead of `UsdFileCfg(dex_cube_instanceable.usd)` | `UsdFileCfg` has no `physics_material` field — friction and Newton ke/kd cannot be set via its constructor. `CuboidCfg` (shape spawner) has `physics_material: RigidBodyMaterialCfg` natively. Same approach used by dexsuite for Newton. The DexCube USD visuals are irrelevant for RL training. | `UsdFileCfg` — does not support physics material override |
| Physics backend preset | `FrankaCubePickPhysicsCfg(PresetCfg)` with `default=PhysxCfg(...)`, `physx=PhysxCfg(...)`, `newton=NewtonCfg(...)` | Follows the pattern used by ant, cartpole, humanoid, and dexsuite tasks. Timing scalars (`sim.dt`, `decimation`, `render_interval`) use `preset(default=X, newton=Y)` inline. Both backends give 50 Hz control rate: PhysX via 2×10ms, Newton via 10×2ms. | Separate `newton_env_cfg.py` + separate gym IDs — superseded; maintenance overhead, inconsistent with project-wide preset pattern |
| Sequence replayer physics | Newton FK + full physics (robot PD tracks recorded joint_pos_cmd; cube fully simulated) | Pure FK (analytic cube teleportation) bypasses the contact/friction problem. Replayer now uses the same Newton physics model as the generator: same PD gains, gravcomp, contact params, solver, substeps. Robot motion is FK-equivalent (PD tracks recorded commands). Cube is free rigid body. | Analytic cube (Z follows EE Z after gripper closes); standalone FK |
| Replay high/low threshold | `10.0` total episode reward | Empirically validated: LOW sequences score 0.2–0.7, HIGH sequences score 103–1577. Any value in [1, 100] would be unambiguous. | Was a placeholder pending first run |
| Generator batching | Multi-world Newton (`begin_world`/`add_builder`/`end_world`) with `--num-worlds` (default 16) | Single-world generator is GPU-idle (IK bottleneck at N=1). Batching tiles N worlds into one physics model so all N episodes run simultaneously. IK also batched: `IKSolver(n_problems=N)`. CUDA graph capture disabled (SolverMuJoCo switches CUDA streams internally). Robot builder is unfinalized (no ground plane); cube added per-world inside `begin_world`/`end_world`; IK model is robot-only for `IKSolver`. Cube positions set via `state_0.joint_q` at reset so model is built once and reused across batches. Actual speedup: ~29× vs. single-world for N=16 (4.5 min / 16 seqs). | Single-world sequential (old: ~8 min/seq), 4 parallel shards via --seq_id_start (workaround, 63 min / 100 seq) |
| Reward architecture | `reward_utils.py` at task root — single source of truth, no Isaac Lab imports | Original design had `reward_eval.py` reimplementing the same math as `mdp/rewards.py`. Any drift between them would silently break validation. `reward_utils.py` is imported by both: env wrapping layer (`mdp/rewards.py`) and tools (`reward_eval.py` re-exports it). Tests, replayer, and visualizer all use the same kernels as RL training. | Keeping duplicate implementations in sync manually |
| Visualizer | Newton ViewerGL (`visualize_sequence.py`) — no build, pure Python | Newton ships ViewerGL as a Python package; no compilation needed. Same reward kernels as training so what you see in the viewer is what the reward function sees. | Isaac Sim USD renderer; custom OpenGL app |

---

## Open Questions

- **Position validation**: The geometry constants (reachable_radius_min/max, success_ee_position, signal_ee_position, cube_spawn_x/y) were proposed by the agent and have not yet been explicitly confirmed by the user. Marked `PENDING VALIDATION` in `04-task-domain.md`.
- **Reward weights**: REWARD_WEIGHTS validated by toolchain run (HIGH/LOW gap is clear); final tuning may happen during RL training. Current structure: reachable branch (approach=1.0, grip=5.0, lift=10.0, success=15.0) + unreachable branch (go_to_signal=1.0 shaping, signal_reached=10.0 binary). Remaining branch imbalance (~2.8× unreachable vs reachable) requires early episode termination to fully resolve; acceptable for initial RL training.
- **Finger joint indices**: `grip_cube_reachable` uses hardcoded joint indices [7, 8] for `panda_finger_joint1/2`. Validated by observation that reachable_success reward increased from ~672 to ~1318 when `closed_threshold=0.06` (contact resistance from cube keeps q₁+q₂≈0.04 when grasping). If robot URDF changes, these indices must be updated.

---

## Validation Criteria

The project's validation phase is complete when:

1. `pytest tests/validation/franka_cube_pick/` passes with 0 failures (**86/86 ✓ 2026-04-08**).
2. Generator produces N sequences without errors; JSON is well-formed and all labels are geometrically consistent (**✓ 2026-04-10** — 40 seqs: 9 R+S, 15 R+F, 9 U+S, 7 U+F).
3. Replayer produces reward JSON where reachable+success episodes actually lift the cube and unreachable+success episodes reach signal_pos in > 90% of cases (**✓ 2026-04-10** — 100% accuracy, 100/100 sequences). Criterion: `success_frame >= 0` (task completion), not a reward threshold.
4. Analyzer confusion matrix shows > 90% accuracy (success_frame matches expected_success label) (**✓ 2026-04-10** — 100.0% accuracy, 48 TP, 52 TN, 0 FN, 0 FP).
5. Signal reward is zero throughout all reachable episodes; approach/lift/success rewards are zero throughout all unreachable episodes (branch exclusivity) (**✓ 2026-04-10** — 0/25,000 reachability mask mismatches).

**Validation phase complete as of 2026-04-10 (re-validated with batched Newton physics, 100/100 correct).**

**Reward improvements validated 2026-04-11 — see Changelog. New structure: 6 terms (was 4), 100/100 correct.**

---

## Changelog

- 2026-04-08: Initial PRD bootstrapped from conversation history (no prior PRD existed). Requirements synthesized retroactively — implementation was already in progress.
- 2026-04-08: Added frame recording of `robot_pos_w` and `joint_vel` after noting reward function accesses full world state.
- 2026-04-08: Updated Design Decisions with sampling extraction decision and frame recording decision.
- 2026-04-08: Added Newton-first constraint; updated task location from experimental to `isaaclab_tasks`.
- 2026-04-08: Added F7 (observation space) requirement group — dexsuite-style, 47-dim, state-based, Newton-compatible (no FrameTransformer).
- 2026-04-08: Added NF5 (Newton compatibility). Added Design Decisions for physics engine, package location, EE tracking approach, observation style, and observation history.
- 2026-04-08: Toolchain investigation: Isaac Sim 5.1.0 + Newton warp conflict (bundled warp 1.8.2 vs standalone 1.12.1). Solution: use Newton IK standalone (no AppLauncher) for generate_sequences.py. Added NF6.
- 2026-04-08: Rewrote replay_sequences.py to run standalone with Newton FK (no AppLauncher). Added Design Decision for replayer physics approach. Ran full toolchain: 100 sequences, 100% accuracy, 0 mask mismatches. All validation criteria met. Validation phase complete.
- 2026-04-08: Fixed warp 1.8.2/1.12.1 conflict at root cause: replaced bundled warp 1.8.2 in `omni.warp.core-1.8.2+lx64/` with symlink to pip-installed warp 1.12.1. AppLauncher now starts cleanly with warp 1.12.1 active. Updated Design Decisions.
- 2026-04-08: Eliminated reward code duplication. Created `reward_utils.py` (task root, pure torch) as single source of truth. `mdp/rewards.py` now calls `reward_utils` instead of reimplementing math. `reward_eval.py` is now a thin re-export. Tests updated to `compute_*` naming. Added F2.4–F2.5, F8 requirements and Design Decision. Added Tool 4 (visualizer) using Newton ViewerGL.
- 2026-04-08: Enhanced Tool 4 (visualize_sequence.py) with imgui sidebar: label filter combo, scrollable sequence listbox (click to play), speed slider, progress bar, frame info. `--sequence_id` is now optional (jumps to sequence on launch; defaults to first). Updated F8 requirements.
- 2026-04-08: All four tool scripts now resolve `--input`/`--output` relative to the task folder (`<task_root>/data/validation/`) by default. No path arguments needed for standard runs. Existing outputs copied to task-relative location.
- 2026-04-09: Removed virtual grasp (cube teleportation). Replaced with force-based 3D grip assist: soft finger BOX contacts (ke_BOX=100 N/m) + 3D spring-damper body force on cube after T_GRIP=7.0s (k_p=20, k_d=10). Z-only was tried first but cube decoupled laterally up to 4.06m from EE. 3D control keeps cube within 0.11m of EE (8/9 R+S sequences). Three GPU-CPU sync points eliminated in generate_sequences.py. Regenerated 40-sequence dataset (seed=1234): 39/40 OK replay (seq_0038 has early knock at t=4.96s before T_GRIP — marginal lift, same MISMATCH as Z-only run). Outputs: data/validation/sequences_3d.json + replay_3d.json.
- 2026-04-10: Changed SUCCESS/FAILURE discriminator from episode_total > threshold to success_frame >= 0.
  Near-cube failure modes (approach_no_grip, descend_open_grip) accumulate 20–32 approach reward — any
  fixed threshold would be fragile on regeneration. success_frame is set by physics directly (cube_z >
  lift_height; ee within 0.15m signal). Updated replay_sequences.py and analyze_results.py.
- 2026-04-10: Generated 100-sequence dataset via 4 parallel shards (--seq_id_start added to
  generate_sequences.py; new merge_sequences.py). 99/100 correct, 1 FN (IK at signal_pos singularity).
- 2026-04-10: Replaced 3 simplistic reachable_failure modes with 5 realistic failure modes in
  waypoint_ik.py: approach_no_grip (gripper hardware failure), stop_at_pregrasp (never descends),
  grip_drop_early (times out mid-lift), wrong_approach_target (wrong XY), descend_open_grip (mis-timed
  grasp). All modes provably keep cube_z < 0.45m. Added _GRIP_DROP_MAX_T = T_GRIP + 1.0 = 8.0s safety
  constant. Updated Design Decisions: reachable_failure now has 5 modes. Updated F3.3.
- 2026-04-10: Refactored generate_sequences.py to batched multi-world simulation. Robot builder is
  unfinalized (no ground plane); cube added per-world inside begin_world/end_world; batched model built
  once; cube positions set via state_0.joint_q at reset. IK solvers created once per stage, targets updated
  each step by reassigning warp arrays. --num-worlds CLI arg (default 16) controls batch size. ~29× speedup
  vs. N=1 sequential (4.5 min / 16 seqs vs 8 min / 1 seq).
  Several Newton API bugs fixed in this version: (1) `body_label` not `body_key`; `label=` not `key=` for
  add_shape_box/add_body; (2) IK names are `IKObjectivePosition`, `IKObjectiveRotation`,
  `IKObjectiveJointLimit`, `IKJacobianType` (NOT the `IKPositionObjective`/`IKJacobianMode` variants);
  (3) `control.joint_target_pos` has DOF count (6 per free joint), not coord count (7), so n_ctrl_per_world
  = len(control.joint_target_pos)//N; (4) Robot must initialize to `_HOME_JOINT_Q` (Newton example "ready"
  pose) — all-zero joint start puts EE at [0.088, 0, 0.926], 13cm XY off workspace, cube never gripped;
  (5) SolverMuJoCo can't be CUDA-graph-captured (stream switch inside solver), falls back to eager mode
  with state reset.
- 2026-04-10: Removed force-based grip assist. Replaced with pure contact friction grasping using Newton example physics params (example_ik_cube_stacking.py). Root cause analysis: PD gains 7.5× too low, no gravcomp, no armature/effort limits, kf=0, impratio=1, dt=20ms > contact period T=8.9ms. Fixed: ke=4500/3500/2000, gravcomp, armature, effort limits, ke=5e4, kd=5e2, kf=1e3, mu=0.75, impratio=1000, cone=elliptic, 10 substeps×2ms. Replayer rewritten: FK+full Newton physics, 20×2ms substeps per frame, no analytic cube. Fixed reward_eval.py pxr import (importlib.util). Validated: 40/40 [OK], 100% accuracy (seed=1234, 40 seqs): R+S=304–415 HIGH (9/9, all success_frame=211), R+F=0.2–1.4 LOW (15/15), U+S=1133–1594 HIGH (9/9), U+F=2.2–2.6 LOW (7/7). 0/10000 mask mismatches. Better than old grip-assist (39/40). Outputs: sequences_physics.json, replay_physics.json, report_physics/.
- 2026-04-10: Refactored replay_sequences.py to batched multi-world simulation, mirroring
  generate_sequences.py. Identical build_robot_builder/build_batched_model/reset_batch_state;
  per-frame sets ctrl from recorded joint_pos_cmd for all N worlds simultaneously; vectorised
  reward computation with (N,3) torch tensors. Added --num-worlds arg (default 16).
  15 min / 100 seqs vs 47 min sequential (3.1× speedup). 100/100 [OK], identical accuracy.
- 2026-04-10: Fixed two bugs in generate_sequences.py + replay_sequences.py after batched rewrite
  (A) Generator bug: `joint_pos_cmd` in each recorded frame was storing `joint_q_ik_np[w]` (raw IK
  output with fingers always at 0.04 open) instead of `ctrl_np[w, :_N_ROBOT_JOINTS]` (actual control
  signal, which overrides finger joints 7–8 with the real finger command 0.04→0.0 at grasp time, and
  0.0→0.04 at drop time for grip_drop_early). Fixed: now records `ctrl_np[w, :_N_ROBOT_JOINTS]`.
  (B) Replay bug: `control.joint_target_pos` assignment used `np.zeros(n_phys)` (size 16 = coord count)
  instead of `np.zeros(n_dof)` (size 15 = DOF count). Free joint has 7 coordinates but only 6 DOFs.
  Fixed: now uses n_dof and `.assign()` instead of replacement.
  (C) Replay bug: robot initialised from URDF default (all-zeros, EE at [0.088, 0, 0.926]) rather than
  `_HOME_JOINT_Q` (EE at [0.495, 0, 0.313]). This caused false signal triggers (EE 0.155m from
  signal_pos [0,0,0.8]) and failed grasps. Fixed: added `_HOME_JOINT_Q` initialisation to
  `build_phys_model()` in replay_sequences.py.
  Dataset regenerated and re-validated: 100/100 [OK], 100.0% accuracy. Confusion: 48 TP, 52 TN, 0 FN,
  0 FP. Physics variance: mean ≤ 0.003 rad across all label types. Outputs: sequences.json, replay.json.
- 2026-04-11: Identified and fixed command timing approximation (RECORD_EVERY=2): between two recorded
  frames, generator ran 10 substeps with command A then 10 with command B (B not recorded); replay applied
  A for all 20 substeps → stale-command error. Fix: RECORD_EVERY=1 (500 frames/seq at 50 Hz vs 250 at
  25 Hz) and replay _N_SUBSTEPS_PER_FRAME=10 (was 20). Physics variance dropped to max 0.0002 (floating-
  point noise only). All default paths renamed to reference_ik_sequences.json / reference_ik_replay.json /
  reference_ik_report/. Old validation files (sequences.json, replay.json, report/) removed.
- 2026-04-11: Reward enhancements (two low-risk structural fixes, validated by replay):
  (A) Added `grip_cube_reachable` binary term (weight 5.0): fires when gripper is squeezing the cube
  (gripper_width = q_finger1 + q_finger2 < 0.06; contact resistance keeps sum ≈ 0.04 against 4cm cube)
  AND cube_z > 0.05m. Bridges approach→lift reward gap. reachable_success mean reward: 672 → 1318 (+96%).
  (B) Added `signal_reached_unreachable` binary term (weight 10.0): fires when EE within 0.05m of signal
  position when unreachable. Mirrors lift_cube_reachable for the unreachable branch. `go_to_signal_position`
  weight reduced from 10.0 to 1.0 (dense shaping only, binary carries primary signal). unreachable_failure
  mean reward: 0.24 → 0.06 (much cleaner baseline — no more dense accumulation at random positions).
  Branch gap reduced from 4× to 2.8× (unreachable_success 3744 vs reachable_success 1318). Remaining gap
  requires early episode termination; acceptable for RL training with γ=0.99 discounting.
  All reward logic updated in reward_utils.py (source of truth), mdp/rewards.py, mdp/__init__.py,
  reward_eval.py. Validated: 100/100 [OK], 48 TP, 52 TN, 0 FN, 0 FP. Physics variance ≈ 0.
  Updated analyze_results.py TERM_NAMES/REWARD_WEIGHTS (now 8 terms). Updated env cfg RewardsCfg.
- 2026-04-11: Added Tool 5 — Observation Analyzer (analyze_observations.py). Runs 7 analyses on
  reference_ik_sequences.json + reference_ik_replay.json; outputs 7 PNGs + report.txt to
  reference_ik_obs_report/. No Isaac Sim required (pure numpy/sklearn/matplotlib).
  Key findings from the reference dataset (100 seqs, 50,000 frames):
  (1) Obs–Reward: cube_z is top predictor of grip/lift rewards (r≈0.9/0.8). grip binary and finger
    positions strongly predict approach_cube reward (r≈0.83). ee_z and j4 predict signal rewards (r≈0.78).
  (2) Obs–Action: joint positions near-perfectly correlated with commands (r>0.99) — scripted IK
    sequences; finger commands anti-correlated with grip binary (r≈−0.95).
  (3) Markov: only fv0/fv1 (finger velocities) show ΔR²>0.05 (ΔR²≈0.076) — near-Markovian overall.
    Arm joints and cube position are fully Markovian at 50Hz. Finger velocity transitions at
    open/close events account for the small non-Markovian signal.
  (4) PCA: 90% variance in 13 of 25 dims (52% effective dimensionality). Redundancy from joint–command
    correlation (r>0.99) and slow state evolution.
  (5) Discriminability: ee_z (F=0.57) and j4 (F=0.56) best discriminate reachable/unreachable;
    cube_x (F=0.52) provides direct reachability signal. Success vs failure: ee_z (F=0.31) and
    finger positions (F≈0.28) are top discriminators.
  (6) Reward predictability: R²=0.87–0.94 within each label group — current obs linearly predicts
    ~90% of per-frame reward variance. Suggests no critical missing features for reward correlation.
  (7) Autocorrelation: all 25 dims have lag-1 ac≥0.96 — smooth control at 50Hz, no noisy dims.
  Candidate missing observations: ee_to_cube_xyz (3), gripper_width continuous (1), dist_cube_xy (1).
- 2026-04-12: Ported RL task env to preset system. Replaced separate Newton gym IDs
  (`Isaac-Pick-Cube-Franka-Newton-v0`) with single `Isaac-Pick-Cube-Franka-v0` selecting
  Newton via `presets=newton`. `FrankaCubePickPhysicsCfg(PresetCfg)` defined in base env cfg
  with `default=PhysxCfg(...)` and `newton=NewtonCfg(...)` (validated params). Timing scalars
  (`sim.dt`, `decimation`, `render_interval`) use `preset()` inline — PhysX: 2×10ms, Newton:
  10×2ms, both 50 Hz control. `newton_env_cfg.py` deleted. Cube spawner changed from
  `UsdFileCfg` (no `physics_material` field) to `CuboidCfg(size=(0.05,0.05,0.05))` with
  `physics_material=RigidBodyMaterialCfg(mu=0.75, ke=5e4, kd=5e2)`. Verified: both backends
  instantiate cleanly; `presets=newton` resolves to `NewtonCfg`, decimation=10, dt=0.002.

---

## franka_vbd_cube_pick Extension

*Extension task using VBD (Vertex Block Descent) deformable cube physics.*
*Task lives in `../franka_vbd_cube_pick/`. All decisions below are additive — base rigid task decisions still apply.*

### Problem Statement (VBD)

The same pick-or-signal task but with a deformable cube simulated via VBD (FEM tetrahedral mesh, Vertex Block Descent solver). Validates that the validation + RL pipeline works end-to-end with a non-rigid object, and that two-way rigid-soft coupling produces physically realistic grasping.

### Additional Requirements

**F10 — Deformable cube**
- F10.1: Cube is a VBD soft body created via `add_soft_grid()` (not a rigid asset).  No `scene.object` entry in the Isaac Lab scene.
- F10.2: Cube geometry: `cube_resolution=3` → `(R+1)³=64` particles, `5R³=135` tets.
- F10.3: Material: Young's modulus + Poisson's ratio exposed in config; Lamé (k_mu, k_lambda) computed in manager.
- F10.4: `k_damp` and `soft_contact_kd` MUST stay ≈1e-5 (position-level stiffness multipliers — NOT velocity damping).
- F10.5: Cube CoM position = mean of particle positions per env.  Orientation = Kabsch SVD alignment vs rest pose.
- F10.6: Newton-only task (no PhysX equivalent).  Hardcoded: `decimation=10`, `sim.dt=0.002`, `num_substeps=10`.

**F11 — Two-way coupling**
- F11.1: Same-substep operator splitting: detect soft contacts → write reactions into `state.body_f` → rigid MuJoCo step → VBD step.
- F11.2: Full Coulomb friction coupling (not normal-only): finger actuators feel tangential friction load from cube.
- F11.3: Coupling shared in `physics/vbd_coupling.py` — imported by manager, standalone scripts, and tests.

**F12 — Newton package patches** (branch `nicolasm/isaaclab-task-skills` in newton repo)
- F12.1: `create_soft_contacts_batched` kernel (`36a67b8`) — batched collision avoids int32 overflow at 4096 envs.
- F12.2: `clamp_particle_inertia` kernel (`837d80e`) — velocity clamping prevents NaN at first rigid-VBD contact.
- F12.3: `HeightfieldData` bug fix (`ed62423`) — removes undefined type from kernel signature.

**F13 — VBD validation toolchain**
- F13.1: `scripts/generate_sequences.py` — standalone VBD sequence generator (robot + VBD soft grid, two-phase stepping, particle CoM tracking).
- F13.2: `scripts/replay_sequences.py` — replays generated sequences in VBD physics, computes VBD reward terms.
- F13.3: `scripts/analyze_results.py` — same analysis as rigid task (task-independent).
- F13.4: Same JSON schema as rigid task (`_common/sequence_schema.py`).
- F13.5: `cube_pos_w` in sequences = particle CoM (not rigid body joint_q).
- F13.6: No CUDA graph in standalone scripts (eager mode; VBD+CollisionPipeline warm-up outside loop).

### Gym IDs

- `Isaac-Pick-VBD-Cube-Franka-v0` — training (requires `presets=newton`)
- `Isaac-Pick-VBD-Cube-Franka-Play-v0` — play (16 envs)

### Decision Log (VBD extension)

- 2026-04-13: VBD task created (`6ab65e7`).  16 files: env cfg, physics manager, MDP layer, gym registration, skill 07.
  Key design: no `scene.object`; cube managed by `FrankaVbdCubePickNewtonManager`; graph coloring via manual tet edge build (not `construct_particle_graph` which doesn't exist in this Newton).
- 2026-04-13: Newton VBD patches committed (`36a67b8`, `837d80e`, `ed62423`): batched collision kernel, velocity clamping, HeightfieldData fix.  All three needed before any VBD simulation can run.
- 2026-04-13: Validation scripts created (`scripts/`): generate_sequences.py, replay_sequences.py, analyze_results.py — follow same pattern as rigid skill 05.  **Validation not yet run** (PENDING).
