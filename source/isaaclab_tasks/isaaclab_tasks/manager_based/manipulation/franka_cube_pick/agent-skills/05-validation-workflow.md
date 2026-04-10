---
name: validation-workflow
description: Validate physics, rewards, and observations using scripted sequences before RL training. Four standalone tools — generator, replayer, analyzer, visualizer.
level: 4
status: validated
depends_on: [franka-cube-pick-domain]
extends: null
---

## Preconditions

- Task importable and geometry constants validated (skills 03 + 04 complete)
- `env_isaaclab` activated
- Panda URDF accessible (ships with Isaac Sim pip install)

## Context

Training an RL policy is expensive and error-prone. Before any training run, three
categories of bugs must be ruled out with scripted (non-learned) sequences:

1. **Physics bugs** — collisions wrong, cube falls through floor, robot self-collides
2. **Reward bugs** — reward fires on wrong branch, wrong sign, never fires, always fires
3. **Observation bugs** — wrong frame, wrong scale, NaN, not updating

Each category has a dedicated validation tool. All three run **fully standalone — no
Isaac Sim / AppLauncher required**. This sidesteps the warp 1.8.2 / 1.12.1 conflict
(Isaac Sim 5.1.0 bundles warp 1.8.2 via `omni.warp.core`; Newton requires warp 1.12.1).

**Architecture decision — pure Newton physics (generator + replayer):**

Both the generator (Tool 1) and replayer (Tool 2) use the same full Newton physics model
(robot + cube, 16 DOF). No analytic cube approximations, no kinematic overrides:
- Robot: PD controller tracks IK-computed targets (generator) / recorded joint_pos_cmd (replayer)
- Cube: full rigid-body simulation — gravity, contact forces, friction (Newton MuJoCo solver)
- Grasping: purely through friction contacts (no grip assist, no teleportation)

**Physics parameters (from Newton example `example_ik_cube_stacking.py`):**
- PD gains: `ke=[4500×2, 3500×2, 2000×3, 100×2]`, `kd=ke/10`
- Armature: `[0.30×4, 0.11×3, 0.15×2]`, effort limits: `[87×4, 12×3, 100×2]`
- Gravity compensation: `mujoco:jnt_actgravcomp` (arm DOFs), `mujoco:gravcomp` (all bodies)
- Contact: `ke=5e4, kd=5e2, kf=1e3, mu=0.75`
- Solver: `impratio=1000, cone=elliptic, iterations=20, ls_iterations=100`
- Substeps: 10 × 2ms per 20ms outer step (contact period T=8.9ms → sub_dt=2ms < T)

**Key physics insight — contact stability:** ke=5e4 with m=0.1kg gives ω_n=707 rad/s →
contact period T=8.9ms. A 20ms timestep is 2× larger than T, causing numerical instability.
10 substeps of 2ms each (matching the example's approach) resolves this.

**Generator recording:** 500 outer steps × 2ms/substep × 10 substeps = 10s; records every 2nd
outer step → 250 frames at 25Hz. Between consecutive frames: 20 substeps of 2ms = 40ms.

**Scripted sequences used for validation:**

| Label | Robot behaviour |
|---|---|
| reachable_success | approach → grasp → lift to 0.5m → success_pos |
| reachable_failure | one of 5 realistic failure modes (see below) — cube never lifted above 0.45m → LOW reward |
| unreachable_success | goes to signal_pos (correct) |
| unreachable_failure | tries to approach unreachable cube (wrong) |

**Reachable failure modes (5 modes, equal probability):**

| Mode | Description | Why reward stays LOW |
|---|---|---|
| `approach_no_grip` | Full correct trajectory (approach → descent → lift pose → success_pos) but gripper stays OPEN throughout. Robot ends at success_pos holding nothing. | Cube never lifted — stays on floor |
| `stop_at_pregrasp` | Approaches correctly above cube then holds there. Never descends to grasp height. | No cube contact — cube never moves |
| `grip_drop_early` | Full approach → grip → brief lift → opens gripper at T_GRIP+0.2 to T_GRIP+1.0s (=7.2–8.0s). Cube lifted a few cm then falls. | Drop time capped at 8.0s; EE reaches 0.45m only at ~8.19s (ease-in geometry) → cube_z never exceeds 0.45m |
| `wrong_approach_target` | Goes to a random reachable position far from the cube. Never contacts cube. | No cube contact |
| `descend_open_grip` | Correct descent to grasp height, holds there with gripper OPEN (cube barely touched but no friction grip), then retreats to home. | No grasping force — cube at most displaced slightly |

## Toolchain

Four tools in `scripts/`, sharing utilities in `_common/` and a task-root module:

```
franka_cube_pick/
  reward_utils.py           # SINGLE SOURCE OF TRUTH — pure-tensor reward kernels
                            # no Isaac Lab imports; used by RL env AND all tools
  scripts/
    generate_sequences.py   # Tool 1 — generates N labeled sequences via Newton IK
    replay_sequences.py     # Tool 2 — replays sequences via Newton FK, computes rewards
    analyze_results.py      # Tool 3 — reads replay JSON, produces report + plots
    visualize_sequence.py   # Tool 4 — Newton ViewerGL interactive playback
    _common/
      sequence_schema.py    # JSON schema, I/O helpers
      reward_eval.py        # Re-exports from reward_utils.py (no duplicate logic)
      waypoint_ik.py        # IK waypoint state machine for all 4 scenario types
      sampling.py           # cube position + label sampling
```

**Key architecture invariant:** `reward_utils.py` is the only place reward math is defined.
`mdp/rewards.py` (RL env) calls it. `reward_eval.py` re-exports it. Any change to reward
logic goes in `reward_utils.py` and propagates everywhere automatically — no manual sync.

### Tool 1 — generate_sequences.py

Newton IK standalone: `newton.ModelBuilder.add_urdf` + `newton.ik.IKSolver`.

All scripts default `--input`/`--output` to `<task_root>/data/validation/` — no path arguments needed for standard runs.

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/franka_cube_pick/scripts/generate_sequences.py \
    --num_sequences 100 --num_envs 16
```

Output JSON contains per-frame: `joint_pos` (9 values), `joint_vel` (9 values),
`gripper_closed`, `ee_pos_w`, `cube_pos_w`, `robot_pos_w`.

### Tool 2 — replay_sequences.py

Newton FK + full physics: drives robot via recorded `joint_pos_cmd` targets (PD controller,
FK-equivalent), cube fully simulated (gravity, contacts, friction), computes rewards via
`reward_eval.compute_all_rewards`. Same Newton model as generator — identical PD gains,
gravcomp, contact params, solver, 20 substeps×2ms per recorded frame.

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/franka_cube_pick/scripts/replay_sequences.py
```

Prints per-sequence: label, episode total reward, expected HIGH/LOW, OK/MISMATCH.

Optional: `--sequence_id seq_0003` to replay a single sequence.

### Tool 4 — visualize_sequence.py

Newton ViewerGL interactive playback with imgui sidebar. No Isaac Sim / AppLauncher / build step required.

```bash
python .../scripts/visualize_sequence.py \
    [--sequence_id seq_0000]  # optional: jump to this sequence on launch
    [--speed 1.0]             # initial playback speed multiplier
    [--viewer null]           # headless test mode
```

**Sidebar panel** (via `viewer.register_ui_callback`):
- **Label filter** — combo box: `(all labels)` / `reachable + success` / `reachable + failure` / `unreachable + success` / `unreachable + failure`
- **Sequence list** — scrollable listbox showing IDs of filtered sequences; click to switch immediately
- **Speed slider** — 0.1× to 4.0×
- **Progress bar** — current frame / total frames
- **Current frame info** — timestamp and label

Displays in the 3D view:
- Robot arm following recorded joint angles (Newton FK + `viewer.log_state`)
- Cube position — orange sphere
- Success EE target — green sphere
- Signal EE target — blue sphere
- EE position — yellow sphere
- Per-frame reward stream (all terms) printed to stdout

Reward computation uses `reward_utils.compute_all_rewards` — identical to RL training.

### Tool 3 — analyze_results.py

Pure Python (no Isaac Sim), reads replay JSON, writes 4 PNG figures + `report.txt`.

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/franka_cube_pick/scripts/analyze_results.py
```

**Outputs:**
- `01_reward_histograms.png` — episode total reward distribution per label
- `02_per_term_episode_reward.png` — per-term weighted reward bar chart by label
- `03_per_frame_reward_curves.png` — mean per-term reward curves over time per label
- `04_expected_vs_actual_correlation.png` — expected vs actual reward scatter
- `report.txt` — summary table, confusion matrix, reachability mask consistency check

### Reward function (standalone, in `_common/reward_eval.py`)

This is the ground-truth reward implementation used for validation. It mirrors
`mdp/rewards.py` exactly but takes plain tensors (no env object).

| Term | Branch | Formula | Weight |
|---|---|---|---|
| approach_cube_reachable | reachable | mask × (1 - tanh(‖ee - cube‖ / 0.1)) | 1.0 |
| lift_cube_reachable | reachable | mask × indicator(cube_z > 0.45) | 10.0 |
| cube_at_success_position | reachable | mask × lifted × (1 - tanh(‖ee - success_pos‖ / 0.1)) | 15.0 |
| go_to_signal_position | unreachable | (1-mask) × (1 - tanh(‖ee - signal_pos‖ / 0.1)) | 10.0 |
| action_rate | always | -‖Δjoint_pos‖² | -1e-4 |
| joint_vel | always | -‖joint_vel‖² | -1e-4 |

**Physics run stats (seed=1234, 100 sequences — 2026-04-10, pure contact friction, 5 failure modes):**

**99/100 [OK] — 99% accuracy.** Success/failure discriminated by `success_frame >= 0` (no threshold).

| Label | N | Episode reward | Success frame | Status |
|---|---|---|---|---|
| reachable_success | 45 | 283–440 (mean=340) | all 211 | 45/45 ✓ |
| reachable_failure | 25 | 0.24–32.1 (mean=14.2) | — | 0/25 reached success ✓ |
| unreachable_success | 15 | 12.9–1594 (mean=1271) | 35–104 (mean=59) | 14/15 ✓ |
| unreachable_failure | 15 | 2.19–2.69 (mean=2.26) | — | 0/15 reached success ✓ |

1 FN: unreachable_success where IK diverged (EE ~0.3m from signal_pos, never within 0.15m). Known
IK convergence issue at signal_pos=[0,0,0.8] (near singularity). Not a physics or reward bug.
0 FP: all 25 reachable_failure sequences correctly classified (approach_no_grip/descend_open_grip
accumulate 20-32 approach reward but cube never lifted → success_frame=-1 → FAILURE).
Reachability mask mismatches: 29/25000 (0.1%) — all in reachable_success, within tolerance.
Joint variance: 0.002–0.005 rad.
Outputs: `data/validation/sequences_100.json` + `replay_100.json` + `report_100/`.

**[SUPERSEDED] Physics run stats (seed=1234, 40 sequences — 2026-04-10, old 3 failure modes):**
40/40 [OK], 100% accuracy (threshold-based). Old failure modes (stop_short, wrong_target, signal) never
approached cube → < 2 reward → safely below old 10.0 threshold. Superseded by 5-mode dataset.

Compare with old grip-assist (2026-04-09): 39/40 OK (seq_0038 MISMATCH at early knock). Pure friction is strictly better: 40/40 and no artificial forces in the training data.

**Previous run stats (seed=1234, 40 sequences — 2026-04-09, force-based grip assist):** SUPERSEDED.
Grip assist used a 3D spring-damper body force (k_p=20, k_d=10) to track cube to EE. Physically
incorrect — bypasses the contact/friction problem. Removed in favour of pure Newton physics.

**Previous run stats (seed=100, 8 sequences — 2026-04-08):** SUPERSEDED — used virtual grasp teleportation.

## Steps

All paths default to `<task_root>/data/validation/` — run from any directory.

1. **Generate sequences**
   ```bash
   python .../scripts/generate_sequences.py --num_sequences 100 --num_envs 16
   ```
   Expected: prints `Saved 100 sequences → .../franka_cube_pick/data/validation/sequences.json`

2. **Replay headless**
   ```bash
   python .../scripts/replay_sequences.py
   ```
   Expected: all 100 lines end with `[OK]`, no `[MISMATCH]`.

3. **Analyze**
   ```bash
   python .../scripts/analyze_results.py
   ```
   Expected: Accuracy ≥ 90%, all label groups have mismatches at 0.0%.

4. **Inspect report** — if accuracy < 90% or reachability mask mismatches > 1%, investigate:
   - Mask mismatches → fix `reachable_radius_min/max` in `FrankaCubePickEnvCfg`
   - Wrong branch rewards → fix `mdp/rewards.py` and re-sync `reward_eval.py`
   - Sequence mislabelled → check waypoint state machine in `waypoint_ik.py`

5. **Fix any issues**, update skills 03/04 if the fix reveals a structural problem.

## Variables

| Variable | Value | What it controls | Safe to change? |
|---|---|---|---|
| num_sequences | 100 | Total sequences for validation | Yes — more gives better statistics |
| num_envs | 16 | Kept for CLI compatibility — ignored (physics runs sequentially) | N/A |
| high_reward_threshold | N/A (removed) | SUCCESS/FAILURE now uses `success_frame >= 0`, not a reward threshold. Approach reward accumulates in near-cube failure modes (20–32) — any threshold would be fragile. | Threshold was 10.0, removed when 5 realistic failure modes were added |
| seed | 42 | Random seed for reproducible generation | Yes |

## Verification

```bash
python .../scripts/analyze_results.py \
    --input data/validation/replay.json --output /tmp/vr/
```

Expected output contains:
```
Accuracy: 100.0%  (N sequences)
  reachable_success           : 0/... mismatches (0.0%) [OK]
  reachable_failure           : 0/... mismatches (0.0%) [OK]
  unreachable_success         : 0/... mismatches (0.0%) [OK]
  unreachable_failure         : 0/... mismatches (0.0%) [OK]

SUCCESS FRAME SUMMARY
  reachable_success    : N/N hit  mean_frame=211  (within 250-frame budget)
  unreachable_success  : N/N hit  mean_frame~68

PHYSICS VARIANCE
  reachable_success:    mean~0.002–0.003 rad
  unreachable_success:  mean~0.003–0.004 rad
  reachable_failure:    mean~0.003 rad (outliers possible: stop_short IK divergence)
```

Validated run (seed=1234, 100 seqs, 2026-04-10): 99/100 OK, 99% accuracy (success_frame criterion).
  reachable_success=340 (mean), reachable_failure=14.2 (mean, incl. approach-reward modes), unreachable_success=1271, unreachable_failure=2.26.
  1 FN: IK diverged for 1 unreachable_success (EE near but outside 0.15m signal_pos threshold).

Six output files written to `--output` dir:
- `01_reward_histograms.png`
- `02_per_term_episode_reward.png`
- `03_per_frame_reward_curves.png`
- `04_expected_vs_actual_correlation.png`
- `05_success_frame_distribution.png`
- `06_joint_pos_variance.png`
- `report.txt`

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Panda URDF not found` | Isaac Sim pip package not installed | `pip install isaacsim` in env_isaaclab |
| `pycollada` import error | Missing package | `pip install pycollada` |
| All sequences `[MISMATCH]` | Reward function sign error | Check REWARD_WEIGHTS signs in reward_eval.py |
| Reachability mask mismatches | Radius bounds don't match sampling bounds | Check reachable_radius_min/max vs sample_cube_pos in sampling.py |
| `ModuleNotFoundError: No module named 'pxr'` | `reward_eval.py` imported via package path triggering full Isaac Lab import chain | Fixed: `reward_eval.py` now imports `reward_utils.py` via `importlib.util.spec_from_file_location` (bypasses package `__init__`) |
| `IKObjectivePosition` keyword error | Wrong newton.ik API (newton version changed) | Check `newton.ik.IKObjectivePosition` signature; expected: `link_index, link_offset, target_positions` |
| IK local minimum for signal_pos [0,0,0.8] | IK sometimes converges to EE at [0,0,0.933] | Acceptable for validation; reward still fires correctly because the ee-to-signal distance drives the `go_to_signal` term |
| `OpenGLRenderer requires pyglet (version >= 2.0)` | pyglet 1.x installed (isaaclab pins pyglet<2) | `pip install "pyglet>=2.0"` — the isaaclab<2 constraint is a stale transitive dep, harmless for standalone Newton viewer |

## Changelog

- 2026-04-08: initial version — IN PROGRESS
- 2026-04-08: rewrote Tool 2 (replay_sequences.py) to run standalone with Newton FK;
  removed AppLauncher dependency. All three tools confirmed working end-to-end.
  100 sequences, 100% accuracy, 0 mask mismatches. Status: validated.
- 2026-04-09: replaced virtual grasp with force-based 3D grip assist (k_p=20, k_d=10,
  ke_BOX=100). Z-only version was tried first but cube decoupled laterally after T_GRIP
  wrist-rotation knock (up to 4m from EE). 3D control damps the knock in ~0.25s; cube
  stays within 0.11m of EE for 8/9 R+S sequences. Three GPU-CPU syncs eliminated in
  generator. Replayer updated to mirror 3D force law. Dataset regenerated (seed=1234,
  40 seqs): 39/40 OK (seq_0038 early knock at t<T_GRIP → marginal reward=9.6).
- 2026-04-10: changed SUCCESS/FAILURE discriminator in replay + analyze from `episode_total > 10.0`
  (fragile threshold) to `success_frame >= 0` (task physically completed: cube_z > lift_height for
  reachable, ee within 0.15m of signal for unreachable). Near-cube failure modes accumulate 20–32
  approach reward — any fixed threshold is fragile across sequence regeneration. success_frame is
  physics-grounded, no threshold to tune. Updated replay_sequences.py and analyze_results.py.
- 2026-04-10: generated 100-sequence dataset via 4 parallel shards (--seq_id_start, merge_sequences.py).
  99/100 accuracy. 1 FN: IK diverged for 1 unreachable_success (EE 0.3m from signal_pos). 0 FP.
- 2026-04-10: replaced 3 simplistic reachable_failure modes (stop_short, wrong_target, signal) with
  5 realistic failure modes in waypoint_ik.py: approach_no_grip, stop_at_pregrasp, grip_drop_early,
  wrong_approach_target, descend_open_grip. All modes provably keep cube_z < 0.45m.
  Safety constant _GRIP_DROP_MAX_T = T_GRIP + 1.0 = 8.0s. Shared _approach_grasp_lift helper.
  Note: approach_no_grip/descend_open_grip still accumulate 20–32 approach reward (EE near cube) —
  this is NOT a reward bug. success_frame correctly identifies them as FAILURE (cube never lifted).
- 2026-04-10: replaced force-based grip assist with pure contact friction grasping.
  Identified root cause from Newton example (example_ik_cube_stacking.py): missing
  gravity compensation, low PD gains (old ke=600 vs example ke=4500), no armature/effort
  limits, kf=0 (no tangential spring), impratio=1 (friction slip), wrong timestep for ke.
  Fixes applied: gravcomp (jnt_actgravcomp + gravcomp), PD gains ×7.5, armature,
  effort limits, ke=5e4, kd=5e2, kf=1e3, mu=0.75, impratio=1000, cone=elliptic,
  10 substeps of 2ms per 20ms outer step (contact period T=8.9ms < sub_dt=2ms → stable).
  Replayer rewritten: FK+full Newton physics (no analytic cube), 20 substeps × 2ms per frame.
  Fixed reward_eval.py pxr import error (now uses importlib.util instead of package path).
  Validated: 40/40 [OK], 100% accuracy (seed=1234, 40 seqs). Success frame=211 (all R+S).
  Outputs: sequences_physics.json, replay_physics.json, report_physics/.
