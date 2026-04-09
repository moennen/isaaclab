---
name: validation-workflow
description: Validate physics, rewards, and observations using scripted sequences before RL training. Four standalone tools — generator, replayer, analyzer, visualizer.
level: 4
status: draft
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

**Architecture decision — standalone replay:** The replayer (Tool 2) was originally
designed to run live Isaac Sim physics. Because the warp conflict prevents clean Isaac
Sim startup with Newton, Tool 2 was redesigned to run standalone using Newton FK:
- Recorded joint angles → Newton `eval_fk` → EE position in world frame
- Cube physics: analytic (cube XY fixed at init; Z follows EE Z after gripper closes)
- Reward computation: `reward_eval.py` (pure torch, no env object)

This is sufficient to validate the reward function against the scripted trajectory.
FK-derived EE positions are independent of the generator's IK, so IK errors would surface.

**Scripted sequences used for validation:**

| Label | Robot behaviour |
|---|---|
| reachable_success | approach → grasp → lift to 0.5m → success_pos |
| reachable_failure | goes to signal_pos instead (wrong) |
| unreachable_success | goes to signal_pos (correct) |
| unreachable_failure | tries to approach unreachable cube (wrong) |

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

Newton FK standalone: re-derives EE position from joint angles, analytic cube physics,
computes rewards via `reward_eval.compute_all_rewards`.

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

**Physics run stats (seed=1234, 40 sequences — 2026-04-09, force-based grip assist, 3D XYZ control):**

| Label | N | Episode reward | Success frame | Status |
|---|---|---|---|---|
| reachable_success | 9 | 9.6–308.3 | 210–212 | 8/9 HIGH, 1 MISMATCH (seq_0038 marginal) |
| reachable_failure | 15 | 0.3–1.9 | — | all LOW ✓ |
| unreachable_success | 9 | 1091.9–1526.9 | 34–98 | all HIGH ✓ |
| unreachable_failure | 7 | 0.5–0.7 | — | all LOW ✓ |

39/40 OK, 1 MISMATCH (seq_0038: early knock at t=4.96s before T_GRIP → marginal lift, reward=9.6 vs threshold 10.0).
Outputs: `data/validation/sequences_3d.json` + `replay_3d.json`.

**Physics correctness (cube-EE distance at episode end, R+S only):**

| Sequence | Distance | Notes |
|---|---|---|
| seq_0004–seq_0036, seq_0039 | 0.097–0.109m | 3D grip-assist tracking EE perfectly |
| seq_0038 | 0.261m | early knock at t=4.96s (before T_GRIP); pathological |

Compared to Z-only (old): seq_0039 was 4.06m, seq_0032 was 1.85m — 3D control fixes all.

**Grip-assist physics (reachable+success only):**
- After T_GRIP=7.0s, a 3D spring-damper body force tracks the cube to the EE position in XYZ.
- Parameters: k_p=20 N/m, k_d=10 N·s/m, gravity compensation F_z += m*g=1.226 N.
- Finger BOX contacts kept soft (ke_BOX=100 N/m) to prevent 200N+ lateral impulses from IK drift.
- Replayer mirrors generator exactly (same 3D force law per physics substep, using body_q[hand_idx][:3]).

**Previous run stats (seed=100, 8 sequences — 2026-04-08):** reachable_success ~1641, unreachable_success ~1735 (superseded — those used virtual grasp teleportation which was physically incorrect).

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
| num_envs | 16 | IK batch size per cycle (generator) | Yes — adjust for memory |
| high_reward_threshold | 10.0 | HIGH/LOW classification cutoff in analyzer | Yes if reward weights change; re-verify gap is > 2× |
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
  reachable_success    : N/N hit  mean_frame~180  (within 250-frame budget)
  unreachable_success  : N/N hit  mean_frame~50

PHYSICS VARIANCE
  all labels: mean~0.001–0.002 rad
```

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
