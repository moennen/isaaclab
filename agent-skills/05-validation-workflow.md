---
name: validation-workflow
description: Validate physics, rewards, and observations using scripted sequences before RL training. Three tools in order — physics check, reward check, observation check.
level: 4
status: draft
depends_on: [franka-cube-pick-domain]
extends: null
---

## Preconditions

- Task importable and geometry constants validated (skills 03 + 04 complete)
- `env_isaaclab` activated

## Context

Training an RL policy is expensive and error-prone. Before any training run, three
categories of bugs must be ruled out with scripted (non-learned) sequences:

1. **Physics bugs** — collisions wrong, cube falls through floor, robot self-collides
2. **Reward bugs** — reward fires on wrong branch, wrong sign, never fires, always fires
3. **Observation bugs** — wrong frame, wrong scale, NaN, not updating

Each category has a dedicated validation tool (scripts in `scripts/validation/`).
Tools are run in order — physics first, then rewards, then observations.

**Scripted sequences used for validation:**

| Sequence | Expected branch | Expected total reward |
|---|---|---|
| Cube at [0.45, 0.0] → EE reaches cube → lifts → goes to success pos | Reachable | High (all 3 reachable terms fire) |
| Cube at [0.45, 0.0] → robot stays at home | Reachable | Low (only approach_cube partially) |
| Cube at [0.80, 0.0] → EE goes to signal pos | Unreachable | High (signal term fires) |
| Cube at [0.80, 0.0] → robot stays at home | Unreachable | Low (wrong branch) |

## Toolchain

Three tools in `scripts/validation/franka_cube_pick/`, sharing utilities in `_common/`:

```
scripts/validation/franka_cube_pick/
  generate_sequences.py     # Tool 1 — generates N labeled sequences via IK
  replay_sequences.py       # Tool 2 — replays sequences, computes per-frame rewards
  analyze_results.py        # Tool 3 — reads replay JSON, produces report + plots
  _common/
    sequence_schema.py      # JSON schema, I/O helpers
    reward_eval.py          # Standalone per-frame reward computation (no env object)
    waypoint_ik.py          # IK waypoint state machine for all 4 scenario types
```

### Tool 1 — generate_sequences.py

Generates N simulation sequences using DifferentialIK to follow per-scenario waypoints.

```bash
./isaaclab.sh -p scripts/validation/franka_cube_pick/generate_sequences.py \
    --num_sequences 100 \
    --output outputs/validation/sequences.json \
    --num_envs 16
```

**Four scenario types** (generated according to --reachable_ratio=0.7 and --success_ratio=0.5):

| Label | Robot behaviour |
|---|---|
| reachable_success | approach → grasp → lift to 0.5m → success_pos |
| reachable_failure | goes to signal_pos instead (wrong) |
| unreachable_success | goes to signal_pos (correct) |
| unreachable_failure | tries to approach unreachable cube (wrong) |

Output JSON contains per-frame: `joint_pos` (9 values), `gripper_closed`, `ee_pos_w`, `cube_pos_w`.

### Tool 2 — replay_sequences.py

Replays recorded joint positions, re-runs physics, computes per-frame rewards via `reward_eval.py`.

```bash
# Headless — all sequences
./isaaclab.sh -p scripts/validation/franka_cube_pick/replay_sequences.py \
    --input  outputs/validation/sequences.json \
    --output outputs/validation/replay.json

# Visualize a single sequence
./isaaclab.sh -p scripts/validation/franka_cube_pick/replay_sequences.py \
    --input  outputs/validation/sequences.json \
    --output outputs/validation/replay.json \
    --visualize --sequence_id seq_0003
```

Prints per-frame: label, t, total reward, reachable_mask, cube_z.
Prints per-sequence: episode total reward vs expected HIGH/LOW → MATCH/MISMATCH.

### Tool 3 — analyze_results.py

Pure Python (no Isaac Sim), reads replay JSON, writes 4 PNG figures + `report.txt`.

```bash
python scripts/validation/franka_cube_pick/analyze_results.py \
    --input  outputs/validation/replay.json \
    --output outputs/validation/report/ \
    --show
```

**Outputs:**
- `01_reward_histograms.png` — episode total reward distribution per label
- `02_per_term_episode_reward.png` — per-term weighted reward bar chart by label
- `03_per_frame_reward_curves.png` — mean per-term reward curves over time per label
- `04_expected_vs_actual_correlation.png` — expected vs actual reward scatter
- `report.txt` — summary table, confusion matrix, reachability mask consistency check

### Reward function (standalone, in `_common/reward_eval.py`)

This is the ground-truth reward implementation used for validation. It mirrors
`mdp/rewards.py` exactly but takes plain tensors (no env object) so it can run
independently of the IsaacLab environment manager.

| Term | Branch | Formula | Weight |
|---|---|---|---|
| approach_cube_reachable | reachable | mask × (1 - tanh(‖ee - cube‖ / 0.1)) | 1.0 |
| lift_cube_reachable | reachable | mask × indicator(cube_z > 0.5) | 10.0 |
| cube_at_success_position | reachable | mask × lifted × (1 - tanh(‖ee - success_pos‖ / 0.1)) | 15.0 |
| go_to_signal_position | unreachable | (1-mask) × (1 - tanh(‖ee - signal_pos‖ / 0.1)) | 10.0 |
| action_rate | always | -‖Δjoint_pos‖² | -1e-4 |
| joint_vel | always | -‖joint_vel‖² | -1e-4 |

**Expected episode reward ranges** (used by Tool 3 threshold and Tool 2 MATCH/MISMATCH):

| Label | Expected | Reasoning |
|---|---|---|
| reachable_success | HIGH (>> 10) | All 3 reachable terms fire in sequence |
| reachable_failure | LOW (~0) | Reachable branch terms never fire; goes to wrong pos |
| unreachable_success | HIGH (>> 10) | signal term fires for most of episode |
| unreachable_failure | LOW (~0) | Neither branch fires correctly |

## Steps

1. **Generate sequences**
   ```bash
   ./isaaclab.sh -p scripts/validation/franka_cube_pick/generate_sequences.py \
       --num_sequences 100 --output outputs/validation/sequences.json
   ```

2. **Replay headless**
   ```bash
   ./isaaclab.sh -p scripts/validation/franka_cube_pick/replay_sequences.py \
       --input outputs/validation/sequences.json \
       --output outputs/validation/replay.json
   ```

3. **Analyze**
   ```bash
   python scripts/validation/franka_cube_pick/analyze_results.py \
       --input outputs/validation/replay.json \
       --output outputs/validation/report/
   ```

4. **Inspect report** — if accuracy < 90% or reachability mask mismatches > 1%, investigate:
   - Mask mismatches → fix `reachable_radius_min/max` in `FrankaCubePickEnvCfg`
   - Wrong branch rewards → fix `mdp/rewards.py` and re-sync `reward_eval.py`
   - Sequence mislabelled → check waypoint state machine in `waypoint_ik.py`

5. **Fix any issues**, update skills 03/04 if the fix reveals a structural problem.

## Variables

| Variable | Value | Notes |
|---|---|---|
| VALIDATION_SCRIPTS_DIR | scripts/validation/ | Relative to IsaacLab repo root |
| NUM_ENVS_VALIDATION | 4 | Small — easier to inspect |
| PHYSICS_CHECK_STEPS | 100 | Enough to see settling |
| REWARD_CHECK_EPISODE_STEPS | 200 | ~2s at 100Hz with decimation=2 |

## Verification

All three tools exit with code 0 and print "PASS" for each check.

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Cube z drifts below 0 | Ground plane collision not set up | Check GroundPlaneCfg and cube rigid body props |
| Reward always 0 on reachable branch | `_is_reachable` returns wrong mask | Print cube XY distance and compare to thresholds |
| Observation cube position doesn't match world position | Wrong frame transform in `object_position_in_robot_root_frame` | Check robot root pos subtraction |
| NaN in observations | Uninitialized state on first step | Skip first step in validation or add warm-up |

## Changelog

- 2026-04-08: initial version — IN PROGRESS
