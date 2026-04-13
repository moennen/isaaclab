# IsaacLab Franka Cube Pick — Agent Skills

## Project Requirements

Read [../PRD.md](../PRD.md) first. It defines what is being built, why, and records all
design decisions made during the project. Skills describe *how* to build what the PRD specifies.

## What This Project Does

Creates a manager-based reinforcement learning task in Isaac Lab where a Franka
Panda arm must either pick up a rigid cube from the ground (if reachable) or
move to a designated signal position (if the cube is unreachable). The project
also produces this agent-skills set so future agents can create new tasks using
the same pattern.

## Parent Skill Set

None — this is an original project.

## Skill Execution Order

Execute skills in this sequence on a clean environment:

1. [01-infrastructure.md](01-infrastructure.md) — Install OS deps, micromamba, create Python 3.11 env
2. [02-isaaclab-setup.md](02-isaaclab-setup.md) — Install Isaac Sim + Isaac Lab via pip inside the env
3. [03-task-structure.md](03-task-structure.md) — Understand and create the experimental task package structure
4. [04-task-domain.md](04-task-domain.md) — Task-specific design: geometry, rewards, observations
5. [05-validation-workflow.md](05-validation-workflow.md) — Validate physics, rewards and observations before RL
6. [06-newton-backend-pitfalls.md](06-newton-backend-pitfalls.md) — Newton backend integration pitfalls: 6 bug classes with fixes
7. [07-vbd-task-pattern.md](07-vbd-task-pattern.md) — VBD soft-body task pattern: franka_vbd_cube_pick reference implementation

## Skill Provenance

| Skill | Status |
|---|---|
| 01-infrastructure.md | original |
| 02-isaaclab-setup.md | original |
| 03-task-structure.md | original |
| 04-task-domain.md | original — positions/rewards validated; preset system + CuboidCfg updated 2026-04-12 |
| 05-validation-workflow.md | original — COMPLETE 2026-04-08 (100% accuracy, 0 mask mismatches, all standalone) |
| 06-newton-backend-pitfalls.md | original — documents 6 bug classes hit and fixed 2026-04-13 during Newton training bring-up |
| 07-vbd-task-pattern.md | original — VBD task pattern from franka_vbd_cube_pick implementation 2026-04-13 |

## Key Project Files

| File | Role |
|---|---|
| `reward_utils.py` | **Single source of truth** for all reward math — pure torch, zero Isaac Lab imports. Imported by `mdp/rewards.py` (RL env), `scripts/_common/reward_eval.py` (tools), and `tests/`. Any reward change goes here and propagates everywhere. |
| `mdp/rewards.py` | Env-wrapping layer — extracts tensors from Isaac Lab env objects, calls `reward_utils`. No math lives here. |
| `scripts/_common/reward_eval.py` | Thin re-export of `reward_utils` for standalone tools and tests. |
| `scripts/generate_sequences.py` | Tool 1 — Newton IK standalone sequence generator |
| `scripts/replay_sequences.py` | Tool 2 — Newton FK standalone replayer + reward evaluator |
| `scripts/analyze_results.py` | Tool 3 — statistical report generator (pure Python) |
| `scripts/visualize_sequence.py` | Tool 4 — Newton ViewerGL interactive visualizer |

## Related Task: franka_vbd_cube_pick

The `../franka_vbd_cube_pick/` task extends this task to use a VBD deformable cube.
Key differences: no `scene.object`, cube managed by `FrankaVbdCubePickNewtonManager`,
cube pose read from manager obs-cache. See [07-vbd-task-pattern.md](07-vbd-task-pattern.md).

## Key Project Decisions

- Task lives in `source/isaaclab_tasks/` (main `isaaclab_tasks` package — `experimental` was removed)
- **Physics backend via `PresetCfg`**: single gym ID `Isaac-Pick-Cube-Franka-v0`; select Newton with `presets=newton`. No separate `-Newton-v0` task. `launch_simulation()` auto-detects `NewtonCfg` — no `--experience` flag needed.
- No table — cube spawns directly on the ground plane (z=0)
- Reachability is computed geometrically inside reward functions; the robot is NOT given it as an explicit observation
- Two reward branches (reachable / unreachable) are gated by a float mask so gradients don't mix
- Observation space is dexsuite-style (47 dims, state-based, no FrameTransformer — Newton-compatible)
- EE state reads from `robot.data.body_link_state_w` (panda_hand link) not from a FrameTransformer sensor
- Physics and rewards are validated with scripted sequences BEFORE RL training begins
- All geometry constants (radii, target positions) live on `FrankaCubePickEnvCfg` so derived configs can override without touching reward code
- **Shared simulation code**: reward kernels in `reward_utils.py` are used by RL training, validation tools, and unit tests — so tools validate the exact same code that runs during training
- **Cube spawner**: `CuboidCfg(size=(0.05,0.05,0.05))` — `UsdFileCfg` has no `physics_material` field so friction + ke/kd cannot be set via constructor. Shape spawners natively support `physics_material=RigidBodyMaterialCfg(...)`.
