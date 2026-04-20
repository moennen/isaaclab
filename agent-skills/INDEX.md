# IsaacLab Franka Cube Pick — Agent Skills

## Project Requirements

Read [PRD.md](PRD.md) first. It defines what is being built, why, and records all
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

## Skill Provenance

| Skill | Status |
|---|---|
| 01-infrastructure.md | original |
| 02-isaaclab-setup.md | original |
| 03-task-structure.md | original |
| 04-task-domain.md | original — PENDING user validation of positions and rewards |
| 05-validation-workflow.md | original — IN PROGRESS |

## Key Project Decisions

- Task lives in `source/isaaclab_tasks_experimental/` (new package, not in core `isaaclab_tasks`)
- No table — cube spawns directly on the ground plane (z=0)
- Reachability is computed geometrically inside reward functions; the robot is NOT given it as an explicit observation
- Two reward branches (reachable / unreachable) are gated by a float mask so gradients don't mix
- Physics and rewards are validated with scripted sequences BEFORE RL training begins
- All geometry constants (radii, target positions) live on `FrankaCubePickEnvCfg` so derived configs can override without touching reward code
