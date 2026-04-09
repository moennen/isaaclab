---
name: isaaclab-experimental-task-structure
description: Create a new manager-based task in the isaaclab_tasks_experimental package, following the MISSING-pattern and two-level config hierarchy used in isaaclab_tasks.
level: 3
status: draft
depends_on: [isaaclab-setup]
extends: null
---

## Preconditions

- Isaac Lab installed and importable (skill 02 complete)
- Familiarity with `isaaclab_tasks/manager_based/manipulation/lift/` as the canonical reference

## Context

New experimental tasks live in `source/isaaclab_tasks_experimental/` — a separate pip package
from the core `isaaclab_tasks`. This keeps experimental work isolated from the mainline task
registry and avoids polluting the upstream package.

**Two-level config hierarchy (the MISSING pattern):**

```
franka_cube_pick_env_cfg.py         ← Level 1: abstract base
  robot: ArticulationCfg = MISSING  ← filled by level 2
  ee_frame: ...         = MISSING
  object: ...           = MISSING

config/franka/joint_pos_env_cfg.py  ← Level 2: robot-specific
  class FrankaCubePickEnvCfg(base):
    def __post_init__(self):
        self.scene.robot = FRANKA_PANDA_CFG  ← fills MISSING
```

This pattern allows new robots to be added by creating a new `config/<robot>/` directory
without touching the base env or reward logic.

**Physics-extension pattern:**
The base env cfg exposes geometry constants as class attributes (`reachable_radius_min`,
`success_ee_position`, etc.). Derived configs for different physics implementations
(e.g. deformable cube, different friction) override only those constants.

## Required File Structure

```
source/isaaclab_tasks_experimental/
  pyproject.toml
  config/
    extension.toml
  isaaclab_tasks_experimental/
    __init__.py                    ← calls import_packages() for auto-registration
    manager_based/
      __init__.py
      manipulation/
        __init__.py
        <task_name>/
          __init__.py              ← empty or docstring only
          <task_name>_env_cfg.py  ← base config (MISSING pattern)
          mdp/
            __init__.py           ← re-exports standard + task-specific mdp symbols
            observations.py       ← custom obs functions
            rewards.py            ← custom reward functions
            terminations.py       ← custom termination functions (may be empty)
            events.py             ← custom event functions (may be empty)
          config/
            __init__.py
            <robot>/
              __init__.py         ← gym.register() calls
              joint_pos_env_cfg.py
              agents/
                __init__.py
                rsl_rl_ppo_cfg.py  ← (add when ready to train)
```

## Steps

1. **Create all directories**
   ```bash
   mkdir -p source/isaaclab_tasks_experimental/config
   mkdir -p source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/manager_based/manipulation/<task_name>/mdp
   mkdir -p source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/manager_based/manipulation/<task_name>/config/<robot>/agents
   ```

2. **Create `pyproject.toml`** — copy from lift task, change name to `isaaclab_tasks_experimental`

3. **Create `config/extension.toml`** — set version, title, add `isaaclab_tasks` to dependencies

4. **Create `isaaclab_tasks_experimental/__init__.py`** — import `import_packages` from
   `isaaclab_tasks.utils` and call it with `_BLACKLIST_PKGS = ["utils", ".mdp"]`

5. **Create the base env cfg** (`<task_name>_env_cfg.py`):
   - Scene config with `robot`, `ee_frame`, `object` as `MISSING`
   - Ground plane and light (no table for ground-level tasks)
   - All MDP config classes (Actions, Observations, Events, Rewards, Terminations)
   - Top-level env cfg class with geometry constants as typed class attributes

6. **Create `mdp/__init__.py`** — re-export all standard mdp symbols needed by the env cfg,
   plus the custom symbols from observations.py and rewards.py

7. **Create `config/<robot>/__init__.py`** — `gym.register()` calls for TRAIN and PLAY variants

8. **Create `config/<robot>/joint_pos_env_cfg.py`** — fills in MISSING fields

9. **Install the package in editable mode**
   ```bash
   cd source/isaaclab_tasks_experimental
   micromamba run -n env_isaaclab pip install -e .
   ```

## Variables

| Variable | Value in this project | What it controls | Safe to change? |
|---|---|---|---|
| TASK_NAME | franka_cube_pick | Directory name, module name | Yes — rename consistently |
| ROBOT_NAME | franka | Config subdirectory | Yes — add new robot dirs alongside |
| GYM_ID | Isaac-FrankaCubePick-v0 | Gym registration ID | Yes — must be globally unique |

## Verification

```bash
micromamba run -n env_isaaclab python -c "
import gymnasium as gym
import isaaclab_tasks_experimental
env_spec = gym.spec('Isaac-FrankaCubePick-v0')
print('Registered:', env_spec.id)
"
```
Expected: `Registered: Isaac-FrankaCubePick-v0`

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: Isaac-FrankaCubePick-v0` | Package not installed or `__init__.py` not calling `import_packages` | Run `pip install -e .` in the package dir; check `__init__.py` |
| `ImportError` in mdp `__init__.py` | A symbol imported from `isaaclab.envs.mdp` doesn't exist | Check exact symbol name in the isaaclab version installed |
| `MISSING` field error at env init | Robot-specific config's `__post_init__` not calling `super().__post_init__()` | Add the super() call |

## Changelog

- 2026-04-08: initial version
