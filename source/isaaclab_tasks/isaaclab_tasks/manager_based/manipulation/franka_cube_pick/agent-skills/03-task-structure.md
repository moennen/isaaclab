---
name: isaaclab-task-structure
description: Create a new manager-based RL task in the main isaaclab_tasks package, using the MISSING pattern, two-level config hierarchy, and PresetCfg for PhysX/Newton backend selection.
level: 3
status: approved
depends_on: [isaaclab-setup]
extends: null
---

## Preconditions

- Isaac Lab installed and importable (skill 02 complete)
- `isaaclab_tasks` package already installed (it is — it is part of the Isaac Lab repo)
- Reference task: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/` (canonical pattern)

## Context

### Where tasks live

New tasks go into `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/<task_name>/`.
This is the **main** `isaaclab_tasks` package, already installed editable by `isaaclab.sh --install`.
No separate `experimental` package or `pip install -e .` step needed.

The task is auto-discovered: when `isaaclab_tasks` is imported it calls `import_packages()`, which
recursively imports all `config/<robot>/__init__.py` files, which call `gym.register()`.

### Two-level config hierarchy (the MISSING pattern)

```
<task_name>_env_cfg.py         ← Level 1: abstract base
  scene.robot  = MISSING       ← filled by level 2
  scene.object = MISSING       ← filled by level 2
  actions.*    = MISSING       ← filled by level 2

config/franka/joint_pos_env_cfg.py   ← Level 2: robot-specific
  class FrankaCubePickEnvCfg(base):
    def __post_init__(self):
        super().__post_init__()      # MUST call super first
        self.scene.robot = FRANKA_PANDA_CFG.replace(...)
        self.scene.object = RigidObjectCfg(spawn=CuboidCfg(...))
        self.actions.arm_action = JointPositionActionCfg(...)
        self.actions.gripper_action = BinaryJointPositionActionCfg(...)
```

This lets new robots be added by creating a `config/<robot>/` directory without touching
the base env or reward logic.

### Preset system (PhysX / Newton backend selection)

Backend is selected at **CLI runtime**, not at task registration time. A single gym ID serves
both backends:

```bash
# PhysX (default):
micromamba run -n env_isaaclab python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cube-Franka-v0

# Newton:
micromamba run -n env_isaaclab python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cube-Franka-v0 presets=newton
```

The mechanism is `PresetCfg` + `preset()` from `isaaclab_tasks.utils`:

```python
from isaaclab_tasks.utils import PresetCfg, preset

@configclass
class FrankaCubePickPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg(...)
    physx:   PhysxCfg = default          # explicit PhysX target
    newton:  NewtonCfg = NewtonCfg(...)  # Newton target

class FrankaCubePickEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.decimation        = preset(default=2,    newton=10)
        self.sim.dt            = preset(default=0.01, newton=0.002)
        self.sim.render_interval = preset(default=2,  newton=10)
        self.sim.physics       = FrankaCubePickPhysicsCfg()
```

`launch_simulation()` auto-detects `NewtonCfg` in the config tree — no `--experience` flag needed.
Both backends maintain 50 Hz control: PhysX 2×10ms, Newton 10×2ms.

### Cube spawner: use CuboidCfg, not UsdFileCfg

`UsdFileCfg` has no `physics_material` field — `spawn_from_usd()` does not apply material
overrides. To set friction, restitution, Newton ke/kd, use `CuboidCfg` (a `ShapeCfg` subclass):

```python
spawn=CuboidCfg(
    size=(0.05, 0.05, 0.05),
    mass_props=MassPropertiesCfg(mass=0.1),
    physics_material=RigidBodyMaterialCfg(
        static_friction=0.75,
        dynamic_friction=0.75,
        restitution=0.0,
        compliant_contact_stiffness=5e4,  # Newton only — ignored by PhysX
        compliant_contact_damping=5e2,    # Newton only — ignored by PhysX
    ),
)
```

## Required File Structure

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/
  <task_name>/
    __init__.py                      ← empty or docstring only
    <task_name>_env_cfg.py           ← Level 1 base config (MISSING pattern)
                                        imports FrankaCubePickPhysicsCfg (PresetCfg)
    reward_utils.py                  ← SINGLE SOURCE OF TRUTH for reward math
                                        pure torch, zero Isaac Lab imports
    mdp/
      __init__.py                    ← re-exports standard + task-specific symbols
      observations.py                ← custom obs functions (use wp.to_torch())
      rewards.py                     ← env-wrapping layer, calls reward_utils
      terminations.py                ← may be empty / time_out only
      events.py                      ← reset events (may be empty)
    config/
      __init__.py                    ← empty
      <robot>/
        __init__.py                  ← gym.register() calls (TRAIN + PLAY)
        joint_pos_env_cfg.py         ← Level 2: fills MISSING fields
        agents/
          __init__.py                ← empty
          rsl_rl_ppo_cfg.py          ← PPO runner config
```

No `pyproject.toml` needed — the task is part of the existing `isaaclab_tasks` package.

## Steps

### 1. Create directories

```bash
TASK=<task_name>
ROBOT=franka
BASE=source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation
mkdir -p $BASE/$TASK/mdp
mkdir -p $BASE/$TASK/config/$ROBOT/agents
```

### 2. Create `<task_name>/__init__.py`

Empty or one-line docstring. No imports needed.

### 3. Create the base env cfg (`<task_name>_env_cfg.py`)

Key sections in order:

```python
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg, preset
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from . import mdp

# --- Physics preset ---
@configclass
class MyTaskPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg(...)
    physx:   PhysxCfg = default
    newton:  NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton", integrator="implicitfast",
            iterations=20, ls_parallel=True, ls_iterations=100,
            cone="elliptic", impratio=1000.0,
            njmax=150, nconmax=40, use_mujoco_contacts=True,
        ),
        num_substeps=10,
        use_cuda_graph=True,
    )

# --- Scene ---
@configclass
class GroundSceneCfg(InteractiveSceneCfg):
    robot:  ArticulationCfg = MISSING
    object: RigidObjectCfg  = MISSING
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(...))

# --- MDP sections ---
@configclass
class ActionsCfg:
    arm_action:     mdp.JointPositionActionCfg          = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg    = MISSING

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ...
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

# EventCfg, RewardsCfg, TerminationsCfg ...

# --- Top-level env cfg ---
@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    # Geometry constants — override in derived configs without touching reward code
    reachable_radius_min: float = 0.22
    ...

    scene:        GroundSceneCfg  = GroundSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()

    def __post_init__(self):
        self.decimation          = preset(default=2,    newton=10)
        self.episode_length_s    = 8.0
        self.sim.dt              = preset(default=0.01, newton=0.002)
        self.sim.render_interval = preset(default=2,    newton=10)
        self.sim.physics         = MyTaskPhysicsCfg()
```

### 4. Create `mdp/__init__.py`

Re-export all standard mdp symbols the env cfg uses, plus custom task symbols:

```python
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg, JointPositionActionCfg,
    action_rate_l2, joint_pos_rel, joint_vel_l2, joint_vel_rel, last_action,
    reset_root_state_uniform, reset_scene_to_default,
    root_height_below_minimum, time_out,
)
from .observations import cube_pos_b, cube_quat_b, ee_state_b
from .rewards import (approach_cube_reachable, ...)
from ..reward_utils import REWARD_WEIGHTS, compute_all_rewards
```

### 5. Create `mdp/observations.py`

All data properties (`robot.data.root_pos_w`, `cube.data.root_pos_w`, etc.) return **warp
arrays** in the Newton backend. Convert with `wp.to_torch()` before passing to Isaac Lab
torch-based math utilities (`quat_inv`, `subtract_frame_transforms`, etc.):

```python
import warp as wp
from isaaclab.utils.math import subtract_frame_transforms

def cube_pos_b(env, robot_cfg, object_cfg):
    robot_pos_w  = wp.to_torch(robot.data.root_pos_w)   # (N, 3)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)  # (N, 4)
    cube_pos_w   = wp.to_torch(cube.data.root_pos_w)    # (N, 3)
    pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, cube_pos_w)
    return pos_b
```

For body-level data: `wp.to_torch(robot.data.body_link_pos_w)` gives `(N, B, 3)`;
`wp.to_torch(robot.data.body_link_state_w)` gives `(N, B, 13)`.

**Rule:** call `wp.to_torch()` on every `asset.data.*` access before using it in torch
operations. This is a no-copy GPU reinterpret — safe to call on every step.

**Why needed:** PhysX also stores data as warp arrays internally, but standard `isaaclab.envs.mdp`
functions already call `wp.to_torch()` (see `joint_pos_rel`). Custom task functions must do the
same. Omitting it works on neither backend because the type mismatch is caught immediately.

### 6. Create `mdp/rewards.py`

Keep it thin — extract tensors from env, call `reward_utils.py`:

```python
import warp as wp

def _get_tensors(env, object_cfg, ee_cfg=None):
    cube_pos_w  = wp.to_torch(cube.data.root_pos_w)
    robot_pos_w = wp.to_torch(robot.data.root_pos_w)
    ee_pos_w    = wp.to_torch(robot.data.body_link_pos_w)[:, ee_cfg.body_ids[0], :]
    return cube_pos_w, robot_pos_w, ee_pos_w
```

`joint_pos` access: `wp.to_torch(robot.data.joint_pos)[:, idx]`.

### 7. Create `reward_utils.py`

Pure-tensor reward kernels — **no Isaac Lab imports**. Used by both:
- `mdp/rewards.py` (RL env) 
- validation tools (`scripts/_common/reward_eval.py`)

This ensures the same code is validated before training and executed during training.

### 8. Create `config/<robot>/__init__.py`

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Pick-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-Pick-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)
```

### 9. Create `config/<robot>/joint_pos_env_cfg.py`

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.<task_name>.<task_name>_env_cfg import <TaskEnvCfg>

@configclass
class FrankaCubePickEnvCfg(<TaskEnvCfg>):
    def __post_init__(self):
        super().__post_init__()  # MUST call super — resolves presets
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True,
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.025], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.05, 0.05, 0.05),
                mass_props=MassPropertiesCfg(mass=0.1),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=0.75, dynamic_friction=0.75, restitution=0.0,
                    compliant_contact_stiffness=5e4, compliant_contact_damping=5e2,
                ),
            ),
        )

@configclass
class FrankaCubePickEnvCfg_PLAY(FrankaCubePickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
```

### 10. Create RSL-RL PPO config

Copy `config/franka/agents/rsl_rl_ppo_cfg.py` from the lift task and adjust
`experiment_name` and network sizes for the new task's observation dimensions.

## Variables

| Variable | Value in this project | What it controls | Safe to change? |
|---|---|---|---|
| TASK_NAME | franka_cube_pick | Directory and module name | Yes — rename consistently |
| ROBOT_NAME | franka | Config subdirectory | Yes — add new robot dirs alongside |
| GYM_ID | Isaac-Pick-Cube-Franka-v0 | Gym registration ID | Yes — must be globally unique |
| NUM_ENVS | 4096 | Default training envs | Yes |
| ENV_SPACING | 2.5 m | Space between envs in scene | Yes |

## Verification

```bash
# 1. Task registers correctly
micromamba run -n env_isaaclab python -c "
import gymnasium as gym
import isaaclab_tasks
env_spec = gym.spec('Isaac-Pick-Cube-Franka-v0')
print('Registered:', env_spec.id)
"

# 2. Config resolves with Newton preset
micromamba run -n env_isaaclab python -c "
from isaaclab_tasks.utils.hydra import resolve_task_config
cfg = resolve_task_config('Isaac-Pick-Cube-Franka-v0', presets=['newton'])
import isaaclab_newton.physics
assert isinstance(cfg.sim.physics, isaaclab_newton.physics.NewtonCfg)
print('Newton preset OK:', cfg.decimation, cfg.sim.dt)
"
# Expected: Newton preset OK: 10 0.002
```

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: Isaac-Pick-Cube-Franka-v0` | Task not auto-discovered | Ensure `config/<robot>/__init__.py` exists with `gym.register()` |
| `MISSING` field error at env init | `__post_init__` not calling `super().__post_init__()` | Add the `super()` call as first line |
| `quat_inv() Expected Tensor, got array` | Missing `wp.to_torch()` in obs/reward functions | See skill 06 |
| `TypeError: UsdFileCfg.__init__() got an unexpected keyword argument 'physics_material'` | Using `UsdFileCfg` for cube | Replace with `CuboidCfg` — see Context above |
| Newton backend crashes at init | `isaaclab_newton` API bugs | See skill 06 for all known bugs + fixes |
| `presets=newton` has no effect | Missing `FrankaCubePickPhysicsCfg(PresetCfg)` or `preset()` calls | Check `__post_init__` uses `preset(default=..., newton=...)` |

## Changelog

- 2026-04-08: initial version (described isaaclab_tasks_experimental — incorrect)
- 2026-04-12: complete rewrite — task lives in main isaaclab_tasks, preset system added,
  CuboidCfg cube spawner, wp.to_torch() requirement, correct gym IDs, no separate pip install
