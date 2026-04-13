# Skill 06 — Newton Backend Integration Pitfalls

Reference for bugs reliably encountered when wiring a new task to Isaac Lab's
Newton (MuJoCo-Warp) backend. Each entry lists: what breaks, why, and the fix.
All bugs below were hit and fixed while running `presets=newton --num_envs 4096`.

---

## 0. Overview: how data flows through the Newton backend

```
Newton solver (warp arrays)
        │
        ▼
ArticulationData / RigidObjectData  ← sim_bind_* attributes (warp, 2D)
        │  wp.to_torch()
        ▼
Task mdp/observations.py + mdp/rewards.py  (torch tensors)
        │
        ▼
Isaac Lab math utilities (subtract_frame_transforms, quat_rotate_inverse, …)
```

Every property exposed by `asset.data.*` is a **warp array** in the Newton
backend. Passing a warp array directly to torch functions raises
`RuntimeError: expected Tensor, got warp.array`. Always call `wp.to_torch()`
first.

---

## Bug class 1 — `[:, 0]` stripping on `get_root_transforms()`

**Where**: `articulation_data.py` `_create_simulation_bindings()`,
`rigid_object_data.py` `_create_simulation_bindings()`.

**Symptom**: `IndexError: tuple index out of range` on the first Newton step.

**Root cause**: `get_root_transforms(state)` returns a 1D warp array of shape
`(N,)` where each element is a `wp.transform` (7-float struct). Someone added
`[:, 0]` thinking this was a 2D `(N, 1)` array — it is not.

**Fix**: remove `[:, 0]` entirely.

```python
# WRONG — crashes with IndexError
self._sim_bind_root_link_pose_w = (
    self._root_view.get_root_transforms(SimulationManager.get_state_0())[:, 0]
)

# CORRECT
self._sim_bind_root_link_pose_w = (
    self._root_view.get_root_transforms(SimulationManager.get_state_0())
)
```

The same pattern appears for `rigid_object_data.py` with an additional
fixed/floating-base branch that used `[:, 0, 0]` for fixed-base — also wrong.
Collapse both branches to a single call without slicing.

---

## Bug class 2 — `[:, 0]` stripping on body / joint attribute bindings

**Where**: `articulation_data.py` and `rigid_object_data.py`,
`_create_simulation_bindings()`, every `get_attribute(...)`,
`get_link_transforms()`, `get_dof_positions()`, `get_link_velocities()`, etc.

**Symptom**: Cascading `RuntimeError` — kernels that expect 2D `(N, B)` or
`(N, J)` inputs receive 1D `(N,)` arrays.

Examples seen:
- `RuntimeError: joint_pos_limits_lower expects 2D, got 1D`
- `RuntimeError: prev_joint_vel expects 2D, got 1D`
- `IndexError: strides[1] on 1D array` (body_inertia)

**Root cause**: All body/joint attribute getters return 2D arrays `(N, B)` or
`(N, J)`. The `[:, 0]` was applied uniformly — correct for the old API, broken
for the current Newton API.

**Fix**: remove `[:, 0]` from **all** attribute bindings, not just the ones
that crash first. There are ~15 of them. Patch them all at once.

```python
# WRONG — strips the body/joint dimension
self._sim_bind_body_link_pose_w = (
    self._root_view.get_link_transforms(SimulationManager.get_state_0())[:, 0]
)

# CORRECT
self._sim_bind_body_link_pose_w = (
    self._root_view.get_link_transforms(SimulationManager.get_state_0())
)
```

Same rule for `_previous_joint_vel` / `_previous_body_com_vel` initializations
in `_create_buffers()`:

```python
# WRONG
self._previous_joint_vel = wp.clone(
    self._root_view.get_dof_velocities(SimulationManager.get_state_0())[:, 0]
)

# CORRECT
self._previous_joint_vel = wp.clone(
    self._root_view.get_dof_velocities(SimulationManager.get_state_0())
)
```

---

## Bug class 3 — `link_names` does not exist

**Where**: `articulation.py` `body_names` property,
`rigid_object.py` `body_names` property.

**Symptom**: `AttributeError: 'ArticulationView' object has no attribute 'link_names'`

**Root cause**: The Newton `ArticulationView` / `RigidObjectView` exposes
`body_names`, not `link_names`.

**Fix**:

```python
# WRONG
return self.root_view.link_names

# CORRECT
return self.root_view.body_names
```

---

## Bug class 4 — `get_max_contact_count()` does not exist

**Where**: `newton_manager.py` solver initialization.

**Symptom**: `AttributeError: 'SolverMuJoCo' object has no attribute 'get_max_contact_count'`

**Root cause**: The method was renamed / never existed in the current Newton
API. The max contact count lives on `mjw_data`.

**Fix**:

```python
# WRONG
rigid_contact_max = cls._solver.get_max_contact_count()

# CORRECT
rigid_contact_max = cls._solver.mjw_data.naconmax
```

---

## Bug class 5 — `Contacts()` has no `requested_attributes` parameter

**Where**: `newton_manager.py`, the `Contacts(...)` constructor call right
after the `get_max_contact_count` line.

**Symptom**: `TypeError: Contacts.__init__() got an unexpected keyword argument 'requested_attributes'`

**Root cause**: The `Contacts` dataclass signature is
`Contacts(rigid_contact_max, soft_contact_max, device)` — no extra attributes.

**Fix**:

```python
# WRONG
cls._contacts = Contacts(
    rigid_contact_max=rigid_contact_max,
    soft_contact_max=0,
    device=PhysicsManager._device,
    requested_attributes=cls._model.get_requested_contact_attributes(),
)

# CORRECT
cls._contacts = Contacts(
    rigid_contact_max=rigid_contact_max,
    soft_contact_max=0,
    device=PhysicsManager._device,
)
```

---

## Bug class 6 — warp arrays not converted to torch in task mdp functions

**Where**: `mdp/observations.py`, `mdp/rewards.py` — any custom function that
reads from `asset.data.*` and passes the result to torch math utilities.

**Symptom**: `RuntimeError: quat_inv() Expected Tensor, got <class 'warp.types.array'>`
(or similar for any torch operation on a warp array).

**Root cause**: In the Newton backend, all `asset.data.*` properties return
warp arrays, not torch tensors. The standard Isaac Lab mdp functions (e.g.,
`joint_pos_rel` in `isaaclab.envs.mdp`) already call `wp.to_torch()` internally,
but custom task functions must do it themselves.

**Fix**: import warp and call `wp.to_torch()` on every data access before
passing to torch functions. `wp.to_torch()` is a zero-copy GPU reinterpret
(no data is moved).

```python
# mdp/observations.py
import warp as wp
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate_inverse

def cube_pos_b(env, robot_cfg, object_cfg):
    robot = env.scene[robot_cfg.name]
    cube  = env.scene[object_cfg.name]

    robot_pos_w  = wp.to_torch(robot.data.root_pos_w)    # (N, 3)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)   # (N, 4)
    cube_pos_w   = wp.to_torch(cube.data.root_pos_w)     # (N, 3)

    cube_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, cube_pos_w)
    return cube_pos_b


def ee_state_b(env, ee_cfg, robot_cfg):
    robot = env.scene[robot_cfg.name]

    robot_pos_w  = wp.to_torch(robot.data.root_pos_w)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)

    # body_link_state_w: warp (N, B) vec13f → torch (N, B, 13)
    ee_state_w  = wp.to_torch(robot.data.body_link_state_w)[:, ee_cfg.body_ids[0], :]
    ee_pos_w    = ee_state_w[:, :3]
    ee_quat_w   = ee_state_w[:, 3:7]
    ee_linvel_w = ee_state_w[:, 7:10]
    ee_angvel_w = ee_state_w[:, 10:13]

    ee_pos_b,  ee_quat_b  = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w)
    ee_linvel_b = quat_rotate_inverse(robot_quat_w, ee_linvel_w)
    ee_angvel_b = quat_rotate_inverse(robot_quat_w, ee_angvel_w)

    return torch.cat([ee_pos_b, ee_quat_b, ee_linvel_b, ee_angvel_b], dim=-1)
```

```python
# mdp/rewards.py — same pattern for every reward helper
import warp as wp

def _get_tensors(env, object_cfg, ee_cfg=None):
    robot = env.scene["robot"]
    cube  = env.scene[object_cfg.name]

    cube_pos_w  = wp.to_torch(cube.data.root_pos_w)
    robot_pos_w = wp.to_torch(robot.data.root_pos_w)

    ee_pos_w = None
    if ee_cfg is not None:
        # body_link_pos_w: warp (N, B) vec3f → torch (N, B, 3)
        ee_pos_w = wp.to_torch(robot.data.body_link_pos_w)[:, ee_cfg.body_ids[0], :]

    return cube_pos_w, robot_pos_w, ee_pos_w

def grip_cube_reachable(env, ...):
    # joint_pos: warp (N, J) → torch (N, J)
    joint_pos = wp.to_torch(robot.data.joint_pos)
    gripper_width = joint_pos[:, 7] + joint_pos[:, 8]
    ...
```

**Rule of thumb**: if a standard `isaaclab.envs.mdp` function does the same
thing (reads joint_pos, root state, etc.), check its source — it will show you
where `wp.to_torch()` is already called. Follow the same pattern in your custom
function.

---

## Verification checklist

After applying all fixes, confirm Newton training starts cleanly:

```bash
cd /path/to/IsaacLab
source ~/.bashrc  # activates env_isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cube-Franka-v0 presets=newton \
    --num_envs 64 --headless --max_iterations 5
```

Expected output: no `IndexError`, `AttributeError`, or `RuntimeError` before
iteration 1. Training loop should reach at least 3–5 iterations cleanly.

Scale to full env count only after this passes:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cube-Franka-v0 presets=newton \
    --num_envs 4096 --headless
```

---

## Files modified when fixing these bugs

All bugs are in `source/isaaclab_newton/` — the Newton backend package — plus
the task's own mdp layer:

| File | Bugs fixed |
|---|---|
| `isaaclab_newton/assets/articulation/articulation_data.py` | 1, 2 (root + all body/joint `[:, 0]` removals), 4 (prev_joint_vel) |
| `isaaclab_newton/assets/articulation/articulation.py` | 3 (`link_names` → `body_names`) |
| `isaaclab_newton/assets/rigid_object/rigid_object_data.py` | 1, 2 (same pattern for rigid objects), 4 (prev_body_com_vel) |
| `isaaclab_newton/assets/rigid_object/rigid_object.py` | 3 (`link_names` → `body_names`) |
| `isaaclab_newton/physics/newton_manager.py` | 4 (`naconmax`), 5 (`Contacts()` params) |
| `mdp/observations.py` | 6 (`wp.to_torch()` wrapping) |
| `mdp/rewards.py` | 6 (`wp.to_torch()` wrapping) |
