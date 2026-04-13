# Skill 07 — VBD Soft-Body Task Pattern

Reference for creating a new Newton+VBD task (deformable object physics) using
the `franka_vbd_cube_pick` task as the canonical implementation.

---

## 0. Overview

VBD tasks differ from rigid tasks in three ways:

1. **No scene asset for the object** — the deformable body is managed entirely
   by an extended `NewtonManager` subclass.  There is no `scene.object` entry
   in the Isaac Lab scene.
2. **Newton-only** — VBD requires the Newton (MuJoCo-Warp) backend.  Hard-code
   `self.decimation`, `self.sim.dt`, etc. without `preset()` since there is no
   PhysX equivalent.
3. **Shared physics module** — the two-way coupling kernel lives in
   `physics/vbd_coupling.py` so RL training, validation tools, and tests all use
   the same code.

---

## 1. Directory structure

```
franka_vbd_cube_pick/
├── __init__.py
├── franka_vbd_cube_pick_env_cfg.py   # base env config (no scene.object)
├── reward_utils.py                    # pure-torch reward math, no IL deps
├── physics/
│   ├── __init__.py
│   ├── vbd_coupling.py               # shared Warp kernels (two-way coupling)
│   ├── vbd_newton_cfg.py             # FrankaVbdCubePickNewtonCfg
│   └── vbd_newton_manager.py         # FrankaVbdCubePickNewtonManager
├── mdp/
│   ├── __init__.py
│   ├── observations.py               # reads from manager obs-cache
│   ├── rewards.py                    # reads cube pos from manager
│   └── events.py                     # calls manager.reset_particles()
└── config/franka/
    ├── __init__.py                    # gym.register()
    ├── joint_pos_env_cfg.py
    └── agents/rsl_rl_ppo_cfg.py
```

---

## 2. Newton config (`physics/vbd_newton_cfg.py`)

Key pattern:

```python
@configclass
class MyVbdNewtonCfg(NewtonCfg):
    class_type: str = "{DIR}.vbd_newton_manager:MyVbdNewtonManager"

    # Cube geometry
    cube_size: float = 0.05
    cube_resolution: int = 3  # N → (N+1)³ particles, 5N³ tets

    # Material (user-facing; manager converts to Lamé internally)
    young_modulus: float = 2e4
    poisson_ratio: float = 0.4
    density: float = 400.0

    # MUST stay ≈1e-5 (position-level stiffness multiplier, NOT damping)
    k_damp: float = 1e-5
    soft_contact_kd: float = 1e-5

    # Contact / VBD
    particle_radius: float = 0.015
    soft_contact_ke: float = 1e4
    soft_contact_mu: float = 1.5
    vbd_iterations: int = 20
    vbd_two_way_coupling: bool = True
    vbd_max_contacts_per_env: int = 200
    vbd_max_particle_velocity: float = 10.0
    solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg(
        iterations=100, impratio=50.0, cone="elliptic",
        use_mujoco_contacts=True, ...
    )
```

**Critical:** `k_damp` and `soft_contact_kd` are position-level stiffness
multipliers in Newton VBD, NOT velocity damping.  Keep at `1e-5`.

---

## 3. Newton manager (`physics/vbd_newton_manager.py`)

The manager overrides four `NewtonManager` lifecycle hooks:

```
start_simulation()         ← add_soft_grid() for env-0, tile, color, super()
initialize_solver()        ← create SolverVBD + CollisionPipeline BEFORE super()
_simulate()                ← dispatch to _simulate_two_phase()
clear()                    ← zero all class-level VBD state
```

### 3a. Fast env tiling (O(1) Python, O(N) numpy)

```python
# 1. call add_soft_grid() for env-0 only
cls._builder.default_particle_radius = cfg.particle_radius
cls._builder.add_soft_grid(pos=..., dim_x=N, ..., k_mu=k_mu, k_lambda=k_lambda, k_damp=cfg.k_damp)

# 2. snapshot env-0 builder lists
snap_before = {...}  # len() before add_soft_grid
tets_env0 = np.array(builder.tet_indices[snap_before["tet_indices"]:])
pq_env0   = np.array(builder.particle_q[snap_before["particle_q"]:])
...

# 3. tile via numpy broadcast — no Python loop
all_pq_new = pq_env0[None,:,:] + all_deltas[:,None,:]   # (N_envs, P, 3)
builder.particle_q.extend([wp.vec3(...) for r in all_pq_new.reshape(-1,3)])
all_tets = tets_env0[None,:,:] + env_offsets[:,None,None]  # (N_envs, T, 4)
builder.tet_indices.extend(map(tuple, all_tets.reshape(-1,4).tolist()))
```

### 3b. Tet-adjacency graph coloring

```python
# Build edges from tet adjacency
edge_set = set()
for row in tet_np:   # tet_np is local (0-based) for env-0 only
    for a, b in itertools.combinations(row, 2):
        edge_set.add((min(a,b), max(a,b)))
edge_np = np.array(sorted(edge_set), dtype=np.int32)
edge_wp = wp.array(edge_np, dtype=int, device="cpu")

# Color single-env, tile result with per-env offset
single_env_colors = color_graph(n_particles, edge_wp, ...)
tiled_colors = [
    np.concatenate([c + existing + env_idx*n_particles for env_idx in range(N_envs)])
    for c in single_env_colors
]
builder.set_coloring(tiled_colors)
builder.body_color_groups = color_rigid_bodies(
    builder.body_count, builder.joint_parent, builder.joint_child)
```

**Note:** `construct_particle_graph` from the dexsuite branch does NOT exist in
the current Newton.  Build tet edges manually as shown above.

### 3c. VBD solver initialization order

```python
@classmethod
def initialize_solver(cls):
    # MUST create SolverVBD BEFORE super().initialize_solver()
    # so CUDA graph capture sees _vbd_solver != None.
    cls._vbd_solver = SolverVBD(cls._model,
        iterations=cfg.vbd_iterations,
        integrate_with_external_rigid_solver=True,
        max_soft_contacts=practical_contact_max,
        particle_max_velocity=cfg.vbd_max_particle_velocity,
    )
    cls._soft_collision_pipeline = CollisionPipeline(
        cls._model,
        soft_contact_margin=cfg.particle_radius * 3.0,
        soft_contact_max=practical_contact_max,
        particles_per_world=particles_per_env,
        shapes_per_world=shapes_per_env,
    )
    cls._soft_contacts = cls._soft_collision_pipeline.contacts()
    cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)  # warm-up
    super().initialize_solver()  # ← captures CUDA graph
```

### 3d. Two-phase stepping

```python
@classmethod
def _simulate_two_phase(cls):
    for i in range(cls._num_substeps):
        cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
        if two_way:
            apply_soft_body_reactions(
                cls._soft_contacts, cls._state_0, cls._model,
                cls._soft_contact_max,
                particle_q_prev=cls._state_1.particle_q,
                friction_epsilon=cls._vbd_solver.friction_epsilon,
                dt=cls._solver_dt,
            )
        cls._solver.step(s0, s1, ...)         # rigid (reads body_f)
        cls._vbd_solver.step(s0, s1, ...)     # VBD (uses same contacts)
        s0, s1 = s1, s0                        # swap buffers
    cls._refresh_obs_cache()                   # updates pos/quat/vel arrays
```

---

## 4. MDP layer

### Observations

Cube pose comes from the manager obs-cache (pre-computed in CUDA graph):

```python
def cube_pos_b(env, robot_cfg):
    pose = FrankaVbdCubePickNewtonManager.get_object_pose()
    cube_pos_w = wp.to_torch(pose[0]).float()   # (N, 3)
    robot_pos_w, robot_quat_w = ...             # from robot.data
    cube_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, cube_pos_w)
    return cube_pos_b
```

### Events

```python
def reset_cube_pose_uniform(env, env_ids, pose_range, cube_size):
    n = env_ids.shape[0]
    x = torch.empty(n).uniform_(*pose_range["x"])
    y = torch.empty(n).uniform_(*pose_range["y"])
    z = torch.full((n,), cube_size / 2.0)
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0   # identity, wxyz
    root_pose = torch.stack([x, y, z, quat[:,0], quat[:,1], quat[:,2], quat[:,3]], dim=1)
    FrankaVbdCubePickNewtonManager.reset_particles(env_ids, root_pose)
    FrankaVbdCubePickNewtonManager.reset_particle_velocities(env_ids)
```

### No `scene.object`

Do NOT add a `scene.object = RigidObjectCfg(...)` entry.  The env_cfg
`GroundSceneCfg` has only `robot`, `plane`, and `light`.

---

## 5. Gym registration

Register Newton-only tasks:

```python
# config/franka/__init__.py
gym.register(
    id="Isaac-Pick-VBD-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRunnerCfg",
    },
    disable_env_checker=True,
)
```

Training command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-VBD-Cube-Franka-v0 presets=newton \
    --num_envs 64 --headless --max_iterations 5
```

---

## 6. Newton package requirements

The following patches must be applied to `/home/horde/projects/newton` before
a VBD task can run at scale (committed as `36a67b8` in the newton repo):

| File | Change |
|---|---|
| `newton/_src/geometry/kernels.py` | Add `create_soft_contacts_batched` kernel |
| `newton/_src/sim/collide.py` | Import + dispatch `create_soft_contacts_batched` when `particles_per_world` set |
| `newton/_src/sim/contacts.py` | Add `soft_contact_tids_dim` param to avoid int32 overflow at 4096 envs |
| `newton/_src/solvers/vbd/solver_vbd.py` | Add `max_soft_contacts` + `particle_max_velocity` params |

Without these patches:
- `soft_contact_max = P × S × num_envs` overflows int32 at 4096 envs
- VBD NaN explosions at first particle-rigid contact (no velocity clamping)

---

## 7. Key pitfalls

1. **`k_damp` and `soft_contact_kd` must stay ≈1e-5** — position-level
   stiffness multipliers, not velocity damping.  Any value > 1e-3 causes
   VBD divergence.
2. **SolverVBD created BEFORE `super().initialize_solver()`** — the CUDA graph
   is captured inside `super()`.  If `_vbd_solver is None` at capture time,
   `_simulate` falls through to the rigid-only base path.
3. **CollisionPipeline warm-up outside CUDA graph** — first `collide()` call
   does d2h shape-type copies (illegal inside graph capture).  Always call once
   before `super().initialize_solver()`.
4. **`construct_particle_graph` does not exist** in this Newton version.  Build
   tet edges manually for graph coloring.
5. **`add_soft_grid()` does not take `particle_radius`** — set
   `builder.default_particle_radius` before calling it.
6. **No `preset()` for Newton-only tasks** — `preset()` requires a `default`
   key.  Hard-code `decimation`, `sim.dt`, etc. directly.
