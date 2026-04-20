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
    young_modulus: float = 2e4  # softer cube; ke/k_mu=1.40 but safe without CUDA graph
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

**CRITICAL:** `add_soft_grid` creates BOTH tetrahedra AND surface triangles.
VBD uses surface-triangle elasticity (`tri_materials`), NOT tetrahedra.
You MUST tile `tri_indices`, `tri_poses`, `tri_materials`, `tri_activations`, `tri_areas`
in addition to the tet fields.  Without this, worlds 1..N-1 have no shape constraint
and collapse under gravity.

A 3×3×3 grid produces: 64 particles, 135 tets, **108 surface tris**.

```python
# 1. call add_soft_grid() for env-0 only
cls._builder.default_particle_radius = cfg.particle_radius
cls._builder.add_soft_grid(pos=..., dim_x=N, ..., k_mu=k_mu, k_lambda=k_lambda, k_damp=cfg.k_damp,
                           tri_ke=k_mu, tri_ka=k_lambda, tri_kd=k_damp)

# 2. snapshot env-0 builder lists (BEFORE add_soft_grid)
snap_before = {
    "particle_q":      len(builder.particle_q),
    "tet_indices":     len(builder.tet_indices),
    "tet_poses":       len(builder.tet_poses),
    "tet_materials":   len(builder.tet_materials),
    "tet_activations": len(builder.tet_activations),
    "tri_indices":     len(builder.tri_indices),    # ← MUST capture
    "tri_poses":       len(builder.tri_poses),
    "tri_materials":   len(builder.tri_materials),
    "tri_activations": len(builder.tri_activations),
    "tri_areas":       len(builder.tri_areas),
}

# 3. after add_soft_grid: capture env-0 data
tets_env0 = np.array(builder.tet_indices[snap_before["tet_indices"]:]).reshape(-1, 4)
tris_env0 = np.array(builder.tri_indices[snap_before["tri_indices"]:]).reshape(-1, 3)
tri_poses_env0 = list(builder.tri_poses[snap_before["tri_poses"]:])
tri_mats_env0  = list(builder.tri_materials[snap_before["tri_materials"]:])
tri_acts_env0  = list(builder.tri_activations[snap_before["tri_activations"]:])
tri_areas_env0 = list(builder.tri_areas[snap_before["tri_areas"]:])
pq_env0   = np.array(builder.particle_q[snap_before["particle_q"]:])

# 4. tile envs 1..N-1
for w in range(1, N_envs):
    offset = snap_before["particle_q"] + w * n_particles
    # particles
    builder.particle_q.extend(...)
    # tets
    tets_w = tets_env0 + offset
    builder.tet_indices.extend(map(tuple, tets_w.tolist()))
    builder.tet_poses.extend(tet_poses_env0)
    builder.tet_materials.extend(tet_mats_env0)
    builder.tet_activations.extend(tet_acts_env0)
    # surface tris ← REQUIRED for VBD elasticity
    tris_w = tris_env0 + offset
    builder.tri_indices.extend(map(tuple, tris_w.tolist()))
    builder.tri_poses.extend(tri_poses_env0)
    builder.tri_materials.extend(tri_mats_env0)
    builder.tri_activations.extend(tri_acts_env0)
    builder.tri_areas.extend(tri_areas_env0)
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

**Current branch**: `nicolasm/dexsuite-vbd` (tracks `origin/nicolasm/isaac-dexsuite-3dg-kuka-experimental-tasks`), Newton **1.0.0rc3**.

Install in env_isaaclab:
```bash
cd /home/horde/projects/newton
git checkout nicolasm/dexsuite-vbd
micromamba run -n env_isaaclab pip install -e .
```

This branch has all needed features built-in — no manual patches required:
- `CollisionPipeline(model, ...)` — model stored internally (new API)
- `pipeline.contacts()` — pre-allocates a `Contacts` buffer
- `pipeline.collide(state, contacts)` — state-first, CUDA-graph-compatible
- `SolverVBD(model, max_soft_contacts=..., particle_max_velocity=...)` — clamping built-in

**API changes vs old `nicolasm/isaaclab-task-skills` branch:**
| Old API | New API (1.0.0rc3) |
|---|---|
| `mb.body_key` | `mb.body_label` |
| `nik.IKPositionObjective(...)` | `nik.IKObjectivePosition(...)` |
| `nik.IKJointLimitObjective(...)` | `nik.IKObjectiveJointLimit(...)` |
| `nik.IKRotationObjective(...)` | `nik.IKObjectiveRotation(...)` |
| `nik.IKJacobianMode.ANALYTIC` | `nik.IKJacobianType.ANALYTIC` |
| `CollisionPipeline(shape_count, particle_count, pairs, ...)` | `CollisionPipeline(model, ...)` |
| `collision.collide(model, state)` | `collision.collide(state, contacts)` |
| `collision.contacts` (lazy-allocated) | `collision.contacts()` (pre-allocate once) |

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
7. **`HeightfieldData` not defined in this Newton version** — our commit `36a67b8`
   accidentally included `shape_heightfield_data` / `heightfield_elevation_data`
   parameters (copied from dexsuite) in `create_soft_contacts_batched`.  Fixed in
   commit `ed62423`: those parameters removed.  Update before running standalone scripts.
8. **Arm links collide with VBD particles during approach — disable `COLLIDE_PARTICLES`**.
   The rigid collision_group mechanism (group=2 for arm URDF shapes) prevents
   rigid-rigid contact with the cube, but VBD particle-rigid contacts use a
   separate `COLLIDE_PARTICLES` flag on each shape.  Without clearing this flag
   on arm link shapes, the arm swings through the cube during HOME→PRE_GRASP
   (t < 2s) and launches it 2–68m laterally.  Symptoms: cube dist_xy > 1m by t=1s;
   `reachable_mask=0` for 91% of reachable frames; R²≈0.02 on reachable branch.

   **Fix** (apply after `model.finalize()`, before creating SolverVBD):
   ```python
   _FLAG_COLLIDE_PARTICLES = 1 << 2  # ShapeFlags.COLLIDE_PARTICLES
   shape_flags_np = model.shape_flags.numpy().copy()
   shape_cg_np    = model.shape_collision_group.numpy()
   shape_flags_np[shape_cg_np == 2] &= ~_FLAG_COLLIDE_PARTICLES
   model.shape_flags.assign(shape_flags_np)
   ```
   This mirrors the rigid task's approach (arm URDF shapes → group=2, finger BOX
   shapes → group=1) but extended to the VBD particle contact path.  The same
   logic applies to the RL training env manager — any shape that should not contact
   VBD particles must have `COLLIDE_PARTICLES` cleared after finalize.

   **Same-root fix in rigid task**: in the rigid generator, arm URDF shapes are
   assigned `shape_collision_group[i] = 2` so they don't appear in the BOX-BOX
   contact list with the cube (group=1).  VBD needs the additional flag step above
   because particle contacts bypass the collision_group filter.

9. **CUDA graph substep order must match proxy_newton_manager.py (two-buffer mode)**.
   The correct order per substep is:
   ```
   1. collision.collide(state_0, soft_contacts)   # collision FIRST, state-first API
   2. apply_soft_body_reactions(soft_contacts, state_0, ...)  # fill body_f
   3. rigid_solver.step(state_0, state_1, ...)    # reads body_f
   4. vbd_solver.step(state_0, state_1, ...)      # uses same contacts
   5. state_0, state_1 = state_1, state_0         # swap buffers
   6. state_0.clear_forces()                      # clear AFTER swap (on new state_0)
   ```
   With EVEN `_N_SUBSTEPS=10`, state_0 holds the result after the loop — no Python swap
   needed after `capture_launch`. Use `use_cuda_graph=True` arg to `simulate_two_phase`
   to enable `need_copy` logic for odd substep counts.

   **Old bugs (fixed)**: Our pre-2026-04-15 code had (1) `clear_forces()` at TOP before
   collision — should be BOTTOM after swap; (2) `collision.collide(model, state)` — old
   API, state-second; (3) no warm-up before capture; (4) `graph_sim=[False]` disabled.

   **Impact on determinism**: In eager mode, non-deterministic atomic accumulation in VBD
   contact kernels means replay MUST use the same `--num-worlds N` as generation.
   Different N → different thread-block sizes → different rounding → different physics.

   **Young's modulus with CUDA graph**: Young's=2e4 Pa works correctly with CUDA graphs
   in Newton 1.0.0rc3. Default is 2e4 Pa.
   **Default `graph_sim = [None]`** in generate_sequences.py — CUDA graph enabled.
   Use `--no-cuda-graph` for eager mode debugging.

10. **Unreachable cube sampling must exclude the inner ring (dist < r_min)**.
    Cubes at dist < r_min (0.22m) sit inside the robot arm's sweep path during
    HOME→PRE_GRASP. Even with COLLIDE_PARTICLES cleared on arm shapes, some arm
    geometry still contacts VBD particles at t≈2.3–2.7s, falsely lifting the cube
    before the gripper closes. Restrict unreachable positions to dist > r_max only
    (implemented in `scripts/_common/sampling.py`):
    ```python
    # Unreachable: dist > r_max (outer ring only — inner ring causes arm-sweep collision)
    is_valid = (dist > r_max)
    ```

---

## 8. Validation workflow (follow skill 05 pattern)

Before RL training, run the full validation sequence in `scripts/`:

```bash
cd source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/franka_vbd_cube_pick

# Step 1 — Generate scripted sequences (physics validation)
# CUDA graph ENABLED (Newton 1.0.0rc3, dexsuite branch); --num-worlds 16 for speed
# Add --no-cuda-graph for debugging if needed
micromamba run -n env_isaaclab python scripts/generate_sequences.py \
    --num_sequences 100 --num-worlds 16 --seed 42 \
    --young-modulus 2e4 \
    --output data/validation/vbd_sequences_v7.json

# Step 2 — Compute rewards from sequences (no re-simulation needed)
micromamba run -n env_isaaclab python scripts/compute_rewards_from_seqs.py \
    --input  data/validation/vbd_sequences_v7.json \
    --output data/validation/vbd_seqrewards_v7.json

# Step 3 — Replay sequences to validate physics consistency
# MUST use same --num-worlds as generation (determinism requirement)
micromamba run -n env_isaaclab python scripts/replay_sequences.py \
    --input  data/validation/vbd_sequences_v7.json \
    --output data/validation/vbd_replay_v7.json \
    --num-worlds 16

# Step 4 — Analyze results (confusion matrix, reward stats)
micromamba run -n env_isaaclab python scripts/analyze_results.py \
    --input  data/validation/vbd_seqrewards_v8.json

# Step 5 — Extract observations (use RAW sequences, not seqrewards)
# seqrewards strips joint_pos/vel fields; raw sequences have them
micromamba run -n env_isaaclab python scripts/compute_observations.py \
    --input  data/validation/vbd_sequences_v8.json \
    --output data/validation/vbd_observations_v8.json

# Step 6 — Analyze observations
micromamba run -n env_isaaclab python scripts/analyze_observations.py \
    --obs    data/validation/vbd_observations_v8.json \
    --seqs   data/validation/vbd_seqrewards_v8.json \
    --output data/validation/vbd_obs_report_v8
```

**Physics validation pass criteria** (generator output):
- No `NaN!` lines in output — all cube CoMs remain finite
- No peaks at peak_z > 1m — CUDA graph works correctly with Newton 1.0.0rc3
- `LIFTED` for some reachable_success sequences (≥50%), `peak_z≈0.033m` for unreachable
- `peak_z` never > 0.05m for unreachable_failure sequences

**Reward validation pass criteria** (seqrewards output):
- Overall accuracy ≥ 75% (78% achieved in v8)
- reachable_success hit rate ≥ 50% (52.4% in v8)
- **0% false positives for reachable_failure** (critical — 24.2% FP in v7 was from wrong substep order)
- unreachable sequences: 0% false positives
- See known limitations below

**Key VBD-specific differences from rigid validation** (skill 05):
- `cube_pos_w` in sequences is particle CoM (not rigid body joint_q)
- CUDA graph **ENABLED** in generate_sequences.py (graph_sim=[None]) with Newton 1.0.0rc3
- Grasping success rate will be LOWER than rigid — deformable cube is harder to grip
  with scripted IK; expect 50–65% RS lift rate (v8: 52.4%)
- **CRITICAL**: Disable `COLLIDE_PARTICLES` on arm link shapes AFTER `model.finalize()`
  (see pitfall #8 in section 7). Without this, approach phase launches cube 2–68m.
- **CRITICAL**: Unreachable cubes must use outer ring only (dist > r_max). Inner ring
  puts cube inside arm sweep path, causing false lifts at t≈2.3s (see pitfall #10).

---

## 9. VBD Validation Results and Known Limitations

### Approach arm-collision fix (2026-04-14)

During HOME→PRE_GRASP (t < 2s), robot arm links sweep through the cube and launch it
laterally.  In the rigid task this is prevented by `shape_collision_group[arm] = 2` (arm
shapes excluded from cube's BOX-BOX group).  For VBD, particle-rigid contacts use the
separate `COLLIDE_PARTICLES` shape flag — collision_group alone is insufficient.

**Measured impact without fix** (`vbd_sequences_v3.json`, 100 seqs, seed=42):
- 91% of reachable frames have `reachable_mask=0` (cube knocked outside r_max=0.65m)
- `R²=0.022` for reachable branch (essentially random)
- Reachable success vs failure indistinguishable (mean=182 vs 181)
- Overall accuracy: 70% (seqrewards) / 65% (replay)

**Fix applied** in `generate_sequences.py` after `model.finalize()`:
```python
_FLAG_COLLIDE_PARTICLES = 1 << 2
shape_flags_np = model.shape_flags.numpy().copy()
shape_flags_np[model.shape_collision_group.numpy() == 2] &= ~_FLAG_COLLIDE_PARTICLES
model.shape_flags.assign(shape_flags_np)
```

Regenerate with `vbd_sequences_v4.json` (seed=42) to get clean validation data.

### V7 validated results (2026-04-15, current best — all fixes applied)

**Fixes applied**: arm-collision (COLLIDE_PARTICLES cleared), CUDA graph disabled,
inner-ring sampling fix, Young's=2e4.

| Metric | V3 (broken) | V4 (arm fix) | V7 (all fixes) |
|---|---|---|---|
| Accuracy | 70% | 76% | **75%** |
| RS hit rate | — | 50% | **64.3%** |
| RF false positive rate | — | 19% | 24.2% |
| reachable_success R² | 0.022 | 0.094 | **0.142** |
| unreachable_success R² | 0.600 | 0.975 | **~0.97** |
| US/UF false positives | many | some | **0%** |

V7 dataset (`vbd_sequences_v7.json`, seed=42, 100 seqs):
- RS (42 seqs): 27 LIFTED, 15 not lifted
- RF (33 seqs): 8 false lifts, 25 correct
- US (12 seqs): 10 signal reached, 2 not
- UF (13 seqs): 0 false lifts ✓

Remaining limitations:
- RF false positive rate 24.2% — robot legitimately grasps reachable cubes regardless
  of "failure" intent; this is inherent to how RF labels work
- RS lift rate not 100% — scripted IK + VBD contact variability; acceptable for RL training
- Replay must use same `--num-worlds 16` for determinism (eager mode non-determinism)

Unreachable branch is clean (0% false positives). Dataset is fit for RL training.

---

### VBD Contact Non-Determinism (Eager Mode)

In eager mode (CUDA graph disabled), VBD contact kernels use batched atomic accumulation
on GPU. Floating-point summation order depends on GPU thread-block scheduling, which varies
with `num_worlds` (different world counts → different thread-block sizes → different rounding
→ different contact forces → different physics outcome).

**Mitigation**: Use same `--num-worlds` for both generation and replay. With 16 worlds,
replay match rate is ~75% (v7 result). Do NOT mix 4-world generation with 16-world replay.

**Why not CUDA graphs**: CUDA graph captures stale VBD internal buffer state
(particle_q_prev, inertia). Replay of captured graph produces explosions at T_GRASP
(step 225) when finger target changes. Eager mode avoids this at the cost of non-determinism.

**Robot joint variance**: low (mean 0.002-0.014 rad). Non-determinism is purely in VBD contact,
not in the arm trajectory.

---

## 10. Material Space Search (physics parameter sweep)

When grasping reliability is poor (reachable_success sequences have low peak_z),
run `scripts/sweep_physics.py` to identify the right contact parameters.

### Root cause: finger effort vs contact normal force

The grip force balance for VBD:

```
F_normal_per_particle = model.soft_contact_ke × particle_radius
F_friction_per_particle = mu_eff × F_normal  where mu_eff = sqrt(soft_mu × contact_mu)

Finger actuator: force = finger_ke × (q_target - q), capped at finger_effort
```

**Critical constraint**: `finger_effort > F_normal_per_particle × n_particles_in_contact`

Otherwise the contact load pushes the finger open faster than the PD can close it.

With defaults (`soft_ke=1e4`, `particle_radius=0.015m`):
- `F_normal = 1e4 × 0.015 = 150N per particle`
- `finger_effort = 100N` → **finger opens immediately** (150N > 100N)

### Key parameters and their roles

| Parameter | Default | Effect |
|---|---|---|
| `soft_contact_ke` | 1e4 | Contact stiffness in `apply_soft_body_reactions` (reaction on finger). **Lower → finger stays closed.** Does NOT affect VBD-internal particle correction (which uses `rigid_contact_k_start`). |
| `soft_contact_mu` | 1.5 | Particle friction. Combined: `mu_eff = sqrt(soft_mu × contact_mu)`. Higher → better grip. |
| `contact_mu` (finger box `_CONTACT_MU`) | 0.75 | Rigid shape friction. Part of `mu_eff`. |
| `density` | 400 | Cube mass = density × (0.05)³ kg. Lower → easier to lift. |
| `finger_ke` | 100 | Finger PD stiffness. Higher → finger can resist more contact load. |
| `finger_effort` | 100 | Finger actuator effort limit [N]. Must exceed total contact normal load. |

**Note**: `rigid_contact_k_start` in SolverVBD = 0.5×(soft_ke + finger_box_ke) ≈ 25000 regardless of `soft_ke` (dominated by large `_CONTACT_KE=5e4`). VBD internal particle correction is essentially constant across probes — only the REACTION to the rigid body changes with `soft_ke`.

### Running the sweep

```bash
# Full sweep — 6 probes × 4 sequences each (~30 min)
micromamba run -n env_isaaclab python scripts/sweep_physics.py

# Targeted sweep — run specific probes
micromamba run -n env_isaaclab python scripts/sweep_physics.py --probes B,D,E

# Dry run — show commands without executing
micromamba run -n env_isaaclab python scripts/sweep_physics.py --dry-run
```

### Sweep results (2026-04-14, 4 sequences per probe, seed=123)

| Probe | soft_ke | soft_mu | contact_mu | density | finger_ke | finger_effort | hit/4 | mean_z |
|---|---|---|---|---|---|---|---|---|
| A:baseline | 1e4 | 1.5 | 0.75 | 400 | 100 | 100 | 2/4 | 0.153 |
| B:lo_ke | 1e3 | 2.0 | 1.5 | 200 | 100 | 100 | 1/4 | 0.119 |
| C:hi_feffort | 1e4 | 2.0 | 1.5 | 200 | 500 | 500 | 2/4 | 0.170 |
| D:balanced | 3e3 | 3.0 | 2.0 | 150 | 300 | 200 | 1/4 | 0.164 |
| E:lo_ke_hi_fr | 5e2 | 5.0 | 4.0 | 100 | 200 | 200 | 2/4 | 0.186 |
| F:hi_ke+fstr | 5e4 | 3.0 | 3.0 | 100 | 2000 | 1000 | 2/4 | 0.172 |

### Rigid-limit validation (2026-04-14)

Tested whether VBD-near-rigid matches the rigid franka_cube_pick success rate (~100%).

| Config | res | radius | Young's | soft_ke | soft_mu | density | effort | hit/N |
|---|---|---|---|---|---|---|---|---|
| Young's-rigid | 3 | 15mm | 2e7 | 5e2 | 0.75 | 400 | 100 | 1/4 |
| high-friction res=3 | 3 | 15mm | 2e4 | 1e4 | 3.0 | 400 | 100 | 6/16 |
| **res=5 (current)** | **5** | **9mm** | **2e4** | **1e4** | **3.0** | **400** | **100** | **16/16** |

**Root cause of previous failures**: `cube_resolution=3` gives only 16 surface particles per
face (4×4 grid, 16.7mm spacing). The finger box doesn't cover the full 50mm face — it covers
roughly the central 15-20mm. With some approach angles only 2-4 particles fell inside the
finger footprint → weak grip. Increasing to `cube_resolution=5` gives 36 particles/face
(6×6, 10mm spacing) → always enough particles in contact regardless of approach angle.

**Particle radius constraint** (avoid interior contacts):
```
inner_layer_spacing = cube_size / cube_resolution = 50mm / 5 = 10mm
equilibrium_penetration = effort / (ke × N_surface) = 100 / (1e4 × 36) ≈ 0.3mm
max_safe_radius = 10mm - 0.3mm ≈ 9.7mm  → use 9mm (3% margin)
```

With radius=9mm: contact spheres overlap by 2×9mm-10mm=8mm → full surface coverage ✓.
Interior particles stay contact-free (10mm - 0.3mm = 9.7mm > 9mm radius) ✓.

**Current parameters** (`generate_sequences.py` defaults, 2026-04-14):
`cube_resolution=5`, `particle_radius=0.009m`, `Young's=2e4`, `soft_ke=1e4`,
`soft_mu=3.0` (mu_eff=1.50), `contact_mu=0.75`, `density=400`, `effort=100`.

```bash
# Generate full dataset
python scripts/generate_sequences.py --num_sequences 100 --num-worlds 4 --seed 42 \
    --output data/validation/vbd_sequences_v3.json
```

---

## 11. Particle Grid Design: Resolution and Radius

This section documents the two parameters that control VBD contact coverage and should
be revisited whenever the cube geometry, finger size, or contact stiffness changes.

### Why these parameters matter

VBD contact works at **discrete particle positions** — not on a continuous surface.
When the finger closes on the cube, it only contacts particles whose centers are within
`particle_radius` of the finger surface. If too few particles fall inside the finger's
physical footprint, the grip is unreliable regardless of friction or stiffness settings.

### Choosing `cube_resolution`

`cube_resolution=R` creates `(R+1)³` particles on a uniform grid with spacing
`cube_size / R`. Surface particles per face = `(R+1)²`.

| resolution | particles | surface/face | face spacing |
|---|---|---|---|
| 3 | 64 | 16 (4×4) | 16.7mm |
| 4 | 125 | 25 (5×5) | 12.5mm |
| **5** | **216** | **36 (6×6)** | **10.0mm** |

The Panda finger box covers roughly 15–20mm of the 50mm cube face. At `res=3` some
approach angles put only 2–4 particles in contact → unreliable grip (37% lift rate).
At `res=5` there are always enough particles in the footprint → reliable grip (100%).

**Rule of thumb**: choose resolution so that `cube_size / resolution < finger_width / 2`.
For the Panda (finger_width ≈ 15mm): need spacing < 7.5mm → `resolution ≥ 7`. In
practice `resolution=5` works because the contact sphere overlap (see below) provides
extra margin.

### Choosing `particle_radius`

`particle_radius` controls the contact detection distance. Two constraints apply:

**Lower bound — surface coverage** (contact spheres must overlap across the face):
```
2 × particle_radius > face_spacing
particle_radius > cube_size / (2 × resolution)
```
For res=5: `particle_radius > 50mm / 10 = 5mm`

**Upper bound — no interior contact** (finger must not reach the second particle layer):
```
particle_radius < inner_layer_spacing - equilibrium_penetration
inner_layer_spacing = cube_size / resolution
equilibrium_penetration = finger_effort / (soft_ke × N_surface_particles)

→ particle_radius < cube_size/resolution - finger_effort/(soft_ke × (resolution+1)²)
```
For res=5, soft_ke=1e4, effort=100N, N=36:
```
particle_radius < 10mm - 100/(1e4 × 36) = 10mm - 0.28mm ≈ 9.7mm
```
Use 9mm (3% safety margin).

**Summary for current setup**:
```
resolution=5 → spacing=10mm
5mm < particle_radius < 9.7mm  → use 9mm
```

### Recalculating when parameters change

If you change `soft_ke` or `finger_effort`, recompute the upper bound:

```python
inner_spacing = cube_size / cube_resolution          # e.g. 0.010 m
equil_depth   = finger_effort / (soft_ke * n_surf)  # e.g. 100/(1e4*36) = 2.8e-4 m
max_radius    = inner_spacing - equil_depth          # e.g. 0.0097 m
particle_radius = 0.97 * max_radius                  # 3% margin
```

Constants to update in `generate_sequences.py`: `_CUBE_RESOLUTION`, `_PARTICLE_RADIUS`,
and the derived `_CUBE_REST_Z = _CUBE_HALF + _PARTICLE_RADIUS` (auto-updated).
