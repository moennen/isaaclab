# Rigid Cube Observation Reference (Dexsuite 3DG Lift Play)

Reference trace of object state and policy observations for the **rigid** spawn object (no Simplicits). Use this to compare with the Simplicits case and to verify that object pose, velocity, and observation slices match expectations.

**How to reproduce:** Run play with rigid cube and debug observations enabled:

```bash
DEXSUITE_3DG_DEBUG_OBSERVATIONS=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-Play-v0 \
  --num_envs 1 --visualizer newton presets=cube
```

(Do **not** add `env.sim.physics=simplicits env.scene=simplicits` — that would switch to Simplicits.)

---

## 1. Initialization position (config default)

Object default pose and velocity from the scene config (before any reset randomization). Source: `RigidObjectCfg.init_state` in the task scene (e.g. `dexsuite_env_cfg.py`).

| Field    | Value |
|----------|-------|
| pos [m]  | (-0.55, 0.1, 0.35) |
| quat (x, y, z, w) | (0.0, 0.0, 0.0, 1.0) |
| lin_vel [m/s] | (0.0, 0.0, 0.0) |

At episode reset, the object may be randomized (e.g. `reset_object` / `reset_root_state_uniform`); the values in the table below are from a single run after such resets.

---

## 1b. Rigid cube size and mass (preset=cube)

The **rigid** spawn object when using ``presets=cube`` comes from ``ObjectCfg.cube`` in ``dexsuite_env_cfg`` (:class:`~isaaclab.sim.MeshCuboidCfg` size and MassPropertiesCfg mass; mesh cuboid gives scale 1,1,1 for Simplicits):

| Field   | Value |
|---------|-------|
| size [m]| (0.05, 0.1, 0.1) — 5 cm × 10 cm × 10 cm |
| mass [kg] | 0.2 (200 g) |

**Simplicits:** Geometry uses the same prim: vertices in **geom local** frame are passed to Kaolin; **``world_from_local``** (USD ``ComputeLocalToWorldTransform`` on the geom) is passed as ``init_transform`` per env so cubature points match the rigid cube in world space. Previously, baking vertices into world space while passing identity ``init_transform`` was wrong: Kaolin’s ``B`` matrix depends on rest point coordinates, so world-space rest points made rigid motion act about the world origin. **Mass** is computed as density × volume: ``SimplicitsObjectCfg.density`` [kg/m³] × ``appx_vol`` [m³] (volume from mesh bbox in the factory). With default density 500 kg/m³ and the factory’s ``appx_vol = max(bbox_vol*0.5, 1e-6)``, total mass is typically ~0.03 kg (31 g), which is **lighter** than the rigid 0.2 kg. To align Simplicits mass with the rigid cube, set ``target_mass=0.2`` in ``SimplicitsObjectCfg``; the factory then uses density = 0.2 / appx_vol so total mass is 0.2 kg. Otherwise set ``density`` manually so that density × appx_vol ≈ 0.2.

---

## 2. Trace metadata (step 100)

| Field           | Value |
|-----------------|-------|
| object_type     | rigid |
| policy shape    | (1, 170) |
| proprio shape   | (1, 615) |
| perception shape| (1, 960) |

Policy layout: `object_quat_b` (dims 0:20), `target_object_pose_b` (dims 20:55), `actions` (dims 55:170).

---

## 3. Object state (env 0) every 100 steps

Object pose and linear velocity from `scene["object"].data` (root_pos_w, root_quat_w, root_com_vel_w). Units: position [m], quaternion (x,y,z,w), linear velocity [m/s].

| Step | pos (x, y, z) | quat (x, y, z, w) | lin_vel (x, y, z) |
|------|----------------|-------------------|-------------------|
| 100  | (-0.5507, -0.1410, 0.6553) | (0.1161, -0.1053, -0.3212, 0.9340) | (0.0354, 0.0373, 0.0262) |
| 200  | (-0.4948, -0.1819, 0.7472) | (0.0982, 0.0642, -0.3301, 0.9366) | (-0.0520, -0.0296, -0.0482) |
| 300  | (-0.4956, -0.1830, 0.7457) | (0.1494, 0.1644, -0.2726, 0.9362) | (0.0196, -0.0957, 0.0435) |
| 400  | (-0.5785, 0.1030, 0.5255)  | (0.4997, -0.8627, 0.0475, -0.0616) | (1.7796, 2.1174, 3.3136) |
| 500  | (-0.5127, 0.2303, 0.6020)  | (0.4614, -0.8639, 0.1277, -0.1565) | (0.0100, -0.0058, 0.0913) |

**Expected behavior (rigid):** Position, quaternion, and linear velocity **change over time** as the cube is moved by the policy and physics.

---

## 4. Policy observation `object_quat_b` (dims 0:20)

First 20 dimensions of the policy observation = object quaternion in body frame (5 fingertips × 4 quat components). Values below are from env 0 at each step.

### Step 100

```
[0.0961, -0.0869, -0.3136, 0.9427, 0.1181, -0.1194, -0.3052, 0.9504, 0.1067, -0.0834, -0.3368, 0.9571, 0.1379, -0.0856, -0.2945, 0.9415, 0.1002, -0.1086, -0.3349, 0.9267]
```

### Step 200

```
[0.0758, 0.0811, -0.3296, 0.9354, 0.0659, 0.0348, -0.3167, 0.9271, 0.1162, 0.0456, -0.3432, 0.9180, 0.1016, 0.0667, -0.3059, 0.9395, 0.0938, 0.0852, -0.3167, 0.9125]
```

### Step 300

```
[0.1619, 0.1654, -0.2705, 0.9511, 0.1288, 0.1832, -0.2767, 0.9474, 0.1375, 0.1824, -0.2880, 0.9381, 0.1621, 0.1877, -0.2611, 0.9610, 0.1419, 0.1779, -0.2783, 0.9440]
```

### Step 400

```
[0.5333, -0.8575, 0.1303, -0.0388, 0.4981, -0.8502, 0.1104, -0.0737, 0.5293, -0.8691, 0.0777, -0.0920, 0.5184, -0.8271, 0.0676, -0.0678, 0.5046, -0.8606, 0.0337, -0.0356]
```

### Step 500

```
[0.4773, -0.8625, 0.1024, -0.1767, 0.4613, -0.8630, 0.1109, -0.1540, 0.4408, -0.8907, 0.1399, -0.1719, 0.4409, -0.8530, 0.1379, -0.1387, 0.4854, -0.8680, 0.1349, -0.1774]
```

**Expected behavior (rigid):** These 20 values **change over time** and reflect the object pose as seen from each fingertip.

---

## 5. Comparison: rigid vs Simplicits

With `env.sim.physics=simplicits` and `env.scene=simplicits`, the spawn object has no Newton rigid
body; MDP still uses `scene["object"]` via `SimplicitsObjectAdapter`.

### 5.1 Object state (pos, quat, lin_vel) — intended behaviour

| Aspect | Rigid (reference) | Simplicits (current) |
|--------|-------------------|----------------------|
| **Spawn / reset pose** | `RigidObject` root state from MDP | On reset, `T_reset @ x_local` with mesh-local samples fixed at build (repeatable cloud). |
| **Each simulation step** | Rigid body integrator | Phase 2 runs `SimplicitsSolver.step`; adapter reads CoM, Kabsch rotation vs rest particles, and mean particle linear velocity. |
| **CUDA graph** | Optional | Off for the `simplicits` physics preset so two-phase Python runs every frame. |

Use `DEXSUITE_3DG_DEBUG_OBSERVATIONS=1` (and optionally `DEXSUITE_3DG_TRACE_TWO_PHASE=1`) to log
per-step CoM; values should change when the object moves.

### 5.2 Policy observation `object_quat_b` (dims 0:20)

- **Rigid:** The 20-dim slice changes each step (see section 4).
- **Simplicits:** Same observation layout; object pose in the finger frame reflects the particle
  cloud pose from the adapter (not a frozen rigid body).

### 5.3 If Simplicits object state still looks static

Historical failure mode: constant CoM, identity quat, zero velocity (particles not stepped every
frame, or stuck at build pose vs reset). Check:

1. **Phase 2 runs** — `use_cuda_graph=False` for Simplicits; confirm `_simulate_two_phase` runs
   after capture.
2. **Particles advance** — CoM before vs after `SimplicitsSolver.step` (trace env above).
3. **Reset sync** — After `reset_root_state_uniform`, particle positions should match the new
   root pose (adapter → `apply_simplicits_root_pose_reset`).
4. **Frame alignment** — USD geom transform at build vs MDP root pose should describe the same
   object frame; a mismatch shows up as offset or drift vs rigid.

### 5.4 Further checks (alignment with rigid reference)

1. **Confirm Simplicits state is stepped** — In `_simulate_two_phase()`, debug logs compare `_state_0.particle_q` (and `particle_qd`) for env 0 before and after `SimplicitsSolver.step` and `state_0.assign(state_1)`. Look for `[DEXSUITE_3DG_DEBUG] Simplicits phase2 (call …): env0 CoM before=… after=… pos_changed=…`. If these logs do not appear, note that with **CUDA graph** enabled (default), the Python step runs mainly during graph capture, so you may only see the first-call log; to see repeated phase2 logs, run with `env.sim.physics_cfg.use_cuda_graph=False` or equivalent. If CoM/velocity do not change before vs after, the bug is in the solver or in how state is copied.
2. **Confirm world-frame pose** — Ensure the pose we expose (root_pos_w, root_quat_w) is in the same world frame as the rigid object. If particles are built in env-local frame, add the env transform (e.g. from `env_proto_xforms` or scene) when computing CoM/orientation for the adapter.
3. **Match rigid init** — Optionally spawn Simplicits so the initial CoM in world frame is close to (-0.55, 0.1, 0.35) (or whatever the rigid init is after reset), so that the policy sees a similar initial condition.
4. **Velocity source** — Ensure `get_simplicits_object_velocity()` uses the **updated** `particle_qd` (after the Simplicits step) and that the adapter’s `update()` runs after `sim.step()` so it reads the latest state.

---

## 6. Step 5 status and Step 6 scope

**Reference test (rigid — working):**

```bash
DEXSUITE_3DG_DEBUG_OBSERVATIONS=1 CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-Play-v0 \
  --num_envs 1 --visualizer newton presets=cube
```

**Reference test (Simplicits):**

```bash
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-Play-v0 \
  --num_envs 1 --visualizer newton presets=cube env.sim.physics=simplicits env.scene=simplicits
```

**State:** Two-phase stepping (CUDA graph off), USD spawn transform, and reset-time particle
teleport give time-varying pose/velocity in observations aligned with MDP resets.
**Visualization:** Particles can be shown in the Newton viewer (`show_particles`).
**Step 6 success criterion (remaining):** Grasping and dynamics comparable to the rigid cube
(material, friction, particle count tuning); observations/rewards in the same ballpark so the
policy does not behave erratically.

**Step 6 requirements (refined):**

1. **Visualization: cubature/collision points** — The points used for Simplicits collision and force (cubature/particle positions) must be **visible** in the Newton visualizer (e.g. via a “show particles” or equivalent option) so we can confirm particle layout and motion.

2. **Skinning weights at model creation** — Simplicits **skinning weights** (mesh–particle binding) must be **built during model creation** (in the Simplicits builder/assembly when building the Simplicits object from the mesh), not deferred, so the simulated object is consistent from the first step.

3. **Simulation correctness** — The Simplicits simulation must **behave correctly**: particle state (`particle_q`, `particle_qd`) advances each step, the adapter exposes pose/velocity from that state, and the object responds to contacts and gravity in a way comparable to the rigid case (e.g. rests on table, can be moved by the policy).

4. **Debug and validation** — There must be a **clear way to debug and validate** the Simplicits simulation and observations: e.g. `DEXSUITE_3DG_DEBUG_OBSERVATIONS`, phase2 before/after logs, this reference document for rigid vs Simplicits comparison, and optionally a short validation script or test that compares observation/pose ranges (rigid vs Simplicits) over a short run.

**Step 6 implementation notes (current):**

- **Visualization:** The Simplicits object is not a Newton rigid body; the Newton OpenGL viewer shows it via **particles** (`show_particles=True`). The env enables that whenever Simplicits physics is active—including `env.sim.physics=simplicits` alone (e.g. with `presets=cube`), not only when `env.scene=simplicits` was set up front. Earlier, only the latter path called `_ensure_newton_visualizer_show_particles`, so particles stayed off. You can still override via `env.sim.visualizer_cfgs.0.show_particles=true` if needed.
- **Simulation:** CUDA graph is disabled when Simplicits is enabled (`cfg.use_cuda_graph = False` in `_start_simulation_simplicits`) so the two-phase step runs every frame and the adapter sees updated particle state.
- **Gravity:** Gravity is set **once** in `finalize_multi_world` (same order as Kaolin `SimplicitsModelBuilder.finalize()`): `acc_gravity[up_axis] = -gravity` so Z-up gives (0, 0, -9.81). Not set in `build_worlds_with_particles` to avoid applying twice. To validate, spawn higher (e.g. z=0.7); if the object does not fall, check particles have mass (debug log) and that the Newton ground plane and particle–ground contacts exist.
- **Spawn position:** Build reads the Object geom local-to-world matrix
  (`get_geom_world_transform_4x4`), applies it to mesh vertices
  (:func:`transform_points_mat4` with geom local-to-world), and passes **identity** as Simplicits ``init_transform`` so
  the particle cloud matches the prim on stage without polar/SVD. On reset,
  ``reset_root_state_uniform`` (and similar) call
  :meth:`~config.kuka_allegro.physic.newton.simplicits_object_adapter.SimplicitsObjectAdapter.write_root_pose_to_sim_index`,
  which teleports particles with ``T_reset @ T_build^{-1} @ p_build`` and sets velocities from the
  sampled root twist via :meth:`Dexsuite3dgNewtonManager.apply_simplicits_particles_velocity_reset`.
- **World frame:** Particles are built and simulated in world space consistent with baked vertices;
  gravity is set in ``finalize_multi_world`` as for the Kaolin Simplicits scene.
- **Skinning:** Simplicits mesh→particle binding is handled by the Kaolin factory (`create_rigid_simplicits_object_from_mesh`) during model creation; no extra skinning pass was added in the task.
- **Debug:** Set `DEXSUITE_3DG_DEBUG_OBSERVATIONS=1` to log object pose, policy obs slices, and (when enabled) phase2 CoM/velocity before-after in the manager.
- **State handling (refs):** Two-phase step: (1) rigid step as in base NewtonManager; (2) **two-pass Simplicits** so collision is computed from the state after gravity/forces. Pass 1: step with start-of-step contacts → predicted state in `state_1`; collide(`state_1`) → contacts from predicted state; Pass 2: step from `state_1` with those contacts → final state in `state_0`. No swap after Phase 2 (final is already in `state_0`). This avoids tunneling: using only start-of-step contacts would let the object fall through the table in one step.
- **Phase 1 and particles:** The Newton (MuJoCo Warp) solver’s `_update_newton_state` only writes joint/body state; it does **not** write `particle_q`/`particle_qd`. So during Phase 1 (rigid substeps), particle state is unchanged (buffers are swapped, not overwritten). Simplicits updates are applied only in Phase 2 and written back into `state_1`’s particle slice; the swap then makes that the new `state_0`.
- **Write-back trace:** With phase2 debug logging, the manager prints `state_0(in)` vs `state_1(out)` particle CoM and `|diff|` immediately after `SimplicitsSolver.step` (before swap). If `|diff|` is zero or very small while gravity should be pulling the object down, the bug is likely inside Kaolin (e.g. gravity not in the Simplicits energy or `run_sim_step` not advancing `sim_z`).
- **Kaolin Simplicits trace:** Set ``KAOLIN_SIMPLICITS_TRACE=1`` (and reinstall Kaolin if you added traces in the Kaolin tree). Then in ``kaolin.physics.simplicits.easy_api``: (1) ``set_scene_gravity`` logs ``acc_gravity``, ``coeff``, and ``total_gravity_mass`` (sum of ``sim_rhos * sim_vols``, i.e. the mass used for F=mg in the gravity force); (2) ``run_sim_step`` logs at steps 0–2 and every 120: ``pt_wise_keys``, ``has_gravity``, ``sim_z_prev_CoM`` before Newton and ``sim_z_CoM``, ``dCoM`` after. Compare ``total_gravity_mass`` to the manager’s ``total_mass`` log; if the former is zero or much smaller, gravity is weak and the object can float (mass issue).
- **Kaolin state initialization:** ``SimplicitsSolver.step(state_in, state_out, ...)`` copies **state_in.sim_z** and **state_in.sim_z_dot** into the scene before ``run_sim_step()``; the scene uses that as the previous state. The manager keeps the frame-start in **state_1** and explicitly syncs **sim_z** / **sim_z_dot** from **state_0** to **state_1** after ``assign()`` so both predict and correct steps receive the intended input. The trace ``sim_z_prev_CoM`` is the CoM of **reduced** coordinates; for centered reduced coords this can be ``[0, 0, 0]`` even when the object is not at the world origin.

**Interpreting traces when the object floats**

- Kaolin logs **reduced (sim_z)** CoM and ``dCoM``; DEXSUITE logs **world** particle CoM. They are in different spaces.
- If ``acc_gravity=[0,0,-9.81]`` and ``has_gravity=True`` but the object still floats: (1) In sim_z space, step 0 may show ``dCoM`` z negative (gravity pulling in reduced space). (2) In world space, particle CoM z may still increase (e.g. 0.701 → 0.703). So the **sim_z → world** map (LBS matrix / ``sim_pts``) or material/collision forces can produce world motion that does not go down. (3) At later steps (e.g. 120), ``dCoM=[0,0,0]`` means the Newton solver converged to no change (equilibrium).
- **Frame (Z-up):** The Simplicits scene gravity is set to Z-up in assembly (``acc_gravity[2]=-9.81``), so ``particle_q`` is in world Z-up; no y/z swap is applied (a previous swap was removed because it turned the solver's Z-up output into Y-up).
- **Mass:** Gravity force is F = (rho * vol) * g per point. If ``sim_rhos`` or ``sim_vols`` were zero or very small in the scene, gravity would be weak and the object would float. With ``KAOLIN_SIMPLICITS_TRACE=1``, ``set_scene_gravity`` logs ``total_gravity_mass`` (sum of rho*vol). It should match the manager’s ``total_mass`` (e.g. ~0.031 for a 5 cm cube, density 500). If ``total_gravity_mass`` is near zero, that’s a mass bug (e.g. wrong scene buffers or units).
- Next checks: ensure the Simplicits scene has a floor or particle–ground contact so the object can land; inspect how rest positions and the LBS basis are built (object-local vs world) so that a decrease in sim_z z implies a decrease in world z; consider increasing gravity coefficient or reducing material stiffness so gravity dominates.

---

## 7. Debugging Simplicits grasping

**Rigid vs Simplicits:** Grasping works with the rigid cube; with Simplicits the robot may reach and close fingers but the object slips or is not held. This is expected until particle count, friction (`OBJECT_STATIC_FRICTION` / `model.soft_contact_mu`), and contact behaviour are tuned.

**Gravity:** If the Simplicits object does not appear to fall (e.g. stays at spawn height or only moves when pushed by the robot): (1) Gravity is set once in `finalize_multi_world` via `scene.set_scene_gravity(acc_gravity)` with `acc_gravity[up_axis] = -9.81`. (2) Confirm particles have mass: at startup a debug line logs `density_mean` and `total_mass` for env 0. (3) With ``KAOLIN_SIMPLICITS_TRACE=1``, confirm ``has_gravity=True`` and ``total_gravity_mass`` > 0. If the cube still does not fall under gravity alone, the gravity **pt_wise** force may not be applied in Kaolin’s ``run_sim_step`` (e.g. multi-world or custom finalize order vs standard ``SimplicitsModelBuilder.finalize``). Debug in the Kaolin tree: ensure the ``gravity`` entry in ``scene.force_dict["pt_wise"]`` is iterated and its contribution added during the Simplicits Newton step. (4) Confirm the Newton model has a ground plane and particle–ground contacts so the object can land.

If the robot reaches the object and closes fingers but the object slips or is not held:

**Particle count**

- Simplicits object is represented by **cubature/particle points**; contact with the robot is particle–shape. Fewer particles mean fewer contact points and easier slip.
- Simplicits params (e.g. `num_samples`, `density`) are read from the **Object prim** custom data, written by the **SimplicitsCubeCfg** spawner when using `env.scene=simplicits`. Set them on the scene’s object spawn config (e.g. `SimplicitsCubeCfg(num_samples=300)`). Increase (e.g. 300–500) for denser contact at the cost of compute.

**Friction (particle–rigid)**

- Effective friction for particle–rigid contact is `mu = 0.5 * (particle_mu + shape_mu)` in the contact kernel. Particle side uses **model.soft_contact_mu**; shape side uses **model.shape_material_mu** (from the rigid body shapes, e.g. fingers).
- The task sets **model.soft_contact_mu** from the shared `OBJECT_STATIC_FRICTION` (in `object_defaults.py`, same as the rigid object’s static friction, 0.5) so rigid and Simplicits object friction stay aligned.
- Robot finger friction comes from the rigid proto (USD/MJCF). If the rigid cube task uses a high-friction material for the object, the same scene may define finger friction; otherwise shape_mu is from the builder defaults.

**How to debug**

1. **Confirm particle count:** With `DEXSUITE_3DG_DEBUG_OBSERVATIONS=1`, the first log line from `build_worlds_with_particles` includes `n=<count>` for env 0.
2. **Confirm friction:** After startup, the model’s `soft_contact_mu` is set in `Dexsuite3dgNewtonManager._start_simulation_simplicits` from the shared `OBJECT_STATIC_FRICTION`; no runtime log by default. Add a one-off log there if needed.
3. **Contacts:** Newton/Kaolin soft contact count and indices are solver-internal. For rigid–rigid contacts, `NewtonManager` can report contacts; particle–shape contacts go through the Simplicits force (newton_soft_collisions). Enabling contact visualization in the Newton viewer (`show_contacts=True`) shows rigid contacts only, not particle–shape.
4. **Compare to rigid:** Run the same policy with the rigid cube (no simplicits). If grasping works there but not with Simplicits, the difference is particle count and/or particle–rigid friction (and possibly collision radius); tune `num_samples` first; friction is shared via `OBJECT_STATIC_FRICTION` in `object_defaults.py`.

Use this document as the baseline when debugging Simplicits observation and state alignment.
