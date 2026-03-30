# Simplicits–Newton/MuJoCo Coupling Analysis

**Context:** The task runs a manipulation policy trained on a rigid cube but deployed on a
deformable Simplicits object (rigid Simplicits, 1 handle). The robot reaches the cube but
cannot grasp it. This document analyses why and what to do about it.

---

## Current architecture

The implementation uses a **two-phase sequential step** each substep:

```
[substep i]
1. _update_proxy_body_state()  → teleport proxy box to Simplicits CoM (writes joint_q)
2. step_fn (MuJoCo LCP)        → robot + proxy box rigid dynamics
3. clear_forces()
4. model.collide(state_0)      → detect particle ↔ rigid shape contacts
5. SimplicitsSolver.step()     → update particles
6. ← accumulate_reaction_on_bodies()  ONLY if proxy is NOT active
```

---

## Root causes of the grasping failure

Three distinct coupling failures, layered on top of each other.

### Problem 1 — Bidirectional force feedback is broken in proxy mode (primary bug)

The code explicitly skips `accumulate_reaction_on_bodies()` when the proxy joint is active
(`dexsuite_3dg_newton_manager.py`, line 382):

```python
if not _use_proxy and _contact_handler is not None:
    _contact_handler.accumulate_reaction_on_bodies(_contact_coeff)
```

`accumulate_reaction_on_bodies()` is the **only mechanism** that writes particle contact
reaction forces back to `body_f` (Newton rigid body force buffers). It implements Newton's
3rd law: same contact gradient, opposite sign, applied to the rigid body. Kaolin's reference
Franka example relies on this call for the coupling to work.

In proxy mode the two coupling directions are:

| Direction | Mechanism | Status |
|-----------|-----------|--------|
| Robot → Simplicits | `model.collide(state_0)` uses robot body positions; particles feel contact force inside `SimplicitsSolver.step()` | ✓ Works |
| Simplicits → Robot | `accumulate_reaction_on_bodies()` writes to `body_f`, consumed by next MuJoCo step | ✗ **Disabled** |

The robot reaches the Simplicits cube but feels **no resistance** from the particles. The only
resistance comes from the proxy box's LCP contact, which is decoupled from the Simplicits
solver (see Problem 2).

### Problem 2 — The proxy box is effectively kinematic despite being labeled dynamic

The proxy box has `is_kinematic=False` and mass 0.5 kg, so MuJoCo distributes contact impulse
between the robot finger and the proxy. However, the proxy position is **teleported back to
the Simplicits CoM at the start of every substep** (`_update_proxy_body_state()`). From
MuJoCo's perspective across multiple substeps:

- Substep i: proxy teleported to CoM_i, robot pushes it, MuJoCo computes contact forces
- Substep i+1: proxy teleported to CoM_{i+1}, regardless of where MuJoCo moved it

The proxy's post-step velocity and displacement are **discarded**. This means the impulse that
should decelerate the cube never propagates to the Simplicits solver.

The Simplicits particles do move (because `model.collide(state_0)` uses the post-rigid-step
proxy shape position), but this is only a secondary geometric penetration effect, not the
actual contact impulse. The proxy intercepts the robot's force but does not relay it to the
particles.

### Problem 3 — Contact stiffness distribution shift between training and deployment

The policy was trained with a rigid cube using MuJoCo's default contact stiffness
(`ke ≈ 1e5–1e6 N/m`). In Simplicits mode the effective soft contact stiffness is:

```
F ≈ soft_contact_ke × soft_contact_coeff × penetration
  = 1e4 × 0.05 × d = 500 N/m × d
```

This is 100–2000× softer than rigid-body training contacts. The proxy box `ke = 5e4` gives
the robot hard contact at the cube surface, but the particles yield much more softly. The
policy interprets contact force signals to decide when to close/hold fingers — with a
completely different effective stiffness the learned finger-closing behaviour fails.

---

## Is deeper Newton/MuJoCo integration feasible?

### Level 1 — Fix the current two-phase coupling (feasible now, no framework changes)

This is the most tractable path. The Kaolin `SimplicitsParticleNewtonShapeSoftContact` handler
already implements correct bidirectional coupling — it just is not wired up in proxy mode.

**What needs to change:**

1. **Enable `accumulate_reaction_on_bodies()` unconditionally.** Remove the `if not _use_proxy`
   guard (manager line 382). Call it after every Simplicits step. Forces land in `body_f` and
   are consumed by the next rigid step (one-substep delay, acceptable for ≥ 4 substeps at
   `dt/4 ≈ 2 ms`).

2. **Make the proxy kinematic or remove it.** With both LCP contact and soft-penalty reactions
   active simultaneously the forces are double-counted. Setting `is_kinematic=True` in
   `simplicits_assembly.py` (line 331) makes MuJoCo direct all contact impulse back to the
   robot (infinite proxy mass); the soft-penalty path then handles the deformable coupling
   exclusively. Alternatively set proxy mass very small (e.g. 0.001 kg).

3. **Match contact stiffness to training.** Either:
   - Retrain with the `simplicits_matched` preset (`object_contact_ke=500, kd=45`), or
   - Raise `soft_contact_ke × soft_contact_coeff` toward `~1e4–1e5` effective stiffness to
     match rigid training (risk of solver instability above ~1e5).

4. **Verify object mass matches training.** Simplicits computes mass from
   `density × approximate_volume`. Confirm this matches the trained rigid cube mass
   (default 0.5 kg). If not, adjust `SimplicitsObjectCfg.density`.

### Level 2 — MuJoCo custom passive forces (requires Newton framework changes)

Newton's MuJoCo wrapper populates `data.xfrc_applied` (external forces on bodies) before
calling `mj_step`. Simplicits contact forces on robot bodies could be injected here before the
rigid step, making the coupling synchronous (no one-substep delay) and consistent with
MuJoCo's LCP.

This requires modifying `NewtonManager._simulate` or the MuJoCo wrapper (outside the task
folder). It also requires running a partial Simplicits solve (contact force only, not full
implicit integration) before the rigid step, which breaks the Simplicits solver's convergence
guarantees.

**Verdict:** Feasible but requires framework changes. The benefit over Level 1 is marginal for
rigid Simplicits (1 handle).

### Level 3 — Joint Newton-Raphson solve (full implicit coupling)

True implicit coupling would embed Simplicits contact energy into MuJoCo's constraint system
— solving robot + soft-body dynamics simultaneously in one KKT system. This is what PhysX 5
FEM and Flex do. It requires:

- Linearising Simplicits contact forces around the current robot configuration.
- Forming block-coupled constraint matrices.
- Modifying MuJoCo's constraint solver to accept external soft-contact contributions.

**Verdict:** Technically correct but requires fundamental MuJoCo changes. Not feasible without
rewriting Newton's physics pipeline. Not necessary for rigid Simplicits (1 handle) since a
well-tuned staggered coupling achieves nearly the same result.

---

## Would fixing the coupling actually solve the grasping problem?

Yes, for the **rigid Simplicits (1 handle)** case, with high confidence.

Rigid Simplicits means the deformable object has exactly 1 DOF (an affine transform = rotation
+ translation). This is mathematically equivalent to a rigid body under the Simplicits
formulation. The effective dynamics should be **indistinguishable from a rigid cube** if:

1. Object mass and inertia match the trained cube.
2. Contact forces are bidirectional and stiffness-matched.
3. The policy sees the same observation format (already handled by `SimplicitsObjectAdapter`).

The policy should then grasp successfully because it is effectively grasping a rigid body,
just one simulated through the Simplicits path.

---

## Applied fixes (Level 1)

All three problems have been addressed. The proxy box has been removed entirely in favour of
the clean bidirectional soft-penalty coupling that Kaolin's reference Franka example uses.

| # | Fix | Files changed |
|---|-----|---------------|
| 1 | **Proxy box removed**: no more dynamic free-joint body that intercepts and discards contact impulse. The robot's collision geometry now drives Simplicits contact directly via `model.collide()`. | `simplicits_assembly.py`, `dexsuite_3dg_newton_manager.py` |
| 2 | **`accumulate_reaction_on_bodies()` always called**: after every Simplicits substep, equal-and-opposite contact wrenches are written to `body_f` so the robot feels resistance on the next rigid step. | `dexsuite_3dg_newton_manager.py` |
| 3 | **Proxy cfg fields removed**: `proxy_mass`, `proxy_contact_ke`, `proxy_contact_kd` removed from `SimplicitsObjectCfg`. Tuning guide updated to reflect soft-penalty-only coupling. | `simplicits_cfg.py` |

### Remaining open item — contact stiffness distribution shift

The policy was trained with a rigid cube whose contact stiffness is much stiffer than the
default Simplicits soft-penalty effective stiffness
(`soft_contact_ke × soft_contact_coeff = 1e4 × 0.05 = 500 N/m`).

Options (either is sufficient):
- **Retrain** using the `simplicits_matched` physics preset
  (`object_contact_ke=500, kd=45`), which applies the same soft-contact stiffness to the
  rigid training cube.
- **Increase** `soft_contact_coeff` (e.g. `0.05 → 1.0`) to raise the effective stiffness
  toward the rigid training regime; verify solver stability at each step.

---

## Key files

| File | Purpose |
|------|---------|
| `config/kuka_allegro/physic/newton/dexsuite_3dg_newton_manager.py` | Two-phase step, proxy teleport, `accumulate_reaction_on_bodies` call site |
| `config/kuka_allegro/physic/newton/simplicits_assembly.py` | Proxy box construction (`is_kinematic`, mass, ke, kd) |
| `config/kuka_allegro/physic/kaolin/simplicits_cfg.py` | All contact/solver tuning parameters |
| `kaolin/kaolin/experimental/newton/collisions.py` | `SimplicitsParticleNewtonShapeSoftContact.accumulate_reaction_on_bodies()` |
| `kaolin/kaolin/experimental/newton/solver.py` | `SimplicitsSolver.step()` — reads Newton body state, writes particle state |
| `kaolin/examples/tutorial/physics/newton_franka_coupling.ipynb` | Reference example with correct bidirectional coupling |
