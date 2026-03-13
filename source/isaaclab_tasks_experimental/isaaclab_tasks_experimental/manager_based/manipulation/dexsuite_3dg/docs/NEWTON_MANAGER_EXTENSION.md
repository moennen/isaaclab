# Extending NewtonManager for Dexsuite 3dg (without changing IsaacLab)

## Why this is non-trivial

- `SimulationContext` uses the physics config's `class_type` and calls `physics_manager.initialize(self)`, so a **subclass** of `NewtonManager` can be used.
- All Newton assets (articulation, contact sensor, etc.) do `from isaaclab_newton.physics import NewtonManager` and call `NewtonManager.get_model()`, `NewtonManager.get_state_0()`, etc. So they use the **name** `NewtonManager` in that module, not “whatever class was initialized.”
- If you only set `class_type = MyNewtonManager` in config, then `MyNewtonManager.initialize()` runs and state lives on `MyNewtonManager`, but assets still reference `NewtonManager` and would see uninitialized state.

So you must make the name `NewtonManager` in `isaaclab_newton.physics` refer to your subclass when running this task in Newton mode.

## Recommended approach (no changes to IsaacLab)

1. **Subclass `NewtonManager`** in the experimental package (e.g. `Dexsuite3dgNewtonManager`) and override only the methods you need (e.g. `step`, `reset`, `_simulate`, or custom hooks).

2. **Custom physics config**  
   Define a config (e.g. `Dexsuite3dgNewtonCfg`) that extends `NewtonCfg` and sets  
   `class_type = "{DIR}.dexsuite_3dg_newton_manager:Dexsuite3dgNewtonManager"`  
   so that `SimulationContext` uses your manager class.

3. **Use your config for the newton preset**  
   In `KukaAllegroPhysicsCfg`, set `newton = Dexsuite3dgNewtonCfg(...)` (with the same solver/params as now) so that when `presets=newton` the sim uses your manager class.

4. **Patch `NewtonManager` before the sim is created**  
   Newton assets must see your class when they import `NewtonManager`. So **before** `SimulationContext(self.cfg.sim)` is created, patch the module:

   - In a **custom env class** that subclasses `ManagerBasedRLEnv` (e.g. `Dexsuite3dgManagerBasedRLEnv`):
     - In `__init__`, **before** calling `super().__init__(...)`, if the resolved physics config’s `class_type` is your `Dexsuite3dgNewtonManager`, do:
       - `import isaaclab_newton.physics as _newton_physics`
       - `_newton_physics.NewtonManager = Dexsuite3dgNewtonManager`
     - Then call `super().__init__(...)` as usual.

   - Register the gym env with `entry_point` pointing to this custom env class instead of `ManagerBasedRLEnv`, so your `__init__` (and thus the patch) always runs for the 3dg task.

5. **Keep everything in the experimental package**  
   - `Dexsuite3dgNewtonManager` and `Dexsuite3dgNewtonCfg` live under `config/kuka_allegro/physic/newton/`.
   - The custom env class and gym registration live in the dexsuite_3dg package. No changes to `isaaclab` or `isaaclab_newton`.

## File layout (skeleton)

- `.../config/kuka_allegro/physic/newton/dexsuite_3dg_newton_manager.py`  
  - `Dexsuite3dgNewtonManager(NewtonManager)` with overrides (e.g. `step`, `reset`, or `_simulate`).
- `.../config/kuka_allegro/physic/newton/dexsuite_3dg_newton_cfg.py`  
  - `Dexsuite3dgNewtonCfg(NewtonCfg)` with `class_type = "{DIR}.dexsuite_3dg_newton_manager:Dexsuite3dgNewtonManager"`.
- `.../config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py`  
  - In `KukaAllegroPhysicsCfg`, set `newton = Dexsuite3dgNewtonCfg()` (import from `.physic.newton`).
- `.../dexsuite_3dg/dexsuite_3dg_env.py`  
  - `Dexsuite3dgManagerBasedRLEnv(ManagerBasedRLEnv)` that patches `isaaclab_newton.physics.NewtonManager` when `cfg.sim.physics.class_type` is `Dexsuite3dgNewtonManager`, then calls `super().__init__(...)`.
- `.../config/kuka_allegro/__init__.py`  
  - Register gym envs with  
    `entry_point="...dexsuite_3dg_env:Dexsuite3dgManagerBasedRLEnv"` (and pass the same `env_cfg_entry_point` / agent configs as now).

## Summary

- **Extend:** Subclass `NewtonManager` and a custom `NewtonCfg` in the experimental package.
- **Wire:** Use that config for the newton preset and a custom env class that patches `NewtonManager` before `SimulationContext` is created and before any Newton assets are used.
- **Scope:** All changes stay in `isaaclab_tasks_experimental`; no edits to IsaacLab core or `isaaclab_newton`.
