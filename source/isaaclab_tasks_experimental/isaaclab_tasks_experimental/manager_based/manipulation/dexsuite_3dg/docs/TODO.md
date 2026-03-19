# Isaac-Dexsuite-3dg-Kuka-Allegro TODO

- [x] Step 1: Kaolin config + factory mesh → rigid SimplicitsObject (tests in `test/physic/kaolin/`).
- [x] Step 2: Rigid proto from USD excluding Object (`build_rigid_proto_excluding_object`, tests in `test/physic/newton/`).
- [x] Step 3: SimplicitsModelBuilder assembly (`build_multi_env_simplicits_model`, including single-env via one path; tests in `test/physic/assembly/`).
- [x] Step 4: Multi-env SimplicitsModelBuilder assembly (`build_multi_env_simplicits_model`, tests in `test/physic/assembly/test_simplicits_multi_env.py`).
- [ ] Add Kaolin dependency to IsaacLab
- [ ] Replace SimplicitObjectCfg and create_rigid_simplicits_object_from_mesh by loading simplicits object from USD
- [ ] **Simplicits memory:** In `dexsuite_3dg_newton_manager`, drop `_particle_rest_q`; store `T_reset` (4×4 or pose) per env on reset and recompute Kabsch rest on demand as `transform_points_mat4(p_build_slice, T_reset[e] @ T_build_inv[e])` (uses `_simplicits_particle_q_build`).
- [ ] Restore default initialization state and randomization. Beware of the cube scaling
