# Isaac-Dexsuite-3dg-Kuka-Allegro TODO

- [x] Step 1: Kaolin config + factory mesh → rigid SimplicitsObject (tests in `test/physic/kaolin/`).
- [x] Step 2: Rigid proto from USD excluding Object (`build_rigid_proto_excluding_object`, tests in `test/physic/newton/`).
- [x] Step 3: Single-env SimplicitsModelBuilder assembly (`build_single_env_simplicits_model`, tests in `test/physic/assembly/`).
- [x] Step 4: Multi-env SimplicitsModelBuilder assembly (`build_multi_env_simplicits_model`, tests in `test/physic/assembly/test_simplicits_multi_env.py`).
- [ ] Add Kaolin dependency to IsaacLab
- [ ] Replace SimplicitObjectCfg and create_rigid_simplicits_object_from_mesh by loading simplicits object from USD
