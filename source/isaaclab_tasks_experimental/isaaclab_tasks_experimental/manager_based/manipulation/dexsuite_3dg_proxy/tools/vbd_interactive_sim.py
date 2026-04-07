# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
VBD Interactive Simulation Tool

Load a surface mesh, optionally decimate it, convert to a tetrahedral mesh,
then run Newton VBD soft-body physics with gravity, a static ground plane,
and a static cube. Middle-click and drag to pick and move particles.

Controls:
    Middle-click drag   Pick nearest particle and drag it
    Left-click drag     Orbit camera
    Right-click drag    Pan camera
    Mouse scroll        Zoom

Usage:
    python vbd_interactive_sim.py mesh.obj
    python vbd_interactive_sim.py mesh.ply
    python vbd_interactive_sim.py already_tetrahedralized.msh
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import threading
import time

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ImportError:
    print("polyscope not found: pip install polyscope", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
except ImportError:
    print("trimesh not found: pip install trimesh", file=sys.stderr)
    sys.exit(1)

try:
    import warp as wp
except ImportError:
    print("warp-lang not found: pip install warp-lang", file=sys.stderr)
    sys.exit(1)

# ── Warp kernels ───────────────────────────────────────────────────────────────

@wp.kernel
def _wp_set_float(arr: wp.array(dtype=float), idx: int, val: float):
    if wp.tid() == 0:
        arr[idx] = val


@wp.kernel
def _wp_set_vec3(arr: wp.array(dtype=wp.vec3), idx: int, val: wp.vec3):
    if wp.tid() == 0:
        arr[idx] = val


@wp.kernel
def _wp_zero_vec3(arr: wp.array(dtype=wp.vec3)):
    arr[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


# Newton must be on the path (e.g. from the same conda/venv as IsaacLab).
# Fall back to the sibling repo path used on this development machine.
try:
    import newton
    import newton.solvers
except ImportError:
    _newton_root = "/mnt/dev/isaac-newton3/newton"
    if os.path.isdir(_newton_root) and _newton_root not in sys.path:
        sys.path.insert(0, _newton_root)
    try:
        import newton
        import newton.solvers
    except ImportError:
        print("newton not found on sys.path", file=sys.stderr)
        sys.exit(1)


# ── Phase state machine ────────────────────────────────────────────────────────
# "view"       : displaying input surface mesh; UI shows decimation/tet sliders
# "converting" : background thread running decimation + tetrahedralization
# "ready"      : background thread done; main thread must build the simulation
# "simulate"   : Newton VBD simulation running
_phase = "view"
_convert_status = ""
_convert_result: tuple[np.ndarray, np.ndarray] | None = None  # (verts, tets)

# ── UI parameters (mutable single-element lists for imgui in-place update) ────
_ui_face_count = [2000]
_ui_edge_length = [0.05]

# Tetrahedralization parameters (fTetWild / wildmeshing)
_ui_stop_quality = [10.0]   # AMIPS energy threshold; lower = better tets, more of them, slower
_ui_log_epsilon = [-3.0]    # envelope size = 10^epsilon × bbox diagonal; smaller = closer to input surface
_ui_coarsen = [False]       # when False, interior tets are preserved; when True, fTetWild strips them out
_ui_floodfill = [False]     # flood-fill interior detection; helps for imperfect/open surfaces

# Isotropic remeshing (pymeshlab) — uniform face sizes → uniform particle coverage
_ui_iso_remesh = [True]        # enable isotropic remesh step before tetrahedralization
_ui_iso_iterations = [5]       # Botsch-Kobbelt iterations (split + collapse + flip + smooth)
_ui_iso_feature_deg = [30.0]   # dihedral angle [°] below which edges are treated as features

# Material parameters (log10-scale sliders; applied on rebuild/reset)
_ui_log_k_mu = [5.0]       # k_mu     = 10^5 Pa
_ui_log_k_lambda = [5.0]   # k_lambda = 10^5 Pa
_ui_k_damp = [1e-3]        # Rayleigh stiffness damping
_ui_density = [1e3]        # particle density  [kg/m³]

# Contact parameters (applied live each frame)
_ui_log_contact_ke = [3.0] # soft_contact_ke = 10^3 N/m
_ui_contact_kd = [10.0]    # soft_contact_kd
_ui_contact_mu = [0.8]     # friction coefficient

# ── Loaded surface mesh ────────────────────────────────────────────────────────
_surf_verts: np.ndarray | None = None
_surf_faces: np.ndarray | None = None

# ── Newton simulation objects ──────────────────────────────────────────────────
_model = None
_solver = None
_state_0 = None
_state_1 = None
_control = None
_contacts = None
_tri_faces: np.ndarray | None = None  # (F, 3) surface triangles from model
_last_good_pos: np.ndarray | None = None  # last NaN-free particle positions (CPU float32)
_initial_particle_q: np.ndarray | None = None  # particle positions at t=0 (for reset)
_sim_verts: np.ndarray | None = None  # verts as passed to _build_simulation (for rebuild)
_sim_tets: np.ndarray | None = None
_tet_save_path: str = "output.msh"   # derived from input path in main()
_tet_usd_path: str = "output_tet.usda"  # derived from input path in main()

_SIM_FPS = 60
_SIM_SUBSTEPS = 10
_SIM_DT = 1.0 / (_SIM_FPS * _SIM_SUBSTEPS)

# ── Solver performance parameters (live) ──────────────────────────────────────
_ui_substeps = [10]    # substeps per rendered frame
_ui_iterations = [30]  # VBD iterations per substep
_sim_step_ms = 0.0  # exponential moving average of _sim_step() wall time [ms]

# Cube geometry (static collision shape + visual)
_CUBE_POS = (0.35, 0.15, 0.0)
_CUBE_HALF = 0.15

# ── Picking state ──────────────────────────────────────────────────────────────
# Picking uses the kinematic-particle approach: on pick-start the particle's
# inv_mass is zeroed so VBD treats it as a fixed boundary condition.  Each
# substep we teleport it to the mouse-projected target position; VBD's elastic
# constraints drag the rest of the mesh along.  On release, the original mass
# is restored and the particle gets the correct velocity from the final step.
_pick_idx = -1
_pick_depth = 1.0           # depth along camera look-dir to the picked particle
_pick_target: np.ndarray | None = None   # world-space target position this frame
_pick_saved_inv_mass = 0.0  # inv_mass value before we zeroed it
_pick_saved_mass = 0.0      # mass value before we zeroed it
_was_right_down = False

# ── Polyscope structure names ──────────────────────────────────────────────────
_PS_INPUT = "input_mesh"
_PS_SURF = "soft_body"
_PS_PARTICLES = "particles"
_PS_GROUND = "ground"
_PS_CUBE = "cube"

# ── Particle visualisation ─────────────────────────────────────────────────────
_ui_show_particles = [False]
_ui_particle_radius = [0.005]   # world-space [m], default matches physics particle_radius


# ── Mesh helpers ───────────────────────────────────────────────────────────────

def _ground_mesh(size: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return v, f


def _box_mesh(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return v, f


# ── Camera ray ─────────────────────────────────────────────────────────────────

def _pixel_ray(mx: float, my: float, W: float, H: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (ray_origin, ray_dir) for screen pixel (mx, my)."""
    cam = ps.get_view_camera_parameters()
    pos = np.asarray(cam.get_position())
    look = np.asarray(cam.get_look_dir())
    right = np.asarray(cam.get_right_dir())
    up = np.asarray(cam.get_up_dir())
    fov_deg = cam.get_fov_vertical_deg()
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    aspect = W / max(H, 1.0)
    ndc_x = 2.0 * mx / max(W, 1.0) - 1.0
    ndc_y = 1.0 - 2.0 * my / max(H, 1.0)
    d = look + ndc_x * aspect * tan_half * right + ndc_y * tan_half * up
    return pos, d / np.linalg.norm(d)


def _nearest_particle(ray_o: np.ndarray, ray_d: np.ndarray, pts: np.ndarray) -> int:
    """Return index of particle nearest to the ray."""
    offset = pts - ray_o
    t = offset @ ray_d
    perp = np.linalg.norm(offset - t[:, None] * ray_d, axis=1)
    perp[t <= 0] = np.inf
    return int(np.argmin(perp))


# ── Conversion (background thread) ────────────────────────────────────────────

def _run_conversion(
    surf_verts: np.ndarray,
    surf_faces: np.ndarray,
    face_count: int,
    edge_length: float,
    stop_quality: float,
    epsilon: float,
    coarsen: bool,
    floodfill: bool,
    iso_remesh: bool,
    iso_iterations: int,
    iso_feature_deg: float,
) -> None:
    global _convert_status, _convert_result, _phase

    try:
        import fast_simplification  # noqa: F401 — required by trimesh QEM
    except ImportError:
        _convert_status = "ERROR: pip install fast-simplification"
        _phase = "view"
        return

    try:
        import wildmeshing
    except ImportError:
        _convert_status = "ERROR: pip install wildmeshing"
        _phase = "view"
        return

    try:
        import meshio
    except ImportError:
        _convert_status = "ERROR: pip install meshio"
        _phase = "view"
        return

    try:
        _convert_status = f"Decimating to {face_count} faces..."
        mesh = trimesh.Trimesh(vertices=surf_verts, faces=surf_faces, process=False)
        if len(surf_faces) > face_count:
            mesh = mesh.simplify_quadric_decimation(face_count=face_count)
        # Ensure watertight for tetrahedralization
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
        if not mesh.is_watertight:
            mesh.fill_holes()

        # ── Isotropic remeshing ────────────────────────────────────────────────
        # Redistributes vertices so all surface triangles have similar area,
        # giving each particle the same contact footprint.
        # Requires pymeshlab (pip install pymeshlab).
        if iso_remesh:
            try:
                import pymeshlab
                _convert_status = (
                    f"Isotropic remesh ({len(mesh.faces)} faces, "
                    f"{iso_iterations} iterations)..."
                )
                bbox_diag = float(np.linalg.norm(
                    mesh.bounds[1] - mesh.bounds[0]
                ))
                target_len = edge_length * bbox_diag
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype(np.float64),
                    face_matrix=mesh.faces.astype(np.int32),
                ))
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=iso_iterations,
                    targetlen=pymeshlab.PureValue(target_len),
                    featuredeg=iso_feature_deg,
                )
                m = ms.current_mesh()
                mesh = trimesh.Trimesh(
                    vertices=m.vertex_matrix(),
                    faces=m.face_matrix(),
                    process=True,
                )
                if not mesh.is_watertight:
                    mesh.fill_holes()
            except ImportError:
                _convert_status = (
                    "WARNING: pymeshlab not found — skipping isotropic remesh. "
                    "pip install pymeshlab for uniform face sizes."
                )
                # Give the user a moment to read the warning
                import time as _time; _time.sleep(1.5)

        _convert_status = (
            f"Tetrahedralizing ({len(mesh.vertices)} surface verts, "
            f"edge_length_r={edge_length}, stop_quality={stop_quality}, "
            f"epsilon={epsilon:.2e}, coarsen={coarsen}, floodfill={floodfill})..."
        )

        tmp_in = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        tmp_out = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name
        try:
            mesh.export(tmp_in)
            tetra = wildmeshing.Tetrahedralizer(
                stop_quality=stop_quality,
                epsilon=epsilon,
                edge_length_r=edge_length,
                coarsen=coarsen,
            )
            tetra.load_mesh(tmp_in)
            tetra.tetrahedralize()
            tetra.save(tmp_out, floodfill=floodfill)
            m = meshio.read(tmp_out)
            verts = m.points.astype(np.float64)
            tets = None
            for cb in m.cells:
                if cb.type == "tetra":
                    tets = cb.data.astype(np.int32)
                    break
            if tets is None:
                _convert_status = "ERROR: tetrahedralization produced no tets"
                _phase = "view"
                return
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        _convert_status = f"Done: {len(verts)} verts, {len(tets)} tets"
        _convert_result = (verts, tets)
        _phase = "ready"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _convert_status = f"ERROR: {exc}"
        _phase = "view"


# ── Build Newton simulation (main thread) ──────────────────────────────────────

def _build_simulation(verts: np.ndarray, tets: np.ndarray) -> None:
    global _model, _solver, _state_0, _state_1, _control, _contacts, _tri_faces
    global _last_good_pos, _initial_particle_q, _sim_verts, _sim_tets
    # Store original verts/tets so the UI "Rebuild" button can re-invoke this function.
    _sim_verts = verts.copy()
    _sim_tets = tets.copy()

    # Center horizontally and lift above ground
    center_xz = np.array([verts[:, 0].mean(), 0.0, verts[:, 2].mean()])
    verts = verts - center_xz
    min_y = verts[:, 1].min()
    verts[:, 1] -= min_y
    verts[:, 1] += 0.5  # start 0.5 m above ground

    # Estimate particle radius from mesh scale
    span = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    p_radius = max(0.005, span / (len(verts) ** (1 / 3)) * 0.3)

    # Build Newton model (Y-up to match polyscope)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    builder.add_ground_plane(height=0.0)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*_CUBE_POS), wp.quat_identity()),
        hx=_CUBE_HALF,
        hy=_CUBE_HALF,
        hz=_CUBE_HALF,
    )
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[tuple(v) for v in verts],
        indices=tets.flatten().tolist(),
        density=_ui_density[0],
        k_mu=10.0 ** _ui_log_k_mu[0],
        k_lambda=10.0 ** _ui_log_k_lambda[0],
        k_damp=_ui_k_damp[0],
        particle_radius=p_radius,
    )
    builder.color()

    model = builder.finalize()
    model.soft_contact_ke = 10.0 ** _ui_log_contact_ke[0]
    model.soft_contact_kd = _ui_contact_kd[0]
    model.soft_contact_mu = _ui_contact_mu[0]

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=30,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )

    _model = model
    _solver = solver
    _state_0 = model.state()
    _state_1 = model.state()
    _control = model.control()
    _contacts = model.contacts()
    _tri_faces = model.tri_indices.numpy().reshape(-1, 3)

    # Seed reset/recovery checkpoints with the initial positions
    _initial_particle_q = _state_0.particle_q.numpy().copy()
    _last_good_pos = _initial_particle_q.copy()

    # Register polyscope structures
    pos_np = _state_0.particle_q.numpy()
    sm = ps.register_surface_mesh(_PS_SURF, pos_np, _tri_faces, smooth_shade=True)
    sm.set_color((0.6, 0.75, 0.9))

    pc = ps.register_point_cloud(_PS_PARTICLES, pos_np)
    pc.set_radius(_ui_particle_radius[0], relative=False)
    pc.set_color((1.0, 0.5, 0.1))
    pc.set_enabled(_ui_show_particles[0])

    gv, gf = _ground_mesh(size=3.0)
    gm = ps.register_surface_mesh(_PS_GROUND, gv, gf, smooth_shade=False)
    gm.set_color((0.35, 0.35, 0.35))
    gm.set_back_face_policy("identical")

    bv, bf = _box_mesh(*_CUBE_POS, _CUBE_HALF, _CUBE_HALF, _CUBE_HALF)
    bm = ps.register_surface_mesh(_PS_CUBE, bv, bf, smooth_shade=False)
    bm.set_color((0.85, 0.45, 0.2))

    print(f"Simulation ready: {model.particle_count} particles, {model.tri_count} surface triangles")


# ── Simulation reset ──────────────────────────────────────────────────────────

def _do_reset() -> None:
    """Restore particles to their t=0 positions and zero all velocities."""
    global _pick_idx, _pick_target, _was_right_down
    if _initial_particle_q is None:
        return
    # Cancel any active pick first (restore mass before resetting positions).
    if _pick_idx >= 0:
        wp.launch(_wp_set_float, dim=1,
                  inputs=[_model.particle_inv_mass, _pick_idx, _pick_saved_inv_mass],
                  device=_model.particle_inv_mass.device)
        wp.launch(_wp_set_float, dim=1,
                  inputs=[_model.particle_mass, _pick_idx, _pick_saved_mass],
                  device=_model.particle_mass.device)
        _pick_idx = -1
        _pick_target = None
        _was_right_down = False
    good = wp.array(_initial_particle_q, dtype=wp.vec3, device=_state_0.particle_q.device)
    wp.copy(_state_0.particle_q, good)
    wp.copy(_state_1.particle_q, good)
    for qd in (_state_0.particle_qd, _state_1.particle_qd):
        wp.launch(_wp_zero_vec3, dim=_model.particle_count, inputs=[qd], device=qd.device)
    ps.get_surface_mesh(_PS_SURF).update_vertex_positions(_initial_particle_q)
    if _ui_show_particles[0]:
        ps.get_point_cloud(_PS_PARTICLES).update_point_positions(_initial_particle_q)


# ── Save tet mesh ─────────────────────────────────────────────────────────────

def _save_tet_mesh() -> None:
    """Write the current tet mesh to a .msh file next to the input mesh."""
    if _sim_verts is None or _sim_tets is None:
        print("[save] no tet mesh available", flush=True)
        return
    try:
        import meshio
    except ImportError:
        print("[save] meshio not found: pip install meshio", flush=True)
        return
    # Derive output path from the original input mesh path (stored in main()).
    out_path = _tet_save_path
    m = meshio.Mesh(
        points=_sim_verts.astype(np.float64),
        cells=[("tetra", _sim_tets.astype(np.int32))],
    )
    meshio.write(out_path, m)
    print(f"[save] tet mesh written to {out_path}", flush=True)


def _save_tet_usd() -> None:
    """Save tet mesh + current material parameters to a .usda file.

    The file can be loaded by the task via proxy_newton_manager._load_tet_mesh()
    which reads the custom ``vbd:`` attributes and overrides the Python cfg defaults.
    """
    if _sim_verts is None or _sim_tets is None:
        print("[save] no tet mesh available", flush=True)
        return
    try:
        from pxr import Sdf, Usd, Vt
    except ImportError:
        print("[save] pxr (USD) not found — install omniverse or usd-core", flush=True)
        return

    out_path = _tet_usd_path
    stage = Usd.Stage.CreateNew(out_path)
    stage.SetMetadata("upAxis", "Y")

    prim = stage.DefinePrim("/TetMesh", "Xform")

    # -- Material parameters --------------------------------------------------
    mat_params = {
        "vbd:k_mu":            float(10.0 ** _ui_log_k_mu[0]),
        "vbd:k_lambda":        float(10.0 ** _ui_log_k_lambda[0]),
        "vbd:k_damp":          float(_ui_k_damp[0]),
        "vbd:density":         float(_ui_density[0]),
        "vbd:soft_contact_ke": float(10.0 ** _ui_log_contact_ke[0]),
        "vbd:soft_contact_kd": float(_ui_contact_kd[0]),
        "vbd:soft_contact_mu": float(_ui_contact_mu[0]),
    }
    for name, val in mat_params.items():
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attr.Set(val)

    # -- Geometry -------------------------------------------------------------
    verts_vt = Vt.Vec3fArray([tuple(float(x) for x in v) for v in _sim_verts])
    attr = prim.CreateAttribute("vbd:vertices", Sdf.ValueTypeNames.Point3fArray)
    attr.Set(verts_vt)

    attr = prim.CreateAttribute("vbd:tet_indices", Sdf.ValueTypeNames.IntArray)
    attr.Set(Vt.IntArray(_sim_tets.flatten().tolist()))

    stage.GetRootLayer().Save()
    print(f"[save] USD tet mesh written to {out_path}", flush=True)
    print(f"       k_mu={mat_params['vbd:k_mu']:.3g}  k_lambda={mat_params['vbd:k_lambda']:.3g}"
          f"  k_damp={mat_params['vbd:k_damp']:.3g}  density={mat_params['vbd:density']:.3g}",
          flush=True)


# ── Simulation step ────────────────────────────────────────────────────────────

def _sim_step() -> None:
    global _state_0, _state_1
    _solver.iterations = _ui_iterations[0]
    for _ in range(_ui_substeps[0]):
        _state_0.clear_forces()
        if _pick_idx >= 0 and _pick_target is not None:
            # Teleport the kinematic particle to the current mouse target.
            # inv_mass was set to 0 on pick-start, so VBD treats it as a fixed
            # boundary condition and propagates the displacement elastically.
            wp.launch(
                _wp_set_vec3, dim=1,
                inputs=[_state_0.particle_q, _pick_idx, wp.vec3(*_pick_target.tolist())],
                device=_state_0.particle_q.device,
            )
        _model.collide(_state_0, _contacts)
        _solver.step(_state_0, _state_1, _control, _contacts, _SIM_DT)
        _state_0, _state_1 = _state_1, _state_0


# ── Polyscope per-frame callback ───────────────────────────────────────────────

def _callback() -> None:
    global _phase, _convert_result
    global _pick_idx, _pick_depth, _pick_target, _pick_saved_inv_mass, _pick_saved_mass, _was_right_down
    global _last_good_pos, _initial_particle_q, _sim_step_ms

    # ── Phase transition: ready → simulate (must happen on main thread) ────────
    if _phase == "ready":
        verts, tets = _convert_result
        _convert_result = None
        if ps.has_surface_mesh(_PS_INPUT):
            ps.remove_surface_mesh(_PS_INPUT)
        _build_simulation(verts, tets)
        _phase = "simulate"
        return

    # ── Phase: view ────────────────────────────────────────────────────────────
    if _phase == "view":
        psim.TextUnformatted("Surface Mesh")
        if _surf_verts is not None:
            psim.TextUnformatted(f"  {len(_surf_verts)} verts, {len(_surf_faces)} faces")
        psim.Separator()

        psim.TextUnformatted("Decimation target faces:")
        psim.TextUnformatted("  Fewer faces = faster tet conversion and lighter sim.")
        changed, val = psim.SliderInt("##faces_slider", _ui_face_count[0], 200, 20000)
        if changed:
            _ui_face_count[0] = max(200, val)
        psim.SameLine()
        changed, val = psim.InputInt("Faces##faces_input", _ui_face_count[0])
        if changed:
            _ui_face_count[0] = max(200, val)

        psim.Separator()
        psim.TextUnformatted("Tet edge length (fraction of bbox diagonal):")
        psim.TextUnformatted("  Controls tet mesh resolution. Smaller = more particles,")
        psim.TextUnformatted("  finer deformation detail, but heavier simulation.")
        psim.TextUnformatted("  Larger = coarser mesh, faster but less detailed.")
        psim.TextUnformatted("  Typical range: 0.02 (fine) to 0.15 (coarse).")
        changed, val = psim.SliderFloat("##edge_slider", _ui_edge_length[0], 0.02, 1.0)
        if changed:
            _ui_edge_length[0] = round(max(0.01, val), 4)
        psim.SameLine()
        changed, val = psim.InputFloat("Edge length##edge_input", _ui_edge_length[0])
        if changed:
            _ui_edge_length[0] = round(max(0.01, val), 4)

        psim.Separator()
        psim.TextUnformatted("Tetrahedralization quality (fTetWild)")
        psim.TextUnformatted("  stop_quality: AMIPS energy cap. Lower = better-shaped tets,")
        psim.TextUnformatted("  more of them, slower conversion. Default 10. Range 2-200.")
        changed, val = psim.SliderFloat("stop_quality##sq", _ui_stop_quality[0], 2.0, 200.0)
        if changed:
            _ui_stop_quality[0] = val
        psim.SameLine()
        changed, val = psim.InputFloat("##sq_input", _ui_stop_quality[0])
        if changed:
            _ui_stop_quality[0] = max(1.0, val)

        psim.TextUnformatted("  epsilon: surface envelope (fraction of bbox diagonal, log10).")
        psim.TextUnformatted("  Smaller = output stays closer to input surface.")
        psim.TextUnformatted("  Default -3 (1e-3). Loosen to -2 if conversion fails.")
        changed, val = psim.SliderFloat("epsilon [log10]##eps", _ui_log_epsilon[0], -4.0, -1.0)
        if changed:
            _ui_log_epsilon[0] = round(val, 2)

        psim.TextUnformatted("  coarsen: when ON, fTetWild strips interior tets to minimise")
        psim.TextUnformatted("  count. Turn OFF to fill the volume — needed for VBD.")
        changed, val = psim.Checkbox("coarsen (strips interior)##coarsen", _ui_coarsen[0])
        if changed:
            _ui_coarsen[0] = val

        psim.TextUnformatted("  floodfill: use flood-fill instead of winding number to")
        psim.TextUnformatted("  detect interior. Helps when the surface has small holes.")
        changed, val = psim.Checkbox("floodfill##ff", _ui_floodfill[0])
        if changed:
            _ui_floodfill[0] = val

        psim.Separator()
        psim.TextUnformatted("Isotropic remesh  (requires: pip install pymeshlab)")
        psim.TextUnformatted("  Redistributes vertices so all surface triangles have similar")
        psim.TextUnformatted("  area. Each particle then covers the same contact footprint.")
        psim.TextUnformatted("  Target edge length is taken from the edge_length_r above.")
        changed, val = psim.Checkbox("Enable isotropic remesh##isorem", _ui_iso_remesh[0])
        if changed:
            _ui_iso_remesh[0] = val
        if _ui_iso_remesh[0]:
            changed, val = psim.SliderInt("iterations##isoiter", _ui_iso_iterations[0], 1, 20)
            if changed:
                _ui_iso_iterations[0] = int(val)
            changed, val = psim.SliderFloat("feature angle [deg]##isofeat", _ui_iso_feature_deg[0], 5.0, 90.0)
            if changed:
                _ui_iso_feature_deg[0] = val

        if psim.Button("Convert to Tet + Simulate"):
            _phase = "converting"
            t = threading.Thread(
                target=_run_conversion,
                args=(
                    _surf_verts, _surf_faces,
                    _ui_face_count[0], _ui_edge_length[0],
                    _ui_stop_quality[0],
                    10.0 ** _ui_log_epsilon[0],
                    _ui_coarsen[0],
                    _ui_floodfill[0],
                    _ui_iso_remesh[0],
                    _ui_iso_iterations[0],
                    _ui_iso_feature_deg[0],
                ),
                daemon=True,
            )
            t.start()
        return

    # ── Phase: converting ──────────────────────────────────────────────────────
    if _phase == "converting":
        psim.TextUnformatted("Converting...")
        psim.TextUnformatted(_convert_status)
        return

    # ── Phase: simulate ────────────────────────────────────────────────────────
    psim.TextUnformatted("VBD Simulation")
    psim.Separator()
    psim.TextUnformatted(f"Particles : {_model.particle_count}")
    psim.TextUnformatted(f"Tets      : {len(_sim_tets)}")
    psim.TextUnformatted(f"Triangles : {_model.tri_count}")
    fps_equiv = 1000.0 / _sim_step_ms if _sim_step_ms > 0 else 0.0
    psim.TextUnformatted(f"Sim step  : {_sim_step_ms:.1f} ms  ({fps_equiv:.0f} fps equiv)")
    psim.Separator()

    # ── Particle visualisation (live) ─────────────────────────────────────────
    changed, val = psim.Checkbox("Show particles##showp", _ui_show_particles[0])
    if changed:
        _ui_show_particles[0] = val
        ps.get_point_cloud(_PS_PARTICLES).set_enabled(val)
        if val:
            ps.get_point_cloud(_PS_PARTICLES).update_point_positions(
                _state_0.particle_q.numpy()
            )
    if _ui_show_particles[0]:
        psim.SameLine()
        changed, val = psim.SliderFloat("radius [m]##prad", _ui_particle_radius[0], 0.001, 0.05)
        if changed:
            _ui_particle_radius[0] = val
            ps.get_point_cloud(_PS_PARTICLES).set_radius(val, relative=False)
    psim.Separator()

    # ── Solver performance (live) ──────────────────────────────────────────────
    psim.TextUnformatted("Solver  (live)")
    changed, val = psim.SliderInt("substeps##sub", _ui_substeps[0], 1, 20)
    if changed:
        _ui_substeps[0] = int(val)
    changed, val = psim.SliderInt("iterations##itr", _ui_iterations[0], 1, 60)
    if changed:
        _ui_iterations[0] = int(val)
    psim.Separator()

    # ── Simulation controls ────────────────────────────────────────────────────
    if psim.Button("Reset"):
        _do_reset()
    psim.SameLine()
    if psim.Button("Rebuild with new material"):
        _build_simulation(_sim_verts, _sim_tets)
    psim.SameLine()
    if psim.Button("Save tet mesh"):
        _save_tet_mesh()
    psim.SameLine()
    if psim.Button("Save USD"):
        _save_tet_usd()
    psim.Separator()

    # ── Material parameters (applied on Rebuild) ───────────────────────────────
    psim.TextUnformatted("Material  (Rebuild to apply)")
    changed, val = psim.SliderFloat("k_mu [log10 Pa]##kmu", _ui_log_k_mu[0], 2.0, 6.0)
    if changed:
        _ui_log_k_mu[0] = round(val, 2)
    changed, val = psim.SliderFloat("k_lambda [log10 Pa]##klam", _ui_log_k_lambda[0], 2.0, 6.0)
    if changed:
        _ui_log_k_lambda[0] = round(val, 2)
    changed, val = psim.SliderFloat("k_damp##kdamp", _ui_k_damp[0], 0.0, 0.01)
    if changed:
        _ui_k_damp[0] = val
    changed, val = psim.SliderFloat("density [kg/m3]##dens", _ui_density[0], 100.0, 5000.0)
    if changed:
        _ui_density[0] = val
    psim.Separator()

    # ── Contact parameters (live) ──────────────────────────────────────────────
    psim.TextUnformatted("Contact  (live)")
    changed, val = psim.SliderFloat("ke [log10 N/m]##ke", _ui_log_contact_ke[0], 1.0, 5.0)
    if changed:
        _ui_log_contact_ke[0] = round(val, 2)
        _model.soft_contact_ke = 10.0 ** _ui_log_contact_ke[0]
    changed, val = psim.SliderFloat("kd##kd", _ui_contact_kd[0], 0.0, 100.0)
    if changed:
        _ui_contact_kd[0] = val
        _model.soft_contact_kd = _ui_contact_kd[0]
    changed, val = psim.SliderFloat("friction mu##mu", _ui_contact_mu[0], 0.0, 2.0)
    if changed:
        _ui_contact_mu[0] = val
        _model.soft_contact_mu = _ui_contact_mu[0]
    psim.Separator()

    psim.TextUnformatted("Middle-click drag: pick & move")
    if _pick_idx >= 0:
        psim.TextUnformatted(f"Picked particle: {_pick_idx}")

    # ── Picking: middle mouse button ───────────────────────────────────────────
    # Middle-click doesn't conflict with polyscope camera controls (left = orbit,
    # right = pan), so no camera freeze is needed and imgui always sees mouse pos.
    io = psim.GetIO()
    mid_down = bool(io.MouseDown[2])
    mx = float(io.MousePos[0])
    my = float(io.MousePos[1])
    W = float(io.DisplaySize[0])
    H = float(io.DisplaySize[1])

    if mid_down and not _was_right_down:
        # ── Pick start ────────────────────────────────────────────────────────
        ray_o, ray_d = _pixel_ray(mx, my, W, H)
        pts = _state_0.particle_q.numpy()
        # Search only among surface particles (referenced by tri_faces).
        # Interior tet vertices are not displayed; picking them has no visible effect.
        surf_indices = np.unique(_tri_faces)
        nearest_local = _nearest_particle(ray_o, ray_d, pts[surf_indices])
        _pick_idx = int(surf_indices[nearest_local])
        cam = ps.get_view_camera_parameters()
        _pick_depth = float(np.dot(pts[_pick_idx] - ray_o, np.asarray(cam.get_look_dir())))
        # Make the particle fully kinematic: forward_step checks inv_mass==0 and
        # solve_elasticity checks mass==0; both must be zero so neither kernel
        # moves the particle away from the teleported position each substep.
        _pick_saved_inv_mass = float(_model.particle_inv_mass.numpy()[_pick_idx])
        _pick_saved_mass = float(_model.particle_mass.numpy()[_pick_idx])
        wp.launch(
            _wp_set_float, dim=1,
            inputs=[_model.particle_inv_mass, _pick_idx, 0.0],
            device=_model.particle_inv_mass.device,
        )
        wp.launch(
            _wp_set_float, dim=1,
            inputs=[_model.particle_mass, _pick_idx, 0.0],
            device=_model.particle_mass.device,
        )

    if mid_down and _pick_idx >= 0:
        ray_o, ray_d = _pixel_ray(mx, my, W, H)
        raw_target = ray_o + ray_d * _pick_depth
        # Rate-limit target movement: cap displacement per frame so the elastic
        # solve never sees a sudden jump large enough to diverge to NaN.
        # 3 cm/frame at 60 fps ≈ 1.8 m/s, fast enough for responsive interaction.
        if _pick_target is not None:
            delta = raw_target - _pick_target
            dist = float(np.linalg.norm(delta))
            if dist > 0.03:
                raw_target = _pick_target + delta * (0.03 / dist)
        _pick_target = raw_target
    else:
        if not mid_down and _was_right_down:
            # ── Pick release ──────────────────────────────────────────────────
            if _pick_idx >= 0:
                # Zero velocity before restoring mass so the suddenly-freed
                # particle doesn't fly away and blow up the simulation.
                wp.launch(
                    _wp_set_vec3, dim=1,
                    inputs=[_state_0.particle_qd, _pick_idx, wp.vec3(0.0, 0.0, 0.0)],
                    device=_state_0.particle_qd.device,
                )
                wp.launch(
                    _wp_set_float, dim=1,
                    inputs=[_model.particle_inv_mass, _pick_idx, _pick_saved_inv_mass],
                    device=_model.particle_inv_mass.device,
                )
                wp.launch(
                    _wp_set_float, dim=1,
                    inputs=[_model.particle_mass, _pick_idx, _pick_saved_mass],
                    device=_model.particle_mass.device,
                )
            _pick_idx = -1
            _pick_target = None

    _was_right_down = mid_down

    # ── Step simulation ────────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    _sim_step()
    wp.synchronize()
    elapsed_ms = (time.perf_counter() - _t0) * 1e3
    _sim_step_ms = 0.9 * _sim_step_ms + 0.1 * elapsed_ms

    # ── Sync polyscope surface ─────────────────────────────────────────────────
    pos_np = _state_0.particle_q.numpy()
    if np.any(np.isnan(pos_np)):
        # Restore both states to the last NaN-free checkpoint and cancel any pick.
        # Without this, all particle positions stay NaN permanently.
        if _last_good_pos is not None:
            good = wp.array(_last_good_pos, dtype=wp.vec3, device=_state_0.particle_q.device)
            wp.copy(_state_0.particle_q, good)
            wp.copy(_state_1.particle_q, good)
        for qd in (_state_0.particle_qd, _state_1.particle_qd):
            wp.launch(_wp_zero_vec3, dim=_model.particle_count,
                      inputs=[qd], device=qd.device)
        if _pick_idx >= 0:
            wp.launch(_wp_set_float, dim=1,
                      inputs=[_model.particle_inv_mass, _pick_idx, _pick_saved_inv_mass],
                      device=_model.particle_inv_mass.device)
            wp.launch(_wp_set_float, dim=1,
                      inputs=[_model.particle_mass, _pick_idx, _pick_saved_mass],
                      device=_model.particle_mass.device)
            _pick_idx = -1
            _pick_target = None
            _was_right_down = False
    else:
        _last_good_pos = pos_np.copy()
        ps.get_surface_mesh(_PS_SURF).update_vertex_positions(pos_np)
        if _ui_show_particles[0]:
            ps.get_point_cloud(_PS_PARTICLES).update_point_positions(pos_np)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global _surf_verts, _surf_faces, _phase, _convert_result, _tet_save_path, _tet_usd_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", help="Input surface mesh (OBJ, PLY, STL) or tet mesh (.msh).")
    parser.add_argument(
        "--face-count", type=int, default=2000,
        help="Initial decimation target face count (default: 2000).",
    )
    parser.add_argument(
        "--edge-length", type=float, default=0.05,
        help="Initial tet edge length relative to bbox diagonal (default: 0.05).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.mesh):
        print(f"ERROR: file not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    _ui_face_count[0] = args.face_count
    _ui_edge_length[0] = args.edge_length
    stem = os.path.splitext(args.mesh)[0]
    _tet_save_path = stem + "_tet.msh"
    _tet_usd_path = stem + "_tet.usda"

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_background_color((0.15, 0.15, 0.15))

    ext = os.path.splitext(args.mesh)[1].lower()

    if ext in (".msh", ".msh2", ".msh4"):
        # Already a tet mesh: skip conversion, go straight to simulation
        try:
            import meshio
        except ImportError:
            print("meshio not found: pip install meshio", file=sys.stderr)
            sys.exit(1)
        m = meshio.read(args.mesh)
        verts = m.points.astype(np.float64)
        tets = None
        for cb in m.cells:
            if cb.type == "tetra":
                tets = cb.data.astype(np.int32)
                break
        if tets is None:
            print("ERROR: no tetrahedra found in mesh", file=sys.stderr)
            sys.exit(1)
        _convert_result = (verts, tets)
        _phase = "ready"
    else:
        # Surface mesh: load and display, then let user convert
        mesh = trimesh.load(args.mesh, process=False, force="mesh")
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            print("ERROR: could not load surface mesh", file=sys.stderr)
            sys.exit(1)
        _surf_verts = np.asarray(mesh.vertices, dtype=np.float64)
        _surf_faces = np.asarray(mesh.faces, dtype=np.int32)
        sm = ps.register_surface_mesh(_PS_INPUT, _surf_verts, _surf_faces, smooth_shade=True)
        sm.set_color((0.7, 0.7, 0.7))
        print(f"Loaded {os.path.basename(args.mesh)}: {len(_surf_verts)} verts, {len(_surf_faces)} faces")

    ps.set_user_callback(_callback)
    ps.show()


if __name__ == "__main__":
    main()
