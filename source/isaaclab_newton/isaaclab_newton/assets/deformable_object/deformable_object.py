# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.physics import PhysicsEvent  # still needed for PHYSICS_READY callback

from isaaclab_newton.physics import NewtonManager as SimulationManager


@dataclass
class DeformableRegistryEntry:
    """Entry in the deformable body registry.

    Registered by :class:`DeformableObject` during ``__init__``, consumed by
    ``newton_physics_replicate`` inside the per-world ``begin_world``/``end_world`` loop.
    After replication, ``particle_offsets`` and ``particles_per_body`` are filled in
    so the asset can bind to the correct particle ranges.
    """

    prim_path: str
    vertices: list  # list of wp.vec3
    indices: list  # flat list of ints
    is_tet: bool
    init_pos: tuple[float, float, float]
    init_rot: tuple[float, float, float, float]  # (w, x, y, z)
    # Cloth params
    density: float = 0.02
    tri_ke: float = 1e4
    tri_ka: float = 1e4
    tri_kd: float = 1.5e-6
    edge_ke: float = 5.0
    edge_kd: float = 1e-2
    particle_radius: float = 0.008
    soft_contact_ke: float = 1e4
    soft_contact_kd: float = 1e-2
    # Tet params
    k_mu: float = 1e5
    k_lambda: float = 1e5
    k_damp: float = 0.0
    # Filled by newton_physics_replicate:
    particle_offsets: list[int] = field(default_factory=list)
    particles_per_body: int = 0

from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    scatter_default_pos_index,
    scatter_particles_vec3f_index,
    scatter_zero_vel_index,
    set_kinematic_flags_to_one,
    vec6f,
)

if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)


class DeformableObject(BaseDeformableObject):
    """A deformable object asset class (Newton backend).

    This class manages cloth/deformable bodies in the Newton physics engine. Newton stores all
    particles in flat arrays (``state.particle_q``, ``state.particle_qd``). This class builds
    a per-instance indexing layer on top of those flat arrays, enabling the standard
    :class:`BaseDeformableObject` interface for reading/writing nodal state.

    The cloth mesh is added to the Newton :class:`ModelBuilder` during the ``MODEL_INIT`` phase.
    The mesh data is read from the USD prim at :attr:`cfg.prim_path`, and cloth simulation
    parameters (density, stiffness, etc.) come from :attr:`DeformableObjectCfg`.
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        # super().__init__ triggers the spawner, creating the USD prim.
        # We need the prim to exist so we can read mesh data for the registry.
        super().__init__(cfg)

        # Read mesh from the spawned USD prim and register in the deformable registry.
        # newton_physics_replicate will consume this inside begin_world/end_world for
        # proper per-world particle assignment.
        self._registry_entry = self._register_deformable()

        # Register custom vec6f type for nodal state validation.
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}

    """
    Properties
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self._particles_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        For selected environments, restores default particle positions and zeros velocities
        in both Newton states. Also zeros VBD solver internal buffers for the affected particles.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated.
                Shape is (num_instances,).
        """
        # Resolve env_ids
        if env_mask is not None:
            env_ids_wp = wp.nonzero(env_mask)
        elif env_ids is None:
            env_ids_wp = self._ALL_INDICES
        else:
            env_ids_wp = self._resolve_env_ids(env_ids)

        num_selected = env_ids_wp.shape[0]
        if num_selected == 0:
            return

        # Reset particle positions and velocities in both states
        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is None:
                continue
            if state.particle_q is not None:
                wp.launch(
                    scatter_default_pos_index,
                    dim=(num_selected, self._particles_per_body),
                    inputs=[self._default_nodal_pos_w, env_ids_wp, self._particle_offsets],
                    outputs=[state.particle_q],
                    device=self.device,
                )
            if state.particle_qd is not None:
                wp.launch(
                    scatter_zero_vel_index,
                    dim=(num_selected, self._particles_per_body),
                    inputs=[env_ids_wp, self._particle_offsets, self._particles_per_body],
                    outputs=[state.particle_qd],
                    device=self.device,
                )

        # Zero VBD solver internal buffers for affected particle ranges
        solver = SimulationManager._solver
        if solver is not None:
            for attr in ("particle_q_prev", "inertia", "pos_prev_collision_detection", "particle_displacements"):
                buf = getattr(solver, attr, None)
                if buf is not None and buf.dtype == wp.vec3f:
                    wp.launch(
                        scatter_zero_vel_index,
                        dim=(num_selected, self._particles_per_body),
                        inputs=[env_ids_wp, self._particle_offsets, self._particles_per_body],
                        outputs=[buf],
                        device=self.device,
                    )

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)
        # Update USD visualization if enabled
        self._update_cloth_vis()

    """
    Operations - Write to simulation.
    """

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(nodal_pos, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_pos")
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        # Scatter into both Newton states
        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is not None and state.particle_q is not None:
                wp.launch(
                    scatter_particles_vec3f_index,
                    dim=(env_ids.shape[0], self._particles_per_body),
                    inputs=[nodal_pos, env_ids, self._particle_offsets, full_data],
                    outputs=[state.particle_q],
                    device=self.device,
                )

        # Invalidate data caches
        self._data._nodal_pos_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3)
                or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self._particles_per_body), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(nodal_vel, (env_ids.shape[0], self._particles_per_body), wp.vec3f, "nodal_vel")
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        # Scatter into both Newton states
        for state in (SimulationManager._state_0, SimulationManager._state_1):
            if state is not None and state.particle_qd is not None:
                wp.launch(
                    scatter_particles_vec3f_index,
                    dim=(env_ids.shape[0], self._particles_per_body),
                    inputs=[nodal_vel, env_ids, self._particle_offsets, full_data],
                    outputs=[state.particle_qd],
                    device=self.device,
                )

        # Invalidate data caches
        self._data._nodal_vel_w.timestamp = -1.0
        self._data._nodal_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies.

        Newton has no native kinematic target API. Instead:
        - Kinematic (flag=0.0): set ``particle_inv_mass`` to 0, write target pos, zero vel
        - Free (flag=1.0): restore original ``particle_inv_mass``

        Args:
            targets: The kinematic targets comprising of nodal positions and flags [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 4)
                or (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(targets, (self.num_instances, self._particles_per_body), wp.vec4f, "targets")
        else:
            self.assert_shape_and_dtype(targets, (env_ids.shape[0], self._particles_per_body), wp.vec4f, "targets")
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)

        # Store kinematic targets in our data buffer
        # Note: actual enforcement via particle_inv_mass is deferred to write_data_to_sim
        # For now, we just store the targets for data access
        if self._data.nodal_kinematic_target is not None:
            # Write targets into our buffer (simple copy for selected envs)
            targets_torch = wp.to_torch(targets)
            buffer_torch = wp.to_torch(self._data.nodal_kinematic_target)
            if full_data:
                for idx in range(env_ids.shape[0]):
                    env_id = int(wp.to_torch(env_ids)[idx].item())
                    buffer_torch[env_id] = targets_torch[env_id]
            else:
                for idx in range(env_ids.shape[0]):
                    env_id = int(wp.to_torch(env_ids)[idx].item())
                    buffer_torch[env_id] = targets_torch[idx]

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids):
        """Resolve environment indices to a warp int32 array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _register_deformable(self) -> DeformableRegistryEntry:
        """Read mesh from the spawned USD prim and register in NewtonManager's deformable registry.

        Called during ``__init__`` after the spawner has created the prim.
        The registry entry is consumed by ``newton_physics_replicate`` inside
        ``begin_world``/``end_world`` for proper per-world particle assignment.

        Returns:
            The registry entry (also stored on NewtonManager._deformable_registry).
        """
        # Find the spawned mesh prim
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")

        # Find the mesh descendant — either UsdGeom.TetMesh or UsdGeom.Mesh
        has_tet_type = hasattr(UsdGeom, "TetMesh")
        mesh_prim = None
        if has_tet_type and template_prim.IsA(UsdGeom.TetMesh):
            mesh_prim = template_prim
        elif template_prim.IsA(UsdGeom.Mesh):
            mesh_prim = template_prim
        else:
            for desc in Usd.PrimRange(template_prim):
                if desc == template_prim:
                    continue
                if has_tet_type and desc.IsA(UsdGeom.TetMesh):
                    mesh_prim = desc
                    break
                if desc.IsA(UsdGeom.Mesh):
                    mesh_prim = desc
                    break

        if mesh_prim is None:
            raise RuntimeError(
                f"No UsdGeom.Mesh or UsdGeom.TetMesh found at or under '{self.cfg.prim_path}'. "
                "Please ensure the spawn config creates a mesh prim (e.g. MeshFromFileCfg or TetMeshCuboidCfg)."
            )

        # Read mesh data
        is_tet = mesh_prim.IsA(UsdGeom.TetMesh) if has_tet_type else False

        if is_tet:
            tet_mesh = UsdGeom.TetMesh(mesh_prim)
            pts = np.array(tet_mesh.GetPointsAttr().Get(), dtype=np.float32)
            vertices = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts]
            raw_tet_indices = tet_mesh.GetTetVertexIndicesAttr().Get()
            indices = []
            for vec4i in raw_tet_indices:
                indices.extend([int(vec4i[0]), int(vec4i[1]), int(vec4i[2]), int(vec4i[3])])
            logger.info(
                f"Registered UsdGeom.TetMesh: {len(pts)} vertices, {len(indices) // 4} tetrahedra."
            )
        else:
            usd_mesh = UsdGeom.Mesh(mesh_prim)
            pts = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
            vertices = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts]
            indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get())
            logger.info(f"Registered UsdGeom.Mesh: {len(pts)} vertices.")

        init_pos = self.cfg.init_state.pos if hasattr(self.cfg.init_state, "pos") else (0.0, 0.0, 0.0)
        init_rot = self.cfg.init_state.rot if hasattr(self.cfg.init_state, "rot") else (1.0, 0.0, 0.0, 0.0)

        entry = DeformableRegistryEntry(
            prim_path=self.cfg.prim_path,
            vertices=vertices,
            indices=indices,
            is_tet=is_tet,
            init_pos=init_pos,
            init_rot=init_rot,
            density=self.cfg.density,
            tri_ke=self.cfg.tri_ke,
            tri_ka=self.cfg.tri_ka,
            tri_kd=self.cfg.tri_kd,
            edge_ke=self.cfg.edge_ke,
            edge_kd=self.cfg.edge_kd,
            particle_radius=self.cfg.particle_radius,
            soft_contact_ke=self.cfg.soft_contact_ke,
            soft_contact_kd=self.cfg.soft_contact_kd,
            k_mu=self.cfg.k_mu,
            k_lambda=self.cfg.k_lambda,
            k_damp=self.cfg.k_damp,
        )
        SimulationManager._deformable_registry.append(entry)
        return entry

    def _initialize_impl(self):
        """Initialize physics handles and buffers after the Newton model is ready."""
        # Read particle offsets from the registry entry (filled by newton_physics_replicate
        # or by the MODEL_INIT fallback)
        entry = self._registry_entry
        self._num_instances = len(entry.particle_offsets)
        self._particles_per_body = entry.particles_per_body
        self._recorded_particle_offsets = entry.particle_offsets

        if self._num_instances == 0:
            raise RuntimeError(
                f"No deformable body instances found for '{self.cfg.prim_path}'. "
                "Ensure newton_physics_replicate or MODEL_INIT processed the registry."
            )

        logger.info(f"Newton deformable object initialized at: {self.cfg.prim_path}")
        logger.info(f"Number of instances: {self._num_instances}")
        logger.info(f"Particles per body: {self._particles_per_body}")

        # Build particle offset array on device
        self._particle_offsets = wp.array(self._recorded_particle_offsets, dtype=wp.int32, device=self.device)

        # Create data container
        self._data = DeformableObjectData(
            particle_offsets=self._particle_offsets,
            particles_per_body=self._particles_per_body,
            num_instances=self._num_instances,
            device=self.device,
        )

        # Bind simulation state arrays
        state = SimulationManager._state_0
        if state is not None:
            self._data.bind_simulation_state(state.particle_q, state.particle_qd)

        # Create buffers
        self._create_buffers()

        # Update data once
        self.update(0.0)

        # Register rebind callback for full resets
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._rebind_state(),
            PhysicsEvent.PHYSICS_READY,
            name=f"deformable_object_rebind_{self.cfg.prim_path}",
        )

    def _rebind_state(self) -> None:
        """Rebind state arrays after a full simulation reset."""
        state = SimulationManager._state_0
        if state is not None and hasattr(self, "_data"):
            self._data.bind_simulation_state(state.particle_q, state.particle_qd)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # Constants
        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self.device)

        # Snapshot default positions from current state (after finalize + FK)
        state = SimulationManager._state_0
        if state is not None and state.particle_q is not None:
            # Gather initial positions per instance
            from .kernels import gather_particles_vec3f

            self._default_nodal_pos_w = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=wp.vec3f, device=self.device
            )
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[state.particle_q, self._particle_offsets, self._particles_per_body],
                outputs=[self._default_nodal_pos_w],
                device=self.device,
            )

            # Compute default nodal state as vec6f (positions + zero velocities)
            nodal_velocities = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=wp.vec3f, device=self.device
            )
            self._data.default_nodal_state_w = wp.zeros(
                (self._num_instances, self._particles_per_body), dtype=vec6f, device=self.device
            )
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self._default_nodal_pos_w, nodal_velocities],
                outputs=[self._data.default_nodal_state_w],
                device=self.device,
            )
        else:
            self._default_nodal_pos_w = None

        # Kinematic targets — allocate and initialize with free flags
        self._data.nodal_kinematic_target = wp.zeros(
            (self._num_instances, self._particles_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self._num_instances * self._particles_per_body,),
            inputs=[self._data.nodal_kinematic_target.reshape((self._num_instances * self._particles_per_body,))],
            device=self.device,
        )

        # Set up the model parameters
        model = SimulationManager._model
        if model is not None:
            if hasattr(model, "edge_rest_angle"):
                model.edge_rest_angle.zero_()
            model.soft_contact_ke = self.cfg.soft_contact_ke
            model.soft_contact_kd = self.cfg.soft_contact_kd

        # Bind spawned mesh prims for Kit viewport updates (Kit only)
        if not SimulationManager._clone_physics_only:
            self._bind_cloth_vis_prims()

    """
    USD mesh visualization (Kit only).
    """

    def _bind_cloth_vis_prims(self) -> None:
        """Bind spawned mesh prims for dynamic point updates in Kit viewport.

        Finds the spawned ``UsdGeom.Mesh`` or ``UsdGeom.TetMesh`` prim for each instance
        (typically at ``{prim_path}/geometry/mesh``), clears parent Xform transforms
        (Newton writes world-space positions), writes initial points, and stores
        references for per-step updates via :meth:`_update_cloth_vis`.
        """
        from pxr import Gf, Vt

        from isaaclab.sim.utils.stage import get_current_stage

        state = SimulationManager._state_0
        if state is None or state.particle_q is None:
            return

        stage = get_current_stage()
        has_tet_type = hasattr(UsdGeom, "TetMesh")
        self._vis_prims = []

        for inst_idx in range(self._num_instances):
            base_path = self.cfg.prim_path.replace("env_.*", f"env_{inst_idx}").replace("*", str(inst_idx))
            offset = self._recorded_particle_offsets[inst_idx]

            base_prim = stage.GetPrimAtPath(base_path)
            if not base_prim.IsValid():
                continue

            # Find the spawned geometry prim — either TetMesh or Mesh
            # Both have GetPointsAttr() for dynamic point updates
            geom_prim = None
            for candidate in [base_prim] + list(Usd.PrimRange(base_prim)):
                if candidate == base_prim and not (candidate.IsA(UsdGeom.Mesh) or (has_tet_type and candidate.IsA(UsdGeom.TetMesh))):
                    continue
                if has_tet_type and candidate.IsA(UsdGeom.TetMesh):
                    geom_prim = UsdGeom.TetMesh(candidate)
                    break
                if candidate.IsA(UsdGeom.Mesh):
                    geom_prim = UsdGeom.Mesh(candidate)
                    break

            if geom_prim is None:
                logger.warning(f"No UsdGeom.Mesh or TetMesh found under '{base_path}' — skipping Kit visualization for env {inst_idx}.")
                continue

            # Clear Xform transforms on all ancestors under base_prim — Newton's
            # particle positions are already in world-space meters.
            for desc in Usd.PrimRange(base_prim):
                if desc.IsA(UsdGeom.Xformable):
                    UsdGeom.Xformable(desc).ClearXformOpOrder()

            # For TetMesh prims, Kit doesn't render them natively in this version.
            # Create a companion UsdGeom.Mesh with the surface faces for rendering.
            if has_tet_type and geom_prim.GetPrim().IsA(UsdGeom.TetMesh):
                from pxr import Sdf

                tet_mesh = UsdGeom.TetMesh(geom_prim.GetPrim())
                surface_indices = tet_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
                if surface_indices is not None and len(surface_indices) > 0:
                    vis_mesh_path = geom_prim.GetPath().pathString + "_vis"
                    vis_mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(vis_mesh_path))
                    # Convert Vec3iArray to flat face vertex indices
                    face_vertex_indices = []
                    for vec3i in surface_indices:
                        face_vertex_indices.extend([int(vec3i[0]), int(vec3i[1]), int(vec3i[2])])
                    vis_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
                    vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
                    vis_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

                    # Copy material binding from the TetMesh's parent geometry prim
                    from pxr import UsdShade

                    geom_path = geom_prim.GetPath().GetParentPath()
                    mat_path = str(geom_path) + "/" + self.cfg.spawn.visual_material_path if hasattr(self.cfg.spawn, "visual_material_path") else None
                    if mat_path is not None:
                        mat_prim = stage.GetPrimAtPath(mat_path)
                        if mat_prim.IsValid():
                            UsdShade.MaterialBindingAPI.Apply(vis_mesh.GetPrim())
                            UsdShade.MaterialBindingAPI(vis_mesh.GetPrim()).Bind(
                                UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants
                            )

                    geom_prim = vis_mesh

            # Write initial vertex positions from Newton particle state
            pts_np = state.particle_q.numpy()[offset : offset + self._particles_per_body]
            points = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_np])
            geom_prim.GetPointsAttr().Set(points)

            self._vis_prims.append((geom_prim, offset))

    def _update_cloth_vis(self) -> None:
        """Write current Newton particle positions into Kit cloth mesh prims."""
        if not hasattr(self, "_vis_prims") or not self._vis_prims:
            return

        state = SimulationManager._state_0
        if state is None or state.particle_q is None:
            return

        from pxr import Gf, Vt

        pts_np = state.particle_q.numpy()
        for mesh, offset in self._vis_prims:
            inst_pts = pts_np[offset : offset + self._particles_per_body]
            points = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in inst_pts])
            mesh.GetPointsAttr().Set(points)

    """
    Internal simulation callbacks.
    """

    def _clear_callbacks(self) -> None:
        """Clears all registered callbacks."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
