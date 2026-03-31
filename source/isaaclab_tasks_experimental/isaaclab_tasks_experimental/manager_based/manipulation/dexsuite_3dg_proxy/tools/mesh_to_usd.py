#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert a mesh file (OBJ/STL/FBX) to an instanceable USD asset with physics properties.

Must be run inside Isaac Sim via isaaclab.sh:
    ./isaaclab.sh -p tools/mesh_to_usd.py input.obj output.usd

The resulting USD can be spawned in the task via:
    sim_utils.UsdFileCfg(usd_path="output.usd", ...)

Usage:
    ./isaaclab.sh -p tools/mesh_to_usd.py mesh.obj asset.usd
    ./isaaclab.sh -p tools/mesh_to_usd.py mesh.obj asset.usd --mass 0.2 --collision-approx convexDecomposition
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("mesh", help="Input mesh file (OBJ, STL, FBX).")
parser.add_argument("usd", help="Output USD file path.")
parser.add_argument("--mass", type=float, default=0.2, help="Object mass in kg (default: 0.2).")
parser.add_argument(
    "--collision-approx",
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "none"],
    help="Collision mesh approximation (default: convexDecomposition).",
)
parser.add_argument("--static-friction", type=float, default=0.5)
parser.add_argument("--dynamic-friction", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- imports after app launch ---
import os

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def main():
    mesh_path = os.path.abspath(args.mesh)
    usd_path  = os.path.abspath(args.usd)

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    os.makedirs(os.path.dirname(usd_path) or ".", exist_ok=True)

    _approx_cfg_map = {
        "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
        "convexHull":          schemas_cfg.ConvexHullPropertiesCfg,
        "boundingCube":        schemas_cfg.BoundingCubePropertiesCfg,
        "boundingSphere":      schemas_cfg.BoundingSpherePropertiesCfg,
        "none":                None,
    }
    collision_props = schemas_cfg.CollisionPropertiesCfg()
    approx_cls = _approx_cfg_map[args.collision_approx]
    mesh_collision_props = approx_cls() if approx_cls is not None else None

    cfg = MeshConverterCfg(
        asset_path=mesh_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=args.mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=collision_props,
        mesh_collision_props=mesh_collision_props,
    )

    print(f"Converting: {mesh_path}")
    print(f"       → : {usd_path}")
    print(f"  mass={args.mass} kg, collision={args.collision_approx}")

    converter = MeshConverter(cfg)
    print(f"Done. USD saved: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
