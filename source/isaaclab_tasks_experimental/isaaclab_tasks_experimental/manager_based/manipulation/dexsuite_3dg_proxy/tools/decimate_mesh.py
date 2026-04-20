#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Decimate a mesh to a target face count using Quadric Edge Collapse (QEM).

Uses trimesh + fast-simplification (VTK-based QEM), which preserves surface
detail far better than vertex clustering.

Usage:
    python tools/decimate_mesh.py input.obj output.obj --faces 2000
    python tools/decimate_mesh.py input.obj output.obj --faces 500
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input mesh file (OBJ, PLY, STL).")
    parser.add_argument("output", help="Output mesh file (OBJ, PLY, STL).")
    parser.add_argument("--faces", type=int, default=2000, help="Target face count (default: 2000).")
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    try:
        import fast_simplification  # noqa: F401 — imported by trimesh internally
    except ImportError:
        raise ImportError("fast-simplification is required: pip install fast-simplification")

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Loading {input_path} ...")
    mesh = trimesh.load(input_path, force="mesh")
    print(f"  Input:  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if len(mesh.faces) <= args.faces:
        print(f"  Already at or below target ({args.faces} faces). Copying as-is.")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    print(f"  Decimating to {args.faces} faces (QEM) ...")
    result = mesh.simplify_quadric_decimation(face_count=args.faces)

    print(f"  Output: {len(result.vertices)} vertices, {len(result.faces)} faces")
    result.export(output_path)
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
