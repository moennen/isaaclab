#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tetrahedralize a surface mesh (OBJ/PLY/STL) using fTetWild (via wildmeshing).

The output .msh (Gmsh format) can be loaded by Newton's VBD/XPBD solver:

    import meshio
    mesh = meshio.read("object.msh")
    nodes = mesh.points            # (V, 3)  float64
    tets  = mesh.cells_dict["tetra"]  # (T, 4)  int

Usage:
    python tools/mesh_to_tet.py input.obj output.msh
    python tools/mesh_to_tet.py input.obj output.msh --stop-quality 8 --target-vertices 500
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input surface mesh (OBJ, PLY, STL).")
    parser.add_argument("output", help="Output tet mesh (.msh, Gmsh format).")
    parser.add_argument(
        "--stop-quality",
        type=float,
        default=10.0,
        metavar="Q",
        help="fTetWild stop quality: lower = higher quality but slower (default: 10, range 5–20).",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=0.05,
        metavar="R",
        help="Relative edge length (fraction of bbox diagonal). Smaller = finer mesh = more vertices "
             "(default: 0.05). Try 0.1 for coarse ~200 nodes, 0.03 for fine ~2000 nodes.",
    )
    args = parser.parse_args()

    try:
        import wildmeshing
    except ImportError:
        print("ERROR: wildmeshing is required: pip install wildmeshing", file=sys.stderr)
        sys.exit(1)

    try:
        import meshio
    except ImportError:
        print("ERROR: meshio is required: pip install meshio", file=sys.stderr)
        sys.exit(1)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        print(f"ERROR: Input mesh not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    kwargs = {"stop_quality": args.stop_quality, "edge_length_r": args.edge_length}

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Quality: stop_quality={args.stop_quality}, edge_length_r={args.edge_length}")
    print("Tetrahedralizing ...")

    tetra = wildmeshing.Tetrahedralizer(**kwargs)
    tetra.load_mesh(input_path)
    tetra.tetrahedralize()
    tetra.save(output_path)

    # Report stats
    mesh = meshio.read(output_path)
    n_verts = len(mesh.points)
    n_tets = len(mesh.cells_dict.get("tetra", []))
    print(f"Done.   {n_verts} nodes, {n_tets} tetrahedra → {output_path}")
    print()
    print("Load in Newton VBD/XPBD:")
    print("    import meshio")
    print(f'    mesh = meshio.read("{output_path}")')
    print('    nodes = mesh.points                  # (V, 3)')
    print('    tets  = mesh.cells_dict["tetra"]     # (T, 4)')


if __name__ == "__main__":
    main()
