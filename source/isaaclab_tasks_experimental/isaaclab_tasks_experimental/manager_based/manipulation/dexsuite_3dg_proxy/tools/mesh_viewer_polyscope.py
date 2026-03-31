"""
Polyscope-based viewer for meshes and point clouds.

Usage:
    python mesh_viewer_polyscope.py path/to/mesh.ply [options]
    python mesh_viewer_polyscope.py path/to/cloud.ply --point-cloud
    python mesh_viewer_polyscope.py a.ply b.obj c.ply   # multiple files

Options:
    --point-cloud       Force loading as point cloud (even if faces present)
    --no-normals        Do not color by vertex normals
    --flip-triangles    Flip triangle winding order
    --background dark|light|white|black
"""

import argparse
import os
import sys

import numpy as np

try:
    import polyscope as ps
except ImportError:
    print("polyscope not found. Install with: pip install polyscope")
    sys.exit(1)

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _normals_to_colors(normals: np.ndarray) -> np.ndarray:
    """Map normals in [-1,1]^3 to colors in [0,1]^3."""
    return (normals + 1.0) / 2.0


def load_mesh(filepath: str, flip_triangles: bool = False):
    """Return (vertices, faces, vertex_normals|None). faces=None → point cloud."""
    ext = os.path.splitext(filepath)[1].lower()

    if HAS_TRIMESH and ext in (".obj", ".ply", ".stl", ".off", ".glb", ".gltf"):
        loaded = trimesh.load(filepath, process=False, force="mesh")
        if isinstance(loaded, trimesh.PointCloud):
            return np.asarray(loaded.vertices), None, None
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int32)
        if flip_triangles:
            faces = faces[:, [0, 2, 1]]
        normals = np.asarray(loaded.vertex_normals, dtype=np.float64) if loaded.vertex_normals is not None else None
        return verts, faces, normals

    if HAS_O3D:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.triangles, dtype=np.int32)
            if flip_triangles:
                faces = faces[:, [0, 2, 1]]
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            return verts, faces, normals
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points, dtype=np.float64)
        nrm = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
        return pts, None, nrm

    raise RuntimeError(f"Cannot load {filepath}: install trimesh or open3d")


def load_point_cloud(filepath: str):
    """Return (points, normals|None)."""
    if HAS_O3D:
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points, dtype=np.float64)
        nrm = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
        return pts, nrm
    if HAS_TRIMESH:
        loaded = trimesh.load(filepath, process=False)
        if hasattr(loaded, "vertices"):
            return np.asarray(loaded.vertices, dtype=np.float64), None
    raise RuntimeError(f"Cannot load {filepath}: install open3d or trimesh")


def load_tet_mesh(filepath: str):
    """Load a tet mesh via meshio. Returns (vertices, tets)."""
    try:
        import meshio
    except ImportError:
        raise RuntimeError("meshio not found: pip install meshio")
    m = meshio.read(filepath)
    verts = np.asarray(m.points, dtype=np.float64)
    tets = None
    for cell_block in m.cells:
        if cell_block.type == "tetra":
            tets = np.asarray(cell_block.data, dtype=np.int32)
            break
    if tets is None:
        raise RuntimeError(f"No tetrahedra found in {filepath}")
    return verts, tets


def register_file(filepath: str, force_point_cloud: bool, flip_triangles: bool, color_normals: bool, wireframe: bool, idx: int):
    name = f"{idx:02d}_{os.path.basename(filepath)}"

    # tet mesh
    if filepath.endswith(".msh") or filepath.endswith(".msh2") or filepath.endswith(".msh4"):
        verts, tets = load_tet_mesh(filepath)
        vm = ps.register_volume_mesh(name, verts, tets=tets)
        if wireframe:
            vm.set_edge_width(1.0)
        print(f"  [{name}] tet mesh: {len(verts)} verts, {len(tets)} tets")
        return

    if force_point_cloud:
        pts, nrm = load_point_cloud(filepath)
        pc = ps.register_point_cloud(name, pts, radius=0.002)
        if color_normals and nrm is not None and len(nrm) == len(pts):
            pc.add_color_quantity("normals", _normals_to_colors(nrm), enabled=True)
        print(f"  [{name}] point cloud: {len(pts)} pts")
        return

    verts, faces, normals = load_mesh(filepath, flip_triangles=flip_triangles)

    if faces is None or len(faces) == 0:
        # treat as point cloud
        pc = ps.register_point_cloud(name, verts, radius=0.002)
        if color_normals and normals is not None and len(normals) == len(verts):
            pc.add_color_quantity("normals", _normals_to_colors(normals), enabled=True)
        print(f"  [{name}] point cloud: {len(verts)} pts")
    else:
        sm = ps.register_surface_mesh(name, verts, faces, smooth_shade=True)
        if color_normals and normals is not None and len(normals) == len(verts):
            sm.add_color_quantity("normals", _normals_to_colors(normals), enabled=True)
        if wireframe:
            sm.set_edge_width(1.0)
        print(f"  [{name}] surface mesh: {len(verts)} verts, {len(faces)} faces")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Polyscope viewer for meshes and point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="+", help="One or more mesh/point-cloud files to visualize.")
    parser.add_argument("--point-cloud", action="store_true", help="Force loading as point cloud.")
    parser.add_argument("--no-normals", action="store_true", help="Disable normal coloring.")
    parser.add_argument(
        "--flip-triangles", action="store_true", help="Flip triangle winding (fix inverted normals)."
    )
    parser.add_argument(
        "--wireframe", action="store_true", help="Show mesh edges (wireframe overlay)."
    )
    parser.add_argument(
        "--background",
        choices=["dark", "light", "white", "black"],
        default="dark",
        help="Polyscope background color (default: dark).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ps.init()

    bg_map = {
        "dark":  (0.15, 0.15, 0.15),
        "light": (0.85, 0.85, 0.85),
        "white": (1.0,  1.0,  1.0),
        "black": (0.0,  0.0,  0.0),
    }
    ps.set_background_color(bg_map[args.background])

    for idx, path in enumerate(args.paths):
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue
        print(f"Loading: {path}")
        register_file(
            path,
            force_point_cloud=args.point_cloud,
            flip_triangles=args.flip_triangles,
            color_normals=not args.no_normals,
            wireframe=args.wireframe,
            idx=idx,
        )

    ps.show()


if __name__ == "__main__":
    main()
