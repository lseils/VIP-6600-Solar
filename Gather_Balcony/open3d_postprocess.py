#!/usr/bin/env python3
"""
open3d_postprocess.py
---------------------
Cleans up COLMAP output mesh/point cloud for CAD/BIM import.

Steps:
  1. Load fused.ply point cloud
  2. Remove statistical outliers
  3. RANSAC plane fitting → isolate building facade
  4. Estimate normals (needed for meshing / BIM tools)
  5. Downsample to manageable size
  6. Poisson reconstruction → export as .ply and .obj

Usage:
  pip install open3d
  python open3d_postprocess.py
  python open3d_postprocess.py --input colmap_workspace/dense/fused.ply --voxel 0.05
  python open3d_postprocess.py --plane_thickness 0.5   # loosen if facade is too sparse
"""

import argparse
import sys
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] open3d not installed. Run: pip install open3d")
    sys.exit(1)

import numpy as np


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    print(f"[1/6] Loading point cloud: {path}")
    pcd = o3d.io.read_point_cloud(path)
    print(f"      {len(pcd.points):,} points loaded")
    return pcd


def remove_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    print("[2/6] Removing statistical outliers...")
    cleaned, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    removed = len(pcd.points) - len(cleaned.points)
    print(f"      Removed {removed:,} outlier points ({removed/len(pcd.points)*100:.1f}%)")
    return cleaned


def extract_facade_plane(pcd: o3d.geometry.PointCloud,
                          plane_thickness: float) -> o3d.geometry.PointCloud:
    """
    Use RANSAC to find the dominant plane (the building facade), then keep
    only points within `plane_thickness` meters of that plane.
    Iterates up to 3 times to find the largest planar surface.
    """
    print(f"[3/6] RANSAC plane fitting (thickness=±{plane_thickness}m)...")

    best_inliers = None
    best_equation = None
    remaining = pcd

    # Try up to 3 planes, keep the one with the most inliers
    for i in range(3):
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=plane_thickness,
            ransac_n=3,
            num_iterations=1000
        )
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_equation = plane_model
            # Remove this plane's inliers and try again on what's left
            remaining = remaining.select_by_index(inliers, invert=True)

    a, b, c, d = best_equation
    print(f"      Dominant plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    facade = pcd.select_by_index(best_inliers)
    noise  = pcd.select_by_index(best_inliers, invert=True)

    kept_pct = len(facade.points) / len(pcd.points) * 100
    print(f"      Kept {len(facade.points):,} facade points ({kept_pct:.1f}%)")
    print(f"      Discarded {len(noise.points):,} non-facade points")

    if len(facade.points) < 500:
        print("      [WARN] Very few facade points — try increasing --plane_thickness")

    return facade


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    print(f"[4/6] Voxel downsampling (voxel_size={voxel_size}m)...")
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"      {len(downsampled.points):,} points after downsampling")
    return downsampled


def estimate_normals(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    print("[5/6] Estimating surface normals...")
    radius = voxel_size * 5
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    # Orient all normals to point in the same direction (toward the camera / street)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def reconstruct_and_export(pcd: o3d.geometry.PointCloud,
                            output_dir: Path,
                            voxel_size: float):
    print("[6/6] Reconstructing mesh & exporting...")

    print("      Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False,
        n_threads=1,  # multithreaded PoissonRecon randomly aborts (open3d 0.19)
    )

    # Trim low-density vertices (removes floating artifacts at mesh boundary)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    # Save cleaned point cloud
    pcd_out = output_dir / "cleaned_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_out), pcd)
    print(f"      Saved: {pcd_out}")

    # Save mesh as PLY
    ply_out = output_dir / "mesh_cleaned.ply"
    o3d.io.write_triangle_mesh(str(ply_out), mesh)
    print(f"      Saved: {ply_out}")

    # Save mesh as OBJ (Revit, AutoCAD, Rhino compatible)
    obj_out = output_dir / "mesh_cleaned.obj"
    o3d.io.write_triangle_mesh(str(obj_out), mesh)
    print(f"      Saved: {obj_out}  ← import this into Revit/AutoCAD")

    print(f"\n  Mesh stats:")
    print(f"    Vertices : {len(mesh.vertices):,}")
    print(f"    Triangles: {len(mesh.triangles):,}")
    is_watertight = mesh.is_watertight()
    print(f"    Watertight: {'Yes' if is_watertight else 'No (normal for facade scans)'}")

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Post-process COLMAP output for CAD/BIM")
    parser.add_argument(
        "--input",
        default="colmap_workspace/dense/fused.ply",
        help="Path to fused.ply from COLMAP stereo fusion"
    )
    parser.add_argument(
        "--output_dir",
        default="colmap_workspace/mesh",
        help="Directory to save cleaned outputs"
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.05,
        help="Voxel size in meters for downsampling (default: 0.05 = 5cm)"
    )
    parser.add_argument(
        "--plane_thickness",
        type=float,
        default=0.3,
        help="Max distance from facade plane to keep points, in meters (default: 0.3). "
             "Increase to 0.5-1.0 if facade is too sparse after extraction."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        print("  Run run_colmap.sh first to generate fused.ply")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Open3D Post-processing for CAD/BIM")
    print("=" * 50)

    pcd = load_point_cloud(str(input_path))
    pcd = remove_outliers(pcd)
    pcd = extract_facade_plane(pcd, args.plane_thickness)
    pcd = voxel_downsample(pcd, args.voxel)
    pcd = estimate_normals(pcd, args.voxel)
    reconstruct_and_export(pcd, output_dir, args.voxel)

    print("\n" + "=" * 50)
    print("  Done! Import mesh_cleaned.obj into your BIM tool.")
    print("=" * 50)


if __name__ == "__main__":
    main()