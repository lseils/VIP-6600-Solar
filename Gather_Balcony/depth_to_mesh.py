#!/usr/bin/env python3
"""
depth_to_mesh.py
-----------------
Generates a 2.5D relief mesh from a single image using Depth Anything V2.

Each pixel becomes a 3D vertex (x, y, depth), and a regular grid of
triangles connects neighboring pixels. Triangles that span a large depth
discontinuity (e.g. the edge of a balcony against the wall behind it) are
cut, so you don't get long "skirt" triangles stretching between surfaces.

Depth is RELATIVE (0-1, higher = closer to camera) — useful for comparing
protrusion between features (balconies vs wall), not for absolute meters.

Output:
  <output_dir>/<image_name>_mesh.ply   ← open in Blender/MeshLab
  <output_dir>/<image_name>_mesh.obj   ← also exported for compatibility
  <output_dir>/<image_name>_depth.png  ← depth map visualization

Usage:
  python depth_to_mesh.py perspective_crops/pano_001_center.jpg
  python depth_to_mesh.py perspective_crops/pano_001_center.jpg --stride 2
  python depth_to_mesh.py perspective_crops/pano_001_center.jpg --depth_scale 3.0 --discontinuity 0.05
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


# =============================================================================
# DEPTH ESTIMATION
# =============================================================================
def estimate_depth(depth_pipe, image_path: str) -> np.ndarray:
    """Relative depth, normalized 0-1 (higher = closer to camera)."""
    img = Image.open(image_path).convert("RGB")
    result = depth_pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin > 0:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


# =============================================================================
# MESH GENERATION
# =============================================================================
def build_mesh(depth: np.ndarray,
              rgb: np.ndarray,
              stride: int,
              depth_scale: float,
              discontinuity: float):
    """
    Build a triangulated relief mesh from a depth map.

    - Grid of vertices at (x, y, z) where z = depth * depth_scale
      (z is "height" out of the wall — higher relative depth = closer = more
      protrusion. We treat depth directly as the z-axis displacement.)
    - x, y are pixel coordinates (subsampled by stride), normalized to a
      reasonable scale.
    - Triangles are skipped if any edge's depth difference exceeds
      `discontinuity` (in normalized 0-1 depth units) — this prevents long
      "skirt" triangles connecting a balcony edge to the wall behind it.

    Returns: vertices (Nx3), colors (Nx3 uint8), faces (Mx3 indices)
    """
    h, w = depth.shape

    # Subsampled grid coordinates
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_h, grid_w = len(ys), len(xs)

    # Build vertex positions
    # x, y normalized to roughly [0, grid_w] / [0, grid_h] in "pixel-like" units
    # z = depth * depth_scale (protrusion toward camera)
    xv, yv = np.meshgrid(xs, ys)
    dv = depth[yv, xv]

    # Flip y so the mesh isn't upside down when viewed in standard 3D viewers
    # (image row 0 = top, but we want top = +Y)
    vertices = np.stack([
        xv.astype(np.float32),
        (h - yv).astype(np.float32),
        (dv * depth_scale).astype(np.float32)
    ], axis=-1).reshape(-1, 3)

    # Vertex colors from RGB
    rgb_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    colors = rgb_small[yv, xv][:, ::-1].reshape(-1, 3)  # BGR -> RGB

    # Build faces (two triangles per grid cell), skipping discontinuous edges
    faces = []
    for r in range(grid_h - 1):
        for c in range(grid_w - 1):
            i00 = r * grid_w + c
            i01 = r * grid_w + (c + 1)
            i10 = (r + 1) * grid_w + c
            i11 = (r + 1) * grid_w + (c + 1)

            d00 = dv[r, c]
            d01 = dv[r, c + 1]
            d10 = dv[r + 1, c]
            d11 = dv[r + 1, c + 1]

            # Triangle 1: (i00, i10, i11)
            if (abs(d00 - d10) < discontinuity and
                abs(d10 - d11) < discontinuity and
                abs(d00 - d11) < discontinuity):
                faces.append((i00, i10, i11))

            # Triangle 2: (i00, i11, i01)
            if (abs(d00 - d11) < discontinuity and
                abs(d11 - d01) < discontinuity and
                abs(d00 - d01) < discontinuity):
                faces.append((i00, i11, i01))

    faces = np.array(faces, dtype=np.int64)
    return vertices, colors, faces


# =============================================================================
# PLY WRITER (with faces)
# =============================================================================
def write_ply(path, vertices, colors, faces):
    n_v = len(vertices)
    n_f = len(faces)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_v}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_f}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output_dir", default="depth_meshes")
    parser.add_argument("--stride", type=int, default=4,
                        help="Pixel subsample stride for mesh grid (default 4). "
                             "Lower = more detail but bigger mesh.")
    parser.add_argument("--depth_scale", type=float, default=300.0,
                        help="Multiplier for depth->Z displacement (default 300). "
                             "Tune so protrusions look proportional to the facade width "
                             "(which is in pixel units, typically ~2000).")
    parser.add_argument("--discontinuity", type=float, default=0.03,
                        help="Max relative-depth difference (0-1) allowed across a "
                             "triangle edge before it's cut (default 0.03). "
                             "Lower = more cuts (cleaner but more holes). "
                             "Higher = fewer cuts (more skirts).")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)
    image_name = Path(args.image).stem

    print(f"Loading Depth Anything V2...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=device,
    )

    print(f"\nProcessing {args.image}...")
    depth = estimate_depth(depth_pipe, args.image)
    rgb = cv2.imread(args.image)
    h, w = depth.shape
    print(f"  Depth map: {w}x{h}")

    # Save depth visualization
    depth_vis = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_out = os.path.join(args.output_dir, f"{image_name}_depth.png")
    cv2.imwrite(depth_out, depth_color)
    print(f"  Saved depth visualization: {depth_out}")

    print(f"\nBuilding mesh (stride={args.stride}, depth_scale={args.depth_scale}, "
          f"discontinuity={args.discontinuity})...")
    vertices, colors, faces = build_mesh(
        depth, rgb, args.stride, args.depth_scale, args.discontinuity
    )
    print(f"  {len(vertices):,} vertices, {len(faces):,} faces")

    ply_out = os.path.join(args.output_dir, f"{image_name}_mesh.ply")
    write_ply(ply_out, vertices, colors, faces)
    print(f"  Saved: {ply_out}")

    obj_out = os.path.join(args.output_dir, f"{image_name}_mesh.obj")
    write_obj(obj_out, vertices, faces)
    print(f"  Saved: {obj_out}")

    print(f"\n{'='*52}")
    print(f"  Done! Open {ply_out} in Blender/MeshLab to inspect.")
    print(f"\n  Tuning tips:")
    print(f"    Mesh looks flat?       increase --depth_scale (try 600, 1000)")
    print(f"    Too many skirts/spikes? decrease --discontinuity (try 0.01)")
    print(f"    Too many holes?         increase --discontinuity (try 0.05)")
    print(f"    Mesh too dense/slow?    increase --stride (try 6, 8)")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
    