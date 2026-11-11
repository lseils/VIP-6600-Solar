#!/usr/bin/env python3
"""
depth_to_pointcloud.py
------------------------
Builds a dense, metrically-consistent point cloud by:
  1. Parsing COLMAP's sparse model (cameras.txt, images.txt, points3D.txt)
  2. Running Depth Anything V2 (relative depth) on each registered image
  3. Using COLMAP's sparse 3D points to calibrate each image's relative
     depth to COLMAP's real-world scale (linear fit: true = a*rel + b)
  4. Back-projecting every pixel's calibrated depth into 3D using the
     camera's intrinsics + pose
  5. Combining all images into one point cloud, colored by RGB

Output:
  colmap_workspace/dense_depth/dense_from_depth.ply

Usage:
  python depth_to_pointcloud.py
  python depth_to_pointcloud.py --stride 4   # subsample pixels (faster, smaller)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# =============================================================================
# CONFIG
# =============================================================================
SPARSE_DIR   = "colmap_workspace/sparse/0"
IMAGE_DIR    = "perspective_crops"
OUTPUT_DIR   = "colmap_workspace/dense_depth"
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Large-hf"
STRIDE       = 2   # subsample factor for back-projection (1 = every pixel)


# =============================================================================
# COLMAP MODEL PARSING
# =============================================================================
def qvec_to_rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ])


def read_cameras(path):
    """Returns dict: camera_id -> dict(model, width, height, params)"""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width  = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "model": model, "width": width, "height": height, "params": params
            }
    return cameras


def read_images(path):
    """
    Returns dict: image_name -> dict(
        image_id, camera_id, qvec, tvec, points2d (Nx2 array), point3d_ids (N array)
    )
    images.txt alternates: line1 = pose info, line2 = 2D points list
    """
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id  = int(parts[0])
        qvec      = np.array(list(map(float, parts[1:5])))   # qw qx qy qz
        tvec      = np.array(list(map(float, parts[5:8])))
        camera_id = int(parts[8])
        name      = parts[9]

        # Next line: 2D points (X Y POINT3D_ID triplets)
        pts_line = lines[i + 1].split()
        pts2d = []
        pt3d_ids = []
        for j in range(0, len(pts_line), 3):
            x = float(pts_line[j])
            y = float(pts_line[j + 1])
            pid = int(pts_line[j + 2])
            pts2d.append((x, y))
            pt3d_ids.append(pid)

        images[name] = {
            "image_id":   image_id,
            "camera_id":  camera_id,
            "qvec":       qvec,
            "tvec":       tvec,
            "points2d":   np.array(pts2d),
            "point3d_ids": np.array(pt3d_ids),
        }
        i += 2

    return images


def read_points3d(path):
    """Returns dict: point3d_id -> (X, Y, Z)"""
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])))
            points[pid] = xyz
    return points


# =============================================================================
# CAMERA INTRINSICS
# =============================================================================
def get_intrinsics(camera: dict):
    """
    Returns fx, fy, cx, cy from a COLMAP camera dict.
    Handles PINHOLE (fx, fy, cx, cy) and SIMPLE_PINHOLE/RADIAL (f, cx, cy, ...).
    """
    model  = camera["model"]
    params = camera["params"]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        # Fallback: assume first 4 params are fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
    return fx, fy, cx, cy


# =============================================================================
# DEPTH ESTIMATION
# =============================================================================
def estimate_depth(depth_pipe, image_path: str) -> np.ndarray:
    """Relative inverse depth (disparity, higher = closer), up to scale and shift.

    Uses the raw float prediction; result["depth"] is an 8-bit visualization.
    """
    img = Image.open(image_path).convert("RGB")
    result = depth_pipe(img)
    return result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)


# =============================================================================
# SCALE CALIBRATION
# =============================================================================
def calibrate_scale(depth: np.ndarray,
                    image_data: dict,
                    points3d: dict,
                    fx, fy, cx, cy,
                    R: np.ndarray, t: np.ndarray,
                    img_w: int, img_h: int,
                    depth_w: int, depth_h: int):
    """
    For each 2D keypoint with a valid 3D point, compute:
      - relative depth at that pixel (from Depth Anything)
      - true depth (distance along camera Z-axis to the 3D point)
    Depth Anything predicts affine-invariant inverse depth, so fit
    1 / true_depth = a * relative_depth + b via least squares.

    COLMAP convention: X_cam = R @ X_world + t
    The camera-space Z coordinate is the "depth" along the optical axis.
    """
    rel_depths = []
    true_depths = []

    for (px, py), pid in zip(image_data["points2d"], image_data["point3d_ids"]):
        if pid == -1 or pid not in points3d:
            continue

        xyz_world = points3d[pid]
        xyz_cam = R @ xyz_world + t
        true_z = xyz_cam[2]

        if true_z <= 0:
            continue  # behind camera, skip

        # Map pixel coords to depth map resolution
        dx = int(px * depth_w / img_w)
        dy = int(py * depth_h / img_h)
        if 0 <= dx < depth_w and 0 <= dy < depth_h:
            rel_depths.append(depth[dy, dx])
            true_depths.append(true_z)

    if len(rel_depths) < 10:
        return None, None, None  # not enough points to calibrate

    rel_depths = np.array(rel_depths)
    true_depths = np.array(true_depths)

    # Least squares fit in inverse-depth space: 1/true = a * rel + b.
    # Refit on the best 80% a few times: bad sparse matches otherwise flip the slope.
    A = np.vstack([rel_depths, np.ones_like(rel_depths)]).T
    inv_true = 1.0 / true_depths
    keep = np.ones(len(inv_true), bool)
    for _ in range(3):
        (a, b), *_ = np.linalg.lstsq(A[keep], inv_true[keep], rcond=None)
        residual = np.abs(A @ [a, b] - inv_true)
        keep = residual <= np.percentile(residual, 80)

    return a, b, true_depths.max()


# =============================================================================
# BACK-PROJECTION
# =============================================================================
def backproject(depth_metric: np.ndarray,
                rgb: np.ndarray,
                fx, fy, cx, cy,
                R: np.ndarray, t: np.ndarray,
                img_w: int, img_h: int,
                stride: int):
    """
    Back-project a metric depth map into world-space 3D points.
    Returns (points Nx3, colors Nx3).

    Camera model: X_cam = R @ X_world + t  =>  X_world = R^T @ (X_cam - t)
    Pixel (u,v) with depth d:
      x_cam = (u - cx) * d / fx
      y_cam = (v - cy) * d / fy
      z_cam = d
    """
    depth_h, depth_w = depth_metric.shape

    # Build pixel grids (subsampled)
    us = np.arange(0, depth_w, stride)
    vs = np.arange(0, depth_h, stride)
    uu, vv = np.meshgrid(us, vs)

    # Scale pixel coords from depth-map resolution to original image resolution
    # (intrinsics fx,fy,cx,cy are defined in the original image resolution)
    scale_x = img_w / depth_w
    scale_y = img_h / depth_h
    u_img = uu * scale_x
    v_img = vv * scale_y

    d = depth_metric[vv, uu]

    valid = d > 1e-6

    x_cam = (u_img - cx) * d / fx
    y_cam = (v_img - cy) * d / fy
    z_cam = d

    cam_points = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)  # Nx3

    # World = R^T @ (cam - t)
    # For row vectors: X_world = (X_cam - t) @ R  (since (R^T @ v)^T = v^T @ R)
    world_points = (cam_points - t) @ R

    # Colors
    rgb_resized = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
    colors = rgb_resized[vv, uu][valid]  # BGR
    colors = colors[:, ::-1]  # -> RGB

    return world_points, colors


# =============================================================================
# PLY WRITER
# =============================================================================
def write_ply(path, points: np.ndarray, colors: np.ndarray):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse",  default=SPARSE_DIR)
    parser.add_argument("--images",  default=IMAGE_DIR)
    parser.add_argument("--output",  default=OUTPUT_DIR)
    parser.add_argument("--stride",  type=int, default=STRIDE,
                        help="Pixel subsample stride for back-projection")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading COLMAP sparse model...")
    cameras  = read_cameras(os.path.join(args.sparse, "cameras.txt"))
    images   = read_images(os.path.join(args.sparse, "images.txt"))
    points3d = read_points3d(os.path.join(args.sparse, "points3D.txt"))
    print(f"  {len(cameras)} camera(s), {len(images)} registered image(s), "
          f"{len(points3d)} sparse 3D points")

    print("\nLoading Depth Anything V2...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=device,
    )

    all_points = []
    all_colors = []

    for name, image_data in images.items():
        image_path = os.path.join(args.images, name)
        if not os.path.exists(image_path):
            print(f"  [SKIP] {name} — file not found in {args.images}")
            continue

        print(f"\nProcessing {name}...")

        camera = cameras[image_data["camera_id"]]
        fx, fy, cx, cy = get_intrinsics(camera)
        img_w, img_h = camera["width"], camera["height"]

        R = qvec_to_rotmat(image_data["qvec"])
        t = image_data["tvec"]

        # Depth estimation
        depth = estimate_depth(depth_pipe, image_path)
        depth_h, depth_w = depth.shape

        # Calibrate scale using sparse points
        a, b, max_sparse_z = calibrate_scale(
            depth, image_data, points3d,
            fx, fy, cx, cy, R, t, img_w, img_h, depth_w, depth_h
        )

        if a is None:
            print(f"  [SKIP] Not enough sparse points to calibrate scale")
            continue

        print(f"  Calibration: 1/true_depth = {a:.6f} * rel_depth + {b:.6f}")
        if a <= 0:
            print(f"  [WARN] Negative/zero scale factor — calibration may be unreliable")

        inv_depth = a * depth + b
        depth_metric = np.where(inv_depth > 0, 1.0 / np.maximum(inv_depth, 1e-12), 0.0)
        # Drop sky / far background: nothing beyond 3x the farthest sparse point.
        depth_metric[depth_metric > 3 * max_sparse_z] = 0.0  # backproject skips depth 0

        # Load RGB
        rgb = cv2.imread(image_path)

        # Back-project
        points, colors = backproject(
            depth_metric, rgb, fx, fy, cx, cy, R, t, img_w, img_h, args.stride
        )
        print(f"  Generated {len(points):,} 3D points")

        all_points.append(points)
        all_colors.append(colors)

    if not all_points:
        print("\n[ERROR] No points generated. Check that image names match "
              "between sparse model and image directory.")
        sys.exit(1)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    out_path = os.path.join(args.output, "dense_from_depth.ply")
    print(f"\nWriting {len(all_points):,} total points to {out_path}...")
    write_ply(out_path, all_points, all_colors)

    print(f"\n{'='*52}")
    print(f"  Done! Dense point cloud (from depth) saved to:")
    print(f"  {out_path}")
    print(f"\n  Next step — clean up with open3d:")
    print(f"    python open3d_postprocess.py --input {out_path}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()