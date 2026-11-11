#!/usr/bin/env python3
"""
crop_perspective.py
--------------------
Reads raw equirectangular panoramas saved by fetch_streetview_tiles.py and
extracts three rectilinear perspective crops per panorama:
  - center: looking directly at the building facade
  - left:   30° left of the facade
  - right:  30° right of the facade

Correct rotation order: yaw (heading) first, then pitch.
The crop heading is TARGET_BEARING - car_heading, NOT car_heading directly.

Outputs:
  perspective_crops/pano_000_center.jpg  ← facing facade directly
  perspective_crops/pano_000_left.jpg    ← 30° left of facade
  perspective_crops/pano_000_right.jpg   ← 30° right of facade
  crop_manifest.json                     ← per-crop camera geometry (for locate_buildings.py)
"""

import json
import numpy as np
import cv2
from pathlib import Path

# =============================================================================
# CONFIG — tune these without re-downloading anything
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
LOCAL_INPUT_DIR = BASE_DIR / "street_images"
ROOT_INPUT_DIR = REPO_ROOT / "street_images"
INPUT_DIR = (
    LOCAL_INPUT_DIR
    if (LOCAL_INPUT_DIR / "panorama_metadata.json").is_file()
    else ROOT_INPUT_DIR
)
OUTPUT_DIR = BASE_DIR / "perspective_crops"  # Also used by the scene builder.
META_FILE = INPUT_DIR / "panorama_metadata.json"

# Compass bearing the building faces FROM the street.
# West = 270, East = 90, North = 0, South = 180.
TARGET_BEARING = 250.0

# Offsets (degrees) from TARGET_BEARING for the three crops.
# 0° = center, -30° = left, +30° = right.
CROP_OFFSETS = {
    "left":   -30.0,
    "center":   0.0,
    "right":  +30.0,
}

FOV_DEG   = 90     # horizontal FOV in degrees. 90° gives strong overlap at 4m spacing.
PITCH_DEG = 16.0   # degrees UP from horizon. Positive = tilt up toward upper floors.
OUT_W     = 2048   # output width  (pixels)
OUT_H     = 1536   # output height (pixels)


# =============================================================================
# Core projection: equirectangular → rectilinear perspective
# =============================================================================
def equirect_to_perspective(pano: np.ndarray,
                             fov_deg: float,
                             yaw_deg: float,
                             pitch_deg: float,
                             out_w: int,
                             out_h: int) -> np.ndarray:
    """
    Extract a rectilinear perspective crop from an equirectangular panorama.

    Args:
        pano:       H×W×3 equirectangular image (full 360°×180°).
        fov_deg:    Horizontal field of view of the output crop (degrees).
        yaw_deg:    Horizontal rotation from the panorama's 0° (straight ahead).
                    Positive = rotate right. Range: -180 to 180.
        pitch_deg:  Vertical tilt. Positive = tilt UP. Range: -90 to 90.
        out_w:      Output image width in pixels.
        out_h:      Output image height in pixels.

    Returns:
        out_h × out_w × 3 perspective image.
    """
    h, w = pano.shape[:2]

    fov   = np.radians(fov_deg)
    yaw   = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Focal length in pixels for a sensor of width out_w
    f  = (out_w / 2.0) / np.tan(fov / 2.0)
    cx = out_w / 2.0
    cy = out_h / 2.0

    # Build a grid of output pixel coordinates
    xs = np.linspace(0, out_w - 1, out_w)
    ys = np.linspace(0, out_h - 1, out_h)
    xv, yv = np.meshgrid(xs, ys)

    # Unproject pixels to unit rays in camera space
    # Camera looks along +Z; X is right; Y is up.
    rx =  (xv - cx) / f
    ry = -(yv - cy) / f   # flip so positive Y is up
    rz =  np.ones_like(rx)

    # --- Rotation 1: Pitch around the camera's X axis (positive = look up) ---
    rx2 = rx
    ry2 = ry * np.cos(pitch) + rz * np.sin(pitch)
    rz2 = -ry * np.sin(pitch) + rz * np.cos(pitch)

    # --- Rotation 2: Yaw around the vertical axis ---
    rx3 = rx2 * np.cos(yaw) + rz2 * np.sin(yaw)
    ry3 = ry2
    rz3 = -rx2 * np.sin(yaw) + rz2 * np.cos(yaw)

    # Normalize to unit sphere
    norm = np.sqrt(rx3**2 + ry3**2 + rz3**2)
    rx3 /= norm
    ry3 /= norm
    rz3 /= norm

    # Convert unit vector → spherical coordinates
    lon = np.arctan2(rx3, rz3)            # horizontal angle: -π to π
    lat = np.arcsin(np.clip(ry3, -1, 1))  # vertical angle:   -π/2 to π/2

    # Map spherical coords → equirectangular pixel coordinates
    map_x = ((lon / np.pi + 1.0) / 2.0 * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

    # Sample the panorama (BORDER_WRAP handles the 360° seam correctly)
    crop = cv2.remap(pano, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    return crop


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(META_FILE) as f:
        metas = json.load(f)

    print(f"Reading panoramas and metadata from: {INPUT_DIR}")
    print(f"Saving perspective crops to: {OUTPUT_DIR}")
    print(f"Target bearing (building direction): {TARGET_BEARING}°")
    print(f"Crop offsets: { {k: f'{TARGET_BEARING + v}°' for k, v in CROP_OFFSETS.items()} }")
    print(f"FOV: {FOV_DEG}°  |  Pitch: {PITCH_DEG}°  |  Output: {OUT_W}×{OUT_H}\n")

    manifest = []   # per-crop camera geometry, consumed by locate_buildings.py

    ok_count = 0
    for meta in metas:
        img_path = INPUT_DIR / meta["image_name"]
        pano = cv2.imread(str(img_path))

        if pano is None:
            print(f"[SKIP] {img_path} — file not found or unreadable")
            continue

        pano_h, pano_w = pano.shape[:2]
        print(f"[{meta['pano_index']:03d}] {meta['image_name']}  "
              f"({pano_w}×{pano_h})  car_heading={meta['car_heading']:.1f}°")

        for crop_name, bearing_offset in CROP_OFFSETS.items():
            # Rotate from pano's forward direction to face TARGET_BEARING + offset
            yaw_deg = (TARGET_BEARING + bearing_offset) - meta["car_heading"]

            # Normalize to [-180, 180]
            yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

            print(f"    [{crop_name:6s}] bearing={TARGET_BEARING + bearing_offset:.1f}°  "
                  f"yaw={yaw_deg:.1f}°")

            crop = equirect_to_perspective(
                pano,
                fov_deg=FOV_DEG,
                yaw_deg=yaw_deg,
                pitch_deg=PITCH_DEG,
                out_w=OUT_W,
                out_h=OUT_H,
            )

            base_name = meta["image_name"].replace(".jpg", "")
            out_name  = f"{base_name}_{crop_name}.jpg"
            out_path  = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            manifest.append({
                "crop_name":      out_name,
                "pano_id":        meta["pano_id"],
                "lat":            meta["lat"],
                "lng":            meta["lng"],
                "car_heading":    meta["car_heading"],
                "bearing_center": (TARGET_BEARING + bearing_offset) % 360.0,
                "yaw_deg":        yaw_deg,
                "pitch_deg":      PITCH_DEG,
                "fov_deg":        FOV_DEG,
                "out_w":          OUT_W,
                "out_h":          OUT_H,
            })
            ok_count += 1

        print()

    with open(BASE_DIR / "crop_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_panos = len(metas)
    print(f"{'='*52}")
    print(f"  Done! {ok_count} crops from {total_panos} panoramas saved to '{OUTPUT_DIR}/'")
    print(f"  ({total_panos} × {len(CROP_OFFSETS)} crops = {total_panos * len(CROP_OFFSETS)} total)")
    print(f"  Crop manifest → crop_manifest.json")
    print(f"  Next step: point run_colmap.sh at '{OUTPUT_DIR}/'")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()