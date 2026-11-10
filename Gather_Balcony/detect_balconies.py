#!/usr/bin/env python3
"""
detect_balconies.py
--------------------
Uses Depth Anything V2 to estimate per-pixel depth from Street View
perspective crops, then detects balconies as regions that protrude
significantly beyond the dominant facade plane.

Steps:
  1. Load each perspective crop from INPUT_DIR
  2. Run Depth Anything V2 to get a relative depth map
  3. Detect the wall plane using the depth histogram
  4. Threshold pixels that protrude beyond the wall
  5. Find connected components → balcony bounding boxes
  6. Save annotated images to OUTPUT_DIR
  7. Write a summary JSON with all detections

Usage:
  python detect_balconies.py
  python detect_balconies.py --input perspective_crops --threshold 0.08
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR     = "perspective_crops"
OUTPUT_DIR    = "balcony_detections"
MODEL_ID      = "depth-anything/Depth-Anything-V2-Large-hf"  # best quality
MIN_AREA_PX   = 2000    # minimum blob area to count as a balcony (pixels)
PROTRUSION    = 0.08    # how far beyond wall plane (fraction of depth range)
                        # increase if too many false positives
                        # decrease if missing real balconies


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model():
    print("Loading Depth Anything V2...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device=device,
    )
    print("  Model loaded.")
    return pipe


# =============================================================================
# DEPTH ESTIMATION
# =============================================================================
def estimate_depth(pipe, image_path: str) -> np.ndarray:
    """Returns a depth map as a float32 numpy array, normalized 0-1."""
    img = Image.open(image_path).convert("RGB")
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)

    # Normalize to 0-1
    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin > 0:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


# =============================================================================
# WALL PLANE DETECTION
# =============================================================================
def find_wall_depth(depth: np.ndarray) -> float:
    """
    Find the dominant depth value (the wall) using a histogram peak.
    In Depth Anything output, HIGHER values = CLOSER to camera.
    The wall is the largest flat surface — it'll be the tallest histogram bin.
    """
    hist, bin_edges = np.histogram(depth.flatten(), bins=100)
    peak_bin = np.argmax(hist)
    wall_depth = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
    return wall_depth


# =============================================================================
# BALCONY DETECTION
# =============================================================================
def detect_balconies(depth: np.ndarray,
                     wall_depth: float,
                     protrusion: float,
                     min_area: int) -> list:
    """
    Find regions that are significantly closer to camera than the wall.
    Returns list of (x, y, w, h) bounding boxes.
    """
    # Pixels much closer than the wall = protruding = balcony candidates
    protrusion_mask = (depth > wall_depth + protrusion).astype(np.uint8) * 255

    # Morphological cleanup — remove tiny speckles, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_CLOSE, kernel)
    protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_OPEN, kernel)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        protrusion_mask, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, h, area))

    return boxes, protrusion_mask


# =============================================================================
# VISUALIZATION
# =============================================================================
def annotate_image(image_path: str,
                   depth: np.ndarray,
                   protrusion_mask: np.ndarray,
                   boxes: list,
                   wall_depth: float,
                   output_path: str):
    """Save side-by-side: original + depth map + detections."""
    orig = cv2.imread(image_path)
    h, w = orig.shape[:2]

    # Depth map as heatmap
    depth_vis = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(depth_color, (w, h))

    # Draw boxes on original
    annotated = orig.copy()
    mask_overlay = cv2.resize(protrusion_mask, (w, h))
    green_overlay = np.zeros_like(annotated)
    green_overlay[mask_overlay > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(annotated, 0.8, green_overlay, 0.2, 0)

    for i, (x, y, bw, bh, area) in enumerate(boxes):
        # Scale boxes to original image size if depth was different res
        sx = w / depth.shape[1]
        sy = h / depth.shape[0]
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + bw) * sx), int((y + bh) * sy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Balcony {i+1}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Wall depth line indicator on depth map
    wall_line_y = int((1.0 - wall_depth) * h)  # approximate
    cv2.line(depth_color, (0, wall_line_y), (w, wall_line_y), (255, 255, 255), 2)
    cv2.putText(depth_color, "wall plane", (10, wall_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stack side by side
    combined = np.hstack([annotated, depth_color])
    cv2.imwrite(output_path, combined)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default=INPUT_DIR)
    parser.add_argument("--output",    default=OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=PROTRUSION,
                        help="Protrusion threshold (0.0-1.0). Lower = more sensitive.")
    parser.add_argument("--min_area",  type=int,   default=MIN_AREA_PX,
                        help="Min blob area in pixels to count as balcony.")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {args.input}")
        return

    print(f"Found {len(image_files)} images in '{args.input}'")

    pipe = load_model()

    all_results = []
    total_balconies = 0

    for i, fname in enumerate(image_files):
        image_path = os.path.join(args.input, fname)
        print(f"\n[{i+1}/{len(image_files)}] {fname}")

        depth = estimate_depth(pipe, image_path)
        wall_depth = find_wall_depth(depth)
        print(f"  Wall depth value: {wall_depth:.3f}")

        boxes, mask = detect_balconies(
            depth, wall_depth, args.threshold, args.min_area
        )
        print(f"  Detected {len(boxes)} balcony candidate(s)")
        total_balconies += len(boxes)

        out_name = fname.replace(".jpg", "_detected.jpg").replace(".png", "_detected.jpg")
        out_path = os.path.join(args.output, out_name)
        annotate_image(image_path, depth, mask, boxes, wall_depth, out_path)

        all_results.append({
            "image": fname,
            "wall_depth": float(wall_depth),
            "balconies_detected": len(boxes),
            "boxes": [
                {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area_px": int(a)}
                for x, y, bw, bh, a in boxes
            ]
        })

    # Save summary JSON
    summary_path = os.path.join(args.output, "detections.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Done! {total_balconies} total balcony candidates across {len(image_files)} images")
    print(f"  Annotated images → {args.output}/")
    print(f"  Summary          → {summary_path}")
    print(f"\n  Tuning tips:")
    print(f"    Too many false positives? --threshold 0.12")
    print(f"    Missing balconies?        --threshold 0.05")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()