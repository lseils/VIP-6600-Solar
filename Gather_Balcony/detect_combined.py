#!/usr/bin/env python3
"""
detect_combined.py
-------------------
Runs BFA-YOLO balcony detection + Depth Anything V2 depth estimation
on perspective crops. Combines both outputs into annotated images and
a summary JSON.

For each image:
  - BFA-YOLO detects balconies, windows, doors, AC units, etc.
  - Depth Anything V2 estimates per-pixel depth
  - For each detected box, the median depth is extracted
  - Output: annotated image (detections + depth heatmap side by side)
  - Output: detections.json with boxes + depth values

Usage:
  python detect_combined.py
  python detect_combined.py --input perspective_crops --conf 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# Add yolo_balcony to path so its custom modules (block, head, rmt) are found
sys.path.insert(0, str(Path(__file__).parent / "yolo_balcony"))
from ultralytics import YOLO

# =============================================================================
# CONFIG
# =============================================================================
WEIGHTS      = "yolo_balcony/runs/detect/train3/weights/best.pt"
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Large-hf"
INPUT_DIR    = "perspective_crops"
OUTPUT_DIR   = "combined_detections"
CONF         = 0.25   # YOLO confidence threshold — lower = more detections
IMGSZ        = 640    # YOLO input size


# =============================================================================
# LOAD MODELS
# =============================================================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("Loading BFA-YOLO...")
    yolo = YOLO(WEIGHTS)
    print("  BFA-YOLO loaded.")

    print("Loading Depth Anything V2...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=device,
    )
    print("  Depth Anything V2 loaded.")

    return yolo, depth_pipe


# =============================================================================
# DEPTH ESTIMATION
# =============================================================================
def estimate_depth(depth_pipe, image_path: str) -> np.ndarray:
    """
    Returns relative depth map, normalized to 0-1 (higher = closer).
    """
    img = Image.open(image_path).convert("RGB")
    result = depth_pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin > 0:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


# =============================================================================
# VISUALIZATION
# =============================================================================
# Label colors per class (extend as needed)
CLASS_COLORS = {
    "balcony":          (0,   255,  0),    # green
    "window":           (255, 200,  0),    # yellow
    "door":             (0,   180, 255),   # blue
    "air conditioner":  (255,  80,  80),   # red
    "billboard":        (200,   0, 255),   # purple
    "glass curtain wall":(0,  255, 200),   # teal
}
DEFAULT_COLOR = (200, 200, 200)


def get_color(label: str) -> tuple:
    for key, color in CLASS_COLORS.items():
        if key in label.lower():
            return color
    return DEFAULT_COLOR


def annotate(orig: np.ndarray,
             depth: np.ndarray,
             detections: list) -> np.ndarray:
    h, w = orig.shape[:2]
    annotated = orig.copy()

    # Depth heatmap
    depth_vis = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(depth_color, (w, h))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf  = det["confidence"]
        color = get_color(label)

        # Draw box on original
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {conf:.2f} d={det['median_depth']:.2f}"
        cv2.putText(annotated, text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw box on depth map too
        cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)

    combined = np.hstack([annotated, depth_color])
    return combined


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--conf",   type=float, default=CONF,
                        help="YOLO confidence threshold (default 0.25)")
    parser.add_argument("--imgsz",  type=int,   default=IMGSZ)
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {args.input}")
        return

    print(f"Found {len(image_files)} images in '{args.input}'\n")

    yolo, depth_pipe = load_models()

    all_results = []
    total_detections = 0

    for i, fname in enumerate(image_files):
        image_path = os.path.join(args.input, fname)
        print(f"[{i+1}/{len(image_files)}] {fname}")

        # --- YOLO detection ---
        results = yolo.predict(
            source=image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )
        result = results[0]
        names  = result.names

        # --- Depth estimation (relative, normalized 0-1) ---
        depth = estimate_depth(depth_pipe, image_path)
        orig  = cv2.imread(image_path)
        oh, ow = orig.shape[:2]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Scale box to depth map coords and extract median depth
            dx1 = int(x1 * depth.shape[1] / ow)
            dy1 = int(y1 * depth.shape[0] / oh)
            dx2 = int(x2 * depth.shape[1] / ow)
            dy2 = int(y2 * depth.shape[0] / oh)
            roi = depth[dy1:dy2, dx1:dx2]
            median_depth = float(np.median(roi)) if roi.size > 0 else 0.0

            detections.append({
                "label":        label,
                "confidence":   round(conf, 3),
                "box":          [x1, y1, x2, y2],
                "median_depth": round(median_depth, 3),
            })

        det_summary = [(d['label'], d['median_depth']) for d in detections]
        print(f"  {len(detections)} detection(s): {det_summary}")
        total_detections += len(detections)

        # --- Annotate and save ---
        combined = annotate(orig, depth, detections)
        out_name = fname.replace(".jpg", "_combined.jpg").replace(".png", "_combined.jpg")
        cv2.imwrite(os.path.join(args.output, out_name), combined)

        all_results.append({
            "image":      fname,
            "detections": detections,
        })

    # Save summary JSON
    summary_path = os.path.join(args.output, "detections.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Done! {total_detections} total detections across {len(image_files)} images")
    print(f"  Annotated images → {args.output}/")
    print(f"  Summary          → {summary_path}")
    print(f"\n  Tuning tips:")
    print(f"    More detections?  --conf 0.15")
    print(f"    Fewer false positives? --conf 0.4")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()