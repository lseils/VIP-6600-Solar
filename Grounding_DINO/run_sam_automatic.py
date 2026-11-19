"""Run prompt-free automatic SAM mask generation on the configured images."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor

from run_sam import draw_mask_panel, load_json, load_model, make_panel


def automatic_sam_masks(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: torch.device,
    grid_size: int,
) -> list[np.ndarray]:
    """Generate unlabeled masks from a regular positive-point grid."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive.")
    width, height = image.size
    points = [
        [[
            width * (column + 0.5) / grid_size,
            height * (row + 0.5) / grid_size,
        ]]
        for row in range(grid_size)
        for column in range(grid_size)
    ]
    labels = [[1] for _ in points]
    inputs = processor(
        images=image,
        input_points=[points],
        input_labels=[labels],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    masks = masks[:, 0].numpy().astype(bool)
    unique_masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        key = np.packbits(mask, axis=None).tobytes()
        if mask.any() and key not in seen:
            seen.add(key)
            unique_masks.append(mask)
    return unique_masks


def draw_point_grid(
    image: Image.Image,
    grid_size: int,
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(panel)
    width, height = image.size
    for row in range(grid_size):
        for column in range(grid_size):
            x = width * (column + 0.5) / grid_size
            y = height * (row + 0.5) / grid_size
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(40, 190, 220, 255))
    draw.rectangle((20, 20, 900, 70), fill=(20, 20, 20, 220))
    draw.text((30, 30), title, fill=(255, 255, 255, 255))
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grounding_dino_config.json"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--image-count", type=int, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override the image directory from the configuration.",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()
    if args.grid_size < 1:
        parser.error("--grid-size must be positive")

    config_path = args.config.resolve()
    config = load_json(config_path)
    data_config = config["data"]
    sam_config = config["sam"]
    runs_config = config["runs"]
    run_name = args.run_name or f"{runs_config['default_run_name']}_sam_automatic"
    if not run_name or Path(run_name).name != run_name:
        parser.error("--run-name must be a simple folder name")
    source_dir = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else (config_path.parent / data_config["source_dir"]).resolve()
    )
    image_paths = sorted(
        path for path in source_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    randomizer = random.Random(
        int(config["grounding_dino"].get("inference_seed", 6600))
    )
    randomizer.shuffle(image_paths)
    image_count = args.image_count or int(
        config["grounding_dino"]["demo_image_count"]
    )
    if image_count <= 0:
        parser.error("--image-count must be positive")
    image_paths = image_paths[:image_count]
    output_dir = config_path.parent / runs_config["output_dir"] / run_name
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {sam_config['model_id']} on {device}...")
    model, processor = load_model(sam_config["model_id"], device)

    metrics = []
    for index, image_path in enumerate(image_paths, start=1):
        image_name = image_path.name
        image = Image.open(image_path).convert("RGB")
        masks = automatic_sam_masks(
            model, processor, image, device, args.grid_size
        )
        colors = [
            (
                40 + (mask_index * 67) % 190,
                80 + (mask_index * 97) % 160,
                100 + (mask_index * 43) % 140,
                255,
            )
            for mask_index in range(len(masks))
        ]
        panels = [
            draw_point_grid(
                image,
                args.grid_size,
                "SAM-ONLY INPUT POINT GRID (NO DINO BOXES)",
            ),
            draw_mask_panel(
                image,
                masks,
                colors,
                ["unlabeled SAM mask"] * len(masks),
                "SAM-ONLY AUTOMATIC MASKS (UNLABELED)",
                show_category_legend=False,
            ),
        ]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_name).stem)
        make_panel(
            panels,
            visualization_dir / f"{index:02d}_{safe_name}_sam_automatic.png",
        )
        metrics.append({"image": image_name, "mask_count": len(masks)})
        print(f"{image_name}: masks={len(masks)}")

    result = {
        "mode": "prompt-free positive-point grid",
        "images": len(metrics),
        "grid_size": args.grid_size,
        "device": str(device),
        "model_id": sam_config["model_id"],
        "note": (
            "SAM is class-agnostic in this test. Masks are intentionally "
            "unlabeled; no tree, facade, ground, or sky categories were "
            "assigned."
        ),
        "per_image": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sam_automatic_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
