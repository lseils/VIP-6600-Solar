"""Run CLIP-guided semantic labeling of candidate SAM masks."""

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
from transformers import CLIPModel, CLIPProcessor

from run_sam import draw_mask_panel, load_json, load_model, make_panel
from run_sam_automatic import automatic_sam_masks


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
CLASS_PROMPTS = {
    "tree": "a street-view photo of a tree or vegetation",
    "building facade": "a street-view photo of a building facade",
    "ground": "a street-view photo of pavement, road, or ground",
    "sky": "a street-view photo of sky or clouds",
}
CLASS_COLORS = {
    "tree": (40, 210, 70, 255),
    "building facade": (255, 145, 30, 255),
    "ground": (170, 105, 45, 255),
    "sky": (40, 150, 255, 255),
}


def masked_crop(
    image: Image.Image,
    mask: np.ndarray,
    padding_fraction: float = 0.08,
) -> Image.Image:
    """Create a CLIP crop with pixels outside the candidate mask neutralized."""
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return image.copy()
    xmin, xmax = columns.min(), columns.max() + 1
    ymin, ymax = rows.min(), rows.max() + 1
    padding_x = max(1, round((xmax - xmin) * padding_fraction))
    padding_y = max(1, round((ymax - ymin) * padding_fraction))
    xmin = max(0, xmin - padding_x)
    ymin = max(0, ymin - padding_y)
    xmax = min(image.width, xmax + padding_x)
    ymax = min(image.height, ymax + padding_y)
    crop = np.asarray(image)[ymin:ymax, xmin:xmax].copy()
    crop_mask = mask[ymin:ymax, xmin:xmax]
    crop[~crop_mask] = 128
    return Image.fromarray(crop)


def classify_masks(
    model: Any,
    processor: Any,
    image: Image.Image,
    masks: list[np.ndarray],
    device: torch.device,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Assign each SAM mask a CLIP category and probability."""
    labels = list(CLASS_PROMPTS)
    text_inputs = processor(
        text=list(CLASS_PROMPTS.values()),
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    classifications = []
    for start in range(0, len(masks), batch_size):
        crops = [masked_crop(image, mask) for mask in masks[start:start + batch_size]]
        image_inputs = processor(images=crops, return_tensors="pt").to(device)
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probabilities = (100 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = probabilities.max(dim=-1)
        for offset, (value, category_index) in enumerate(
            zip(values.cpu().tolist(), indices.cpu().tolist())
        ):
            classifications.append({
                "mask_index": start + offset,
                "label": labels[category_index],
                "probability": float(value),
                "scores": {
                    label: float(score)
                    for label, score in zip(
                        labels, probabilities[offset].cpu().tolist()
                    )
                },
            })
    return classifications


def select_masks(
    classifications: list[dict[str, Any]],
    maximum_per_class: int,
    minimum_probability: float,
) -> list[dict[str, Any]]:
    selected = []
    for label in CLASS_PROMPTS:
        candidates = [
            item for item in classifications
            if item["label"] == label
            and item["probability"] >= minimum_probability
        ]
        selected.extend(
            sorted(candidates, key=lambda item: item["probability"], reverse=True)
            [:maximum_per_class]
        )
    return selected


def draw_labeled_masks(
    image: Image.Image,
    masks: list[np.ndarray],
    selected: list[dict[str, Any]],
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    for item in selected:
        color = CLASS_COLORS[item["label"]]
        overlay = Image.new("RGBA", panel.size, (*color[:3], 0))
        overlay.putalpha(
            Image.fromarray((masks[item["mask_index"]] * 90).astype(np.uint8))
        )
        panel.alpha_composite(overlay)
    draw = ImageDraw.Draw(panel)
    for item in selected:
        rows, columns = np.nonzero(masks[item["mask_index"]])
        if not len(rows):
            continue
        x, y = int(columns.mean()), int(rows.mean())
        draw.text(
            (x, y),
            f"{item['label']} {item['probability']:.2f}",
            fill=CLASS_COLORS[item["label"]],
        )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name(
        "grounding_dino_config.json"
    ))
    parser.add_argument("--run-name", default="demo_10_clip_sam")
    parser.add_argument("--image-count", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--maximum-per-class", type=int, default=2)
    parser.add_argument("--minimum-probability", type=float, default=0.30)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    args = parser.parse_args()
    if args.image_count is not None and args.image_count <= 0:
        parser.error("--image-count must be positive")
    if args.grid_size <= 0 or args.maximum_per_class <= 0:
        parser.error("--grid-size and --maximum-per-class must be positive")
    if not 0.0 <= args.minimum_probability <= 1.0:
        parser.error("--minimum-probability must be between 0 and 1")
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a simple folder name")

    config_path = args.config.resolve()
    config = load_json(config_path)
    source_dir = (config_path.parent / config["data"]["source_dir"]).resolve()
    image_paths = [
        path for path in source_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    randomizer = random.Random(
        int(config["grounding_dino"].get("inference_seed", 6600))
    )
    randomizer.shuffle(image_paths)
    image_paths = image_paths[:args.image_count or int(
        config["grounding_dino"]["demo_image_count"]
    )]
    output_dir = config_path.parent / config["runs"]["output_dir"] / args.run_name
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM on {device}...")
    sam_model, sam_processor = load_model(
        config["sam"]["model_id"], device
    )
    print(f"Loading CLIP {args.clip_model} on {device}...")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()

    per_image = []
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        masks = automatic_sam_masks(
            sam_model, sam_processor, image, device, args.grid_size
        )
        classifications = classify_masks(
            clip_model, clip_processor, image, masks, device
        )
        selected = select_masks(
            classifications,
            args.maximum_per_class,
            args.minimum_probability,
        )
        colors = [
            CLASS_COLORS[item["label"]]
            for item in selected
        ]
        selected_masks = [masks[item["mask_index"]] for item in selected]
        panels = [
            image,
            draw_labeled_masks(image, masks, selected),
        ]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
        make_panel(
            panels,
            visualization_dir / f"{index:02d}_{safe_name}_clip_sam.png",
        )
        per_image.append({
            "image": image_path.name,
            "candidate_mask_count": len(masks),
            "selected_masks": selected,
            "selected_mask_count": len(selected_masks),
        })
        print(
            f"{image_path.name}: candidates={len(masks)} "
            f"selected={len(selected)}"
        )

    result = {
        "mode": "SAM candidate masks classified by CLIP text prompts",
        "clip_model": args.clip_model,
        "text_prompts": CLASS_PROMPTS,
        "grid_size": args.grid_size,
        "maximum_masks_per_class": args.maximum_per_class,
        "minimum_probability": args.minimum_probability,
        "device": str(device),
        "per_image": per_image,
        "note": (
            "CLIP provides semantic ranking but does not refine mask "
            "boundaries. Results should be reviewed before production use."
        ),
    }
    (output_dir / "clip_sam_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
