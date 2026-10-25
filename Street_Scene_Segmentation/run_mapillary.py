"""Run a Mapillary-Vistas-trained Mask2Former checkpoint on local images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation



def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalized_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


def resolve_class_ids(model: Any, class_config: list[dict[str, Any]]) -> dict[str, int]:
    labels = model.config.id2label
    normalized_labels = {normalized_label(str(label)): int(index) for index, label in labels.items()}
    resolved = {}
    missing = []
    for item in class_config:
        name = item["name"]
        class_id = normalized_labels.get(normalized_label(name))
        if class_id is None:
            missing.append(name)
        else:
            resolved[name] = class_id
    if missing:
        available = ", ".join(sorted(normalized_labels))
        raise RuntimeError(f"Model labels missing {missing}. Available labels: {available}")
    return resolved


def choose_images(source_dir: Path, count: int, seed: int) -> list[Path]:
    images = sorted(
        path for path in source_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise RuntimeError(f"No images found in {source_dir}")
    randomizer = random.Random(seed)
    randomizer.shuffle(images)
    return images[: min(count, len(images))]


def make_visualization(
    image: Image.Image,
    local_mask: np.ndarray,
    class_config: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_pixels = np.asarray(overlay).copy()
    for item in class_config:
        color = tuple(item["color"])
        value = item["mask_value"]
        selected = local_mask == value
        overlay_pixels[selected] = (*color, 125)
    overlay = Image.fromarray(overlay_pixels, mode="RGBA")
    result = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    x, y = 25, 25
    for item in class_config:
        color = tuple(item["color"]) + (255,)
        draw.rectangle((x, y, x + 24, y + 24), fill=color)
        draw.text((x + 34, y - 3), item["name"], fill=(255, 255, 255, 255), font=font)
        x += 190

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path)


def save_mask_previews(
    local_mask: np.ndarray,
    class_config: list[dict[str, Any]],
    mask_dir: Path,
    preview_dir: Path,
    stem: str,
) -> None:
    palette = np.zeros((*local_mask.shape, 3), dtype=np.uint8)
    for item in class_config:
        palette[local_mask == item["mask_value"]] = item["color"]
        binary = np.where(local_mask == item["mask_value"], 255, 0).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(mask_dir / f"{stem}_{item['name']}_binary.png")

    Image.fromarray(palette, mode="RGB").save(preview_dir / f"{stem}_flat_mask.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("mapillary_config.json"),
    )
    parser.add_argument("--image-count", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    model_config = config["model"]
    data_config = config["data"]
    output_config = config["outputs"]
    class_config = config["classes"]
    run_name = args.run_name or output_config["run_name"]
    if not run_name or Path(run_name).name != run_name:
        raise SystemExit("--run-name must be a simple folder name")
    image_count = args.image_count or int(data_config["image_count"])
    if image_count <= 0:
        raise SystemExit("--image-count must be positive")

    source_dir = (config_path.parent / data_config["source_dir"]).resolve()
    output_dir = config_path.parent / output_config["root_dir"] / run_name
    mask_dir = output_dir / "masks"
    visualization_dir = output_dir / "visualizations"
    preview_dir = output_dir / "mask_previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_config['model_id']} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_config["model_id"])
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_config["model_id"]).to(device)
    model.eval()
    class_ids = resolve_class_ids(model, class_config)
    print(f"Resolved model classes: {class_ids}")

    records = []
    for image_path in choose_images(source_dir, image_count, int(data_config["random_seed"])):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        prediction = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0].cpu().numpy()

        local_mask = np.zeros(prediction.shape, dtype=np.uint8)
        for item in class_config:
            local_mask[prediction == class_ids[item["name"]]] = item["mask_value"]

        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(local_mask, mode="L").save(mask_path)
        save_mask_previews(local_mask, class_config, mask_dir, preview_dir, image_path.stem)
        visualization_path = visualization_dir / f"{image_path.stem}_visualization.png"
        make_visualization(image, local_mask, class_config, visualization_path)
        records.append(
            {
                "image": image_path.name,
                "mask": str(mask_path.relative_to(output_dir)),
                "visualization": str(visualization_path.relative_to(output_dir)),
            }
        )
        print(f"Processed {image_path.name}")

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_id": model_config["model_id"],
                "device": str(device),
                "class_ids": class_ids,
                "classes": class_config,
                "images": records,
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
