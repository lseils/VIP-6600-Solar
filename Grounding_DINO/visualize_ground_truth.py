"""Render cleaned balcony annotations over their source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


POLYGON_FILL = (255, 40, 40, 70)
POLYGON_OUTLINE = (255, 230, 0, 255)
BOX_OUTLINE = (40, 130, 255, 255)
TEXT_FILL = (255, 255, 255, 255)
TEXT_BACKGROUND = (20, 20, 20, 210)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def annotation_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    values = annotation["segmentation"][0]
    return list(zip(values[::2], values[1::2]))


def render_image(image_path: Path, annotations: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)

    for annotation in annotations:
        points = annotation_points(annotation)
        overlay_draw.polygon(points, fill=POLYGON_FILL, outline=POLYGON_OUTLINE, width=5)

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    for index, annotation in enumerate(annotations, start=1):
        points = annotation_points(annotation)
        xmin, ymin, box_width, box_height = annotation["bbox"]
        xmax = xmin + box_width
        ymax = ymin + box_height
        draw.rectangle((xmin, ymin, xmax, ymax), outline=BOX_OUTLINE, width=4)

        label_x, label_y = points[0]
        label = f"Balcony {index}"
        text_box = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(text_box, fill=TEXT_BACKGROUND)
        draw.text((label_x, label_y), label, fill=TEXT_FILL, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_names",
        nargs="*",
        help="Source image names to render; defaults to the first two annotated images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grounding_dino_config.json"),
        help="Path to the Grounding DINO JSON configuration.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="COCO annotation file; defaults to prepared/all.json relative to the config.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    source_dir = (config_path.parent / config["data"]["source_dir"]).resolve()
    annotations_path = args.annotations or (
        config_path.parent / config["data"]["output_dir"] / "all.json"
    )
    annotations_data = load_json(annotations_path.resolve())

    images = sorted(annotations_data["images"], key=lambda image: image["file_name"])
    selected_names = args.image_names or [image["file_name"] for image in images[:2]]
    annotations_by_image = {}
    for annotation in annotations_data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    images_by_name = {image["file_name"]: image for image in images}

    for image_name in selected_names:
        image = images_by_name.get(image_name)
        if image is None:
            raise SystemExit(f"Image is not present in annotations: {image_name}")
        image_path = source_dir / image_name
        if not image_path.is_file():
            raise SystemExit(f"Image file is missing: {image_path}")
        output_path = config_path.parent / "visualizations" / f"{Path(image_name).stem}_ground_truth.png"
        render_image(image_path, annotations_by_image.get(image["id"], []), output_path)
        print(f"Rendered {image_name} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
