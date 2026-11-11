"""Prepare the LabelMe balcony dataset for Grounding DINO evaluation.

The source annotations contain polygons. This processor preserves each polygon
as COCO segmentation and also emits its axis-aligned detection bounding box.
Incomplete image/annotation pairs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CATEGORY_ID = 1
CATEGORY_NAME = "Balcony"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def polygon_area(points: list[list[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def valid_points(points: Any) -> bool:
    if not isinstance(points, list) or len(points) < 3:
        return False
    return all(
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
        for point in points
    )


def find_image(source_dir: Path, annotation_path: Path, image_path: Any) -> Path | None:
    candidates = []
    if isinstance(image_path, str) and image_path:
        candidates.append(source_dir / Path(image_path).name)
    candidates.extend(source_dir / f"{annotation_path.stem}{extension}" for extension in IMAGE_EXTENSIONS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_annotation(annotation_path: Path, source_dir: Path, image_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    try:
        with annotation_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"WARNING: skipping {annotation_path.name}: {error}", file=sys.stderr)
        return None

    image_path = find_image(source_dir, annotation_path, data.get("imagePath"))
    if image_path is None:
        print(f"WARNING: skipping {annotation_path.name}: matching image not found", file=sys.stderr)
        return None

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        print(f"WARNING: skipping {annotation_path.name}: invalid image dimensions", file=sys.stderr)
        return None

    image_record = {
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }
    annotations = []

    for shape_index, shape in enumerate(data.get("shapes", [])):
        if shape.get("label") != CATEGORY_NAME or shape.get("shape_type") != "polygon":
            continue

        points = shape.get("points")
        if not valid_points(points):
            print(
                f"WARNING: ignoring {annotation_path.name} shape {shape_index}: invalid polygon",
                file=sys.stderr,
            )
            continue

        clipped_points = [
            [min(max(float(x), 0.0), float(width)), min(max(float(y), 0.0), float(height))]
            for x, y in points
        ]
        xmin = min(point[0] for point in clipped_points)
        ymin = min(point[1] for point in clipped_points)
        xmax = max(point[0] for point in clipped_points)
        ymax = max(point[1] for point in clipped_points)
        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 0 or box_height <= 0:
            print(
                f"WARNING: ignoring {annotation_path.name} shape {shape_index}: empty bounding box",
                file=sys.stderr,
            )
            continue

        annotations.append(
            {
                "id": len(annotations),
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "segmentation": [[coordinate for point in clipped_points for coordinate in point]],
                "area": polygon_area(clipped_points),
                "bbox": [xmin, ymin, box_width, box_height],
                "iscrowd": 0,
            }
        )

    return image_record, annotations


def build_coco(records: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1

    for image, image_annotations in records:
        images.append(image)
        for annotation in image_annotations:
            annotation["id"] = annotation_id
            annotations.append(annotation)
            annotation_id += 1

    return {
        "info": {
            "description": "Balcony polygons converted from LabelMe annotations",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "structure"}],
        "images": images,
        "annotations": annotations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grounding_dino_config.json"),
        help="Path to the Grounding DINO JSON configuration.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    data_config = config["data"]
    source_dir = (config_path.parent / data_config["source_dir"]).resolve()
    output_dir = (config_path.parent / data_config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(source_dir.glob("*.json"))
    records = []
    for image_id, annotation_path in enumerate(annotation_paths, start=1):
        record = read_annotation(annotation_path, source_dir, image_id)
        if record is not None:
            records.append(record)

    randomizer = random.Random(data_config["random_seed"])
    randomizer.shuffle(records)
    validation_fraction = float(data_config["validation_fraction"])
    validation_count = round(len(records) * validation_fraction)
    validation_records = records[:validation_count]
    training_records = records[validation_count:]

    for filename, split_records in (
        ("train.json", training_records),
        ("val.json", validation_records),
        ("all.json", records),
    ):
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(build_coco(split_records), file, indent=2)
            file.write("\n")

    print(f"Source directory: {source_dir}")
    print(f"Matched annotated images: {len(records)}")
    print(f"Training images: {len(training_records)}")
    print(f"Validation images: {len(validation_records)}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
