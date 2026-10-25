"""Run SAM from Grounding DINO boxes and evaluate the resulting masks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from transformers import SamModel, SamProcessor


HIT = (30, 220, 80, 255)
MISS = (255, 45, 45, 255)
FALSE_POSITIVE = (30, 130, 255, 255)
GROUND_TRUTH = (255, 220, 40, 255)
DINO_INPUT = (30, 190, 220, 255)
SAM_OUTPUT = (255, 180, 40, 255)
FILL_ALPHA = 75


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def polygon_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    values = annotation["segmentation"][0]
    return list(zip(values[::2], values[1::2]))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def match_masks(
    ground_truth_masks: list[np.ndarray],
    predicted_masks: list[np.ndarray],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for ground_truth_index, expected in enumerate(ground_truth_masks):
        for prediction_index, prediction in enumerate(predicted_masks):
            score = mask_iou(expected, prediction)
            if score >= threshold:
                candidates.append((score, ground_truth_index, prediction_index))

    matches = []
    matched_ground_truth = set()
    matched_predictions = set()
    for score, ground_truth_index, prediction_index in sorted(candidates, reverse=True):
        if ground_truth_index in matched_ground_truth or prediction_index in matched_predictions:
            continue
        matches.append((ground_truth_index, prediction_index, score))
        matched_ground_truth.add(ground_truth_index)
        matched_predictions.add(prediction_index)

    return (
        matches,
        set(range(len(ground_truth_masks))) - matched_ground_truth,
        set(range(len(predicted_masks))) - matched_predictions,
    )


def annotation_mask(annotation: dict[str, Any], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon_points(annotation), fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        return ImageFont.load_default()


def draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    entries = ((HIT, "hit"), (MISS, "miss"), (FALSE_POSITIVE, "false prediction"))
    x, y = 30, 30
    for color, label in entries:
        draw.rectangle((x, y + 7, x + 25, y + 32), fill=color)
        draw.text((x + 35, y), label, fill=(255, 255, 255, 255), font=font)
        x += 230 if label != "false prediction" else 0
        if label == "false prediction":
            y += 45
            x = 30


def draw_title(draw: ImageDraw.ImageDraw, title: str, font: ImageFont.ImageFont) -> None:
    text_box = draw.textbbox((0, 0), title, font=font)
    text_width = text_box[2] - text_box[0]
    draw.rectangle((30, 115, 45 + text_width, 160), fill=(20, 20, 20, 220))
    draw.text((40, 120), title, fill=(255, 255, 255, 255), font=font)


def draw_mask(
    panel: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int, int],
    outline_width: int = 8,
) -> None:
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    overlay = Image.new("RGBA", panel.size, (*color[:3], 0))
    overlay.putalpha(mask_image.point(lambda value: value * FILL_ALPHA // 255))
    panel.alpha_composite(overlay)
    outline = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline)
    boundary = ImageChops.difference(mask_image, mask_image.filter(ImageFilter.MinFilter(3)))
    outline.putalpha(boundary)
    outline_draw.bitmap((0, 0), boundary, fill=color)
    panel.alpha_composite(outline)


def draw_mask_panel(
    image: Image.Image,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int, int]],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    for mask, color in zip(masks, colors):
        draw_mask(panel, mask, color)
    draw = ImageDraw.Draw(panel)
    font = load_font()
    draw_legend(draw, font)
    draw_title(draw, title, font)
    return panel


def draw_dino_input_panel(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(panel)
    font = load_font()
    for prediction in predictions:
        xmin, ymin, width, height = prediction["bbox"]
        draw.rectangle((xmin, ymin, xmin + width, ymin + height), outline=DINO_INPUT, width=8)
        draw.text((xmin, ymin), f"{prediction['score']:.2f}", fill=DINO_INPUT, font=font)
    draw_title(draw, title, font)
    return panel


def draw_difference_panel(
    image: Image.Image,
    ground_truth_masks: list[np.ndarray],
    predicted_masks: list[np.ndarray],
    matches: list[tuple[int, int, float]],
    missed_ground_truth: set[int],
    false_predictions: set[int],
) -> Image.Image:
    panel = ImageOps.grayscale(image.copy()).convert("RGBA")
    for index in missed_ground_truth:
        draw_mask(panel, ground_truth_masks[index], MISS)
    for ground_truth_index, _, _ in matches:
        draw_mask(panel, ground_truth_masks[ground_truth_index], HIT, outline_width=5)
    for index in false_predictions:
        draw_mask(panel, predicted_masks[index], FALSE_POSITIVE)
    draw = ImageDraw.Draw(panel)
    font = load_font()
    draw_legend(draw, font)
    draw_title(draw, "SAM DIFFERENCE", font)
    return panel


def make_panel(images: list[Image.Image], output_path: Path) -> None:
    target_width = 1280
    resized = []
    for image in images:
        ratio = target_width / image.width
        resized.append(image.resize((target_width, round(image.height * ratio)), Image.Resampling.LANCZOS))

    panel = Image.new("RGB", (target_width * len(resized), resized[0].height), (0, 0, 0))
    for index, image in enumerate(resized):
        panel.paste(image.convert("RGB"), (index * target_width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def run_sam(
    model: Any,
    processor: Any,
    image: Image.Image,
    boxes: list[list[float]],
    device: torch.device,
    multimask_output: bool,
) -> list[np.ndarray]:
    if not boxes:
        return []
    sam_boxes = [
        [xmin, ymin, xmin + width, ymin + height]
        for xmin, ymin, width, height in boxes
    ]
    inputs = processor(images=image, input_boxes=[sam_boxes], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=multimask_output)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    masks = masks[:, 0]
    return [mask.numpy().astype(bool) for mask in masks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grounding_dino_config.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Grounding DINO predictions.json; defaults to demo_outputs/predictions.json.",
    )
    parser.add_argument("--run-name", default=None, help="Named output folder for this run.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    data_config = config["data"]
    sam_config = config["sam"]
    runs_config = config["runs"]
    run_name = args.run_name or runs_config["default_run_name"]
    if not run_name or Path(run_name).name != run_name:
        raise SystemExit("--run-name must be a simple folder name")
    output_dir = config_path.parent / runs_config["output_dir"] / run_name / "sam"
    predictions_path = args.predictions or output_dir / "predictions.json"
    if args.predictions is None:
        predictions_path = config_path.parent / runs_config["output_dir"] / run_name / "dino" / "predictions.json"
    predictions_data = load_json(predictions_path.resolve())
    annotations_data = load_json((config_path.parent / data_config["output_dir"] / "all.json").resolve())
    source_dir = (config_path.parent / data_config["source_dir"]).resolve()

    annotations_by_image = {}
    for annotation in annotations_data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    image_by_name = {image["file_name"]: image for image in annotations_data["images"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {sam_config['model_id']} on {device}...")
    processor = SamProcessor.from_pretrained(sam_config["model_id"])
    model = SamModel.from_pretrained(sam_config["model_id"]).to(device)
    model.eval()

    metrics = []
    all_mask_ious = []
    mask_iou_threshold = float(sam_config["mask_iou_threshold"])
    for sample_index, item in enumerate(predictions_data, start=1):
        image_name = item["image"]
        image_record = image_by_name[image_name]
        image = Image.open(source_dir / image_name).convert("RGB")
        ground_truth = annotations_by_image[image_record["id"]]
        ground_truth_masks = [annotation_mask(annotation, image.width, image.height) for annotation in ground_truth]
        predictions = item["predictions"]
        predicted_masks = run_sam(
            model,
            processor,
            image,
            [prediction["bbox"] for prediction in predictions],
            device,
            bool(sam_config["multimask_output"]),
        )
        matches, misses, false_predictions = match_masks(
            ground_truth_masks, predicted_masks, mask_iou_threshold
        )
        matched_ious = [score for _, _, score in matches]
        all_mask_ious.extend(matched_ious)
        metrics.append(
            {
                "image": image_name,
                "ground_truth_count": len(ground_truth_masks),
                "prediction_count": len(predicted_masks),
                "true_positives": len(matches),
                "false_predictions": len(false_predictions),
                "misses": len(misses),
                "precision": len(matches) / len(predicted_masks) if predicted_masks else 0.0,
                "recall": len(matches) / len(ground_truth_masks) if ground_truth_masks else 0.0,
                "f1": (
                    2 * len(matches) / (2 * len(matches) + len(false_predictions) + len(misses))
                    if matches or false_predictions or misses
                    else 1.0
                ),
                "mean_matched_mask_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
            }
        )

        matched_ground_truth = {ground_truth_index for ground_truth_index, _, _ in matches}
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}
        gt_colors = [HIT if index in matched_ground_truth else MISS for index in range(len(ground_truth_masks))]
        prediction_colors = [HIT if index in matched_predictions else FALSE_POSITIVE for index in range(len(predicted_masks))]
        panels = [
            draw_mask_panel(image, ground_truth_masks, [GROUND_TRUTH] * len(ground_truth_masks), "SAM GROUND TRUTH"),
            draw_dino_input_panel(image, predictions, "DINO BOXES SENT TO SAM"),
            draw_mask_panel(image, predicted_masks, [SAM_OUTPUT] * len(predicted_masks), "SAM MASKS"),
            draw_difference_panel(image, ground_truth_masks, predicted_masks, matches, misses, false_predictions),
        ]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_name).stem)
        make_panel(
            panels,
            output_dir / "sam_visualizations" / f"{sample_index:02d}_{safe_name}_sam_comparison.png",
        )

        current = metrics[-1]
        print(
            f"{image_name}: TP={current['true_positives']} FP={current['false_predictions']} "
            f"FN={current['misses']} mask_F1={current['f1']:.3f}"
        )

    true_positives = sum(metric["true_positives"] for metric in metrics)
    false_predictions = sum(metric["false_predictions"] for metric in metrics)
    misses = sum(metric["misses"] for metric in metrics)
    aggregate = {
        "images": len(metrics),
        "true_positives": true_positives,
        "false_predictions": false_predictions,
        "misses": misses,
        "precision": true_positives / (true_positives + false_predictions)
        if true_positives + false_predictions
        else 0.0,
        "recall": true_positives / (true_positives + misses) if true_positives + misses else 0.0,
        "f1": 2 * true_positives / (2 * true_positives + false_predictions + misses)
        if true_positives + false_predictions + misses
        else 1.0,
        "mean_matched_mask_iou": sum(all_mask_ious) / len(all_mask_ious) if all_mask_ious else 0.0,
        "mask_iou_threshold": mask_iou_threshold,
        "device": str(device),
        "model_id": sam_config["model_id"],
        "source_predictions": str(predictions_path),
    }
    with (output_dir / "sam_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"aggregate": aggregate, "per_image": metrics}, file, indent=2)
        file.write("\n")
    print(f"Aggregate: {json.dumps(aggregate)}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())