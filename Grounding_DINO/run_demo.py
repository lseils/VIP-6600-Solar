"""Run Grounding DINO on a small sample and render evaluation panels."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, GroundingDinoProcessor


HIT = (30, 220, 80, 255)
MISS = (255, 45, 45, 255)
FALSE_POSITIVE = (30, 130, 255, 255)
GROUND_TRUTH = (255, 220, 40, 255)
PREDICTION = (30, 190, 220, 255)
FILL_ALPHA = 65
CLASS_COLORS = {
    "tree": (40, 210, 70, 255),
    "building facade": (255, 145, 30, 255),
    "ground": (170, 105, 45, 255),
    "sky": (40, 150, 255, 255),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def intersection_over_union(first: list[float], second: list[float]) -> float:
    first_xmin, first_ymin, first_width, first_height = first
    second_xmin, second_ymin, second_width, second_height = second
    first_xmax = first_xmin + first_width
    first_ymax = first_ymin + first_height
    second_xmax = second_xmin + second_width
    second_ymax = second_ymin + second_height

    overlap_width = max(0.0, min(first_xmax, second_xmax) - max(first_xmin, second_xmin))
    overlap_height = max(0.0, min(first_ymax, second_ymax) - max(first_ymin, second_ymin))
    intersection = overlap_width * overlap_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union else 0.0


def match_boxes(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for ground_truth_index, expected in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            score = intersection_over_union(expected["bbox"], prediction["bbox"])
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

    return matches, set(range(len(ground_truth))) - matched_ground_truth, set(range(len(predictions))) - matched_predictions


def polygon_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    values = annotation["segmentation"][0]
    return list(zip(values[::2], values[1::2]))


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


def draw_ground_truth_panel(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(panel)
    font = load_font()

    for annotation in ground_truth:
        color = GROUND_TRUTH
        points = polygon_points(annotation)
        overlay_draw.polygon(points, fill=(*color[:3], FILL_ALPHA), outline=color, width=8)
        draw.line(points + [points[0]], fill=color, width=8, joint="curve")
    panel = Image.alpha_composite(panel, overlay)
    draw = ImageDraw.Draw(panel)
    draw_title(draw, title, font)
    return panel


def draw_prediction_panel(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(panel)
    font = load_font()

    for prediction in predictions:
        label = str(prediction.get("label", "object")).lower()
        color = CLASS_COLORS.get(label, PREDICTION)
        xmin, ymin, width, height = prediction["bbox"]
        draw.rectangle((xmin, ymin, xmin + width, ymin + height), outline=color, width=8)
        text = f"{prediction.get('label', 'object')} {prediction['score']:.2f}"
        draw.text((xmin, ymin), text, fill=color, font=font)
    legend_x = 30
    legend_y = 30
    for label, color in CLASS_COLORS.items():
        draw.rectangle(
            (legend_x, legend_y + 7, legend_x + 25, legend_y + 32),
            fill=color,
        )
        draw.text(
            (legend_x + 35, legend_y),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )
        legend_x += 260
    draw_title(draw, title, font)
    return panel


def draw_difference_panel(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matches: list[tuple[int, int, float]],
    missed_ground_truth: set[int],
    false_predictions: set[int],
    title: str,
) -> Image.Image:
    panel = ImageOps.grayscale(image.copy()).convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(panel)
    font = load_font()

    for ground_truth_index in missed_ground_truth:
        points = polygon_points(ground_truth[ground_truth_index])
        overlay_draw.polygon(points, fill=(*MISS[:3], FILL_ALPHA), outline=MISS, width=8)
    for ground_truth_index, _, _ in matches:
        points = polygon_points(ground_truth[ground_truth_index])
        draw.line(points + [points[0]], fill=HIT, width=8, joint="curve")
    panel = Image.alpha_composite(panel, overlay)
    draw = ImageDraw.Draw(panel)

    for prediction_index in false_predictions:
        prediction = predictions[prediction_index]
        xmin, ymin, width, height = prediction["bbox"]
        draw.rectangle((xmin, ymin, xmin + width, ymin + height), outline=FALSE_POSITIVE, width=8)
    draw_legend(draw, font)
    draw_title(draw, title, font)
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
    panel.save(output_path)


def run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    inputs = processor(images=image, text=[prompt], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    predictions = []
    labels = result.get("text_labels", result.get("labels", []))
    for box, score, label in zip(
        result["boxes"].tolist(),
        result["scores"].tolist(),
        labels,
    ):
        xmin, ymin, xmax, ymax = box
        predictions.append(
            {
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "score": score,
                "label": str(label),
            }
        )
    return predictions


def load_model(model_id: str, device: torch.device) -> tuple[Any, Any]:
    """Load the configured Grounding DINO processor and model."""
    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grounding_dino_config.json"),
    )
    parser.add_argument("--run-name", default=None, help="Named output folder for this run.")
    parser.add_argument("--image-count", type=int, default=None, help="Number of images to evaluate.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    data_config = config["data"]
    dino_config = config["grounding_dino"]
    runs_config = config["runs"]
    run_name = args.run_name or runs_config["default_run_name"]
    if not run_name or Path(run_name).name != run_name:
        raise SystemExit("--run-name must be a simple folder name")
    source_dir = (config_path.parent / data_config["source_dir"]).resolve()
    randomizer = random.Random(dino_config["inference_seed"])
    if data_config.get("inference_only", False):
        sampled_images = [
            {"id": index, "file_name": path.name}
            for index, path in enumerate(
                sorted(source_dir.glob("*.jpg")), start=1
            )
        ]
        annotations_by_image = {}
    else:
        annotations_path = config_path.parent / data_config["output_dir"] / "all.json"
        annotations_data = load_json(annotations_path.resolve())
        sampled_images = list(annotations_data["images"])
        annotations_by_image = {}
        for annotation in annotations_data["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    randomizer.shuffle(sampled_images)
    image_count = args.image_count or int(dino_config["demo_image_count"])
    if image_count <= 0:
        raise SystemExit("--image-count must be positive")
    sampled_images = sampled_images[:image_count]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {dino_config['model_id']} on {device}...")
    model, processor = load_model(dino_config["model_id"], device)

    output_dir = config_path.parent / runs_config["output_dir"] / run_name / "dino"
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_predictions = []
    iou_threshold = float(dino_config["iou_threshold"])

    for sample_index, image_record in enumerate(sampled_images, start=1):
        image_path = source_dir / image_record["file_name"]
        image = Image.open(image_path).convert("RGB")
        ground_truth = annotations_by_image.get(image_record["id"], [])
        predictions = run_inference(
            model,
            processor,
            image,
            dino_config["text_prompt"],
            float(dino_config["box_threshold"]),
            float(dino_config["text_threshold"]),
            device,
        )
        matches, missed_ground_truth, false_predictions = (
            match_boxes(ground_truth, predictions, iou_threshold)
            if ground_truth
            else ([], set(), set(range(len(predictions))))
        )
        matched_ground_truth = {ground_truth_index for ground_truth_index, _, _ in matches}
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}
        matched_ious = [score for _, _, score in matches]
        metrics = {
            "image": image_record["file_name"],
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
            "true_positives": len(matches),
            "false_predictions": len(false_predictions),
            "misses": len(missed_ground_truth),
            "precision": len(matches) / len(predictions) if predictions else 0.0,
            "recall": len(matches) / len(ground_truth) if ground_truth else 0.0,
            "f1": (
                2 * len(matches) / (2 * len(matches) + len(false_predictions) + len(missed_ground_truth))
                if matches or false_predictions or missed_ground_truth
                else 1.0
            ),
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        }
        all_metrics.append(metrics)
        all_predictions.append({"image": image_record["file_name"], "predictions": predictions})

        if data_config.get("inference_only", False):
            panels = [draw_prediction_panel(
                image, predictions, "DINO PREDICTIONS"
            )]
        else:
            panels = [
                draw_ground_truth_panel(
                    image, ground_truth,
                    f"GROUND TRUTH | P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f}",
                ),
                draw_prediction_panel(
                    image, predictions,
                    f"DINO PREDICTIONS | P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f}",
                ),
                draw_difference_panel(
                    image, ground_truth, predictions, matches,
                    missed_ground_truth, false_predictions,
                    f"DINO COMPARISON | P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f}",
                ),
            ]
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_record["file_name"]).stem)
        output_path = visualization_dir / f"{sample_index:02d}_{safe_stem}_comparison.png"
        make_panel(panels, output_path)
        print(
            f"{image_record['file_name']}: TP={metrics['true_positives']} "
            f"FP={metrics['false_predictions']} FN={metrics['misses']} "
            f"F1={metrics['f1']:.3f}"
        )

    total_true_positives = sum(metric["true_positives"] for metric in all_metrics)
    total_false_predictions = sum(metric["false_predictions"] for metric in all_metrics)
    total_misses = sum(metric["misses"] for metric in all_metrics)
    aggregate = {
        "images": len(all_metrics),
        "true_positives": total_true_positives,
        "false_predictions": total_false_predictions,
        "misses": total_misses,
        "precision": total_true_positives / (total_true_positives + total_false_predictions)
        if total_true_positives + total_false_predictions
        else 0.0,
        "recall": total_true_positives / (total_true_positives + total_misses)
        if total_true_positives + total_misses
        else 0.0,
        "f1": 2 * total_true_positives / (2 * total_true_positives + total_false_predictions + total_misses)
        if total_true_positives + total_false_predictions + total_misses
        else 1.0,
        "mean_matched_iou": sum(metric["mean_matched_iou"] for metric in all_metrics) / len(all_metrics),
        "iou_threshold": iou_threshold,
        "box_threshold": dino_config["box_threshold"],
        "text_threshold": dino_config["text_threshold"],
        "device": str(device),
        "model_id": dino_config["model_id"],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"aggregate": aggregate, "per_image": all_metrics}, file, indent=2)
        file.write("\n")
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(all_predictions, file, indent=2)
        file.write("\n")
    print(f"Aggregate: {json.dumps(aggregate)}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
