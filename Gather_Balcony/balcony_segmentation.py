"""Ground balcony detections with the existing DINO and SAM pipelines."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


BALCONY_COLORS = (
    "#ff6d00",
    "#e91e63",
    "#00bcd4",
    "#ffeb3b",
    "#ab47bc",
    "#76ff03",
)
PROJECTION_MAX_WIDTH = 320
PROJECTION_DEPTH_DISCONTINUITY_M = 2.0
SURFACE_OVERLAY_OFFSET_M = 0.03
SAM_BATCH_SIZE = 8


def _grounding_dino_modules() -> tuple[Any, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    repository_root_text = str(repository_root)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)
    from Grounding_DINO import run_demo, run_sam

    return run_demo, run_sam


def infer_balcony_masks(
    image_path: Path,
    config_path: Path,
    overlay_path: Path,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    """Run existing Grounding DINO and SAM helpers on a perspective crop."""
    import torch

    run_demo, run_sam = _grounding_dino_modules()
    config = run_demo.load_json(config_path.resolve())
    dino_config = config["grounding_dino"]
    sam_config = config["sam"]
    image = Image.open(image_path).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Loading {dino_config['model_id']} on {device}...")
    dino_model, dino_processor = run_demo.load_model(
        str(dino_config["model_id"]), device
    )
    predictions = run_demo.run_inference(
        dino_model,
        dino_processor,
        image,
        str(dino_config["text_prompt"]),
        float(dino_config["box_threshold"]),
        float(dino_config["text_threshold"]),
        device,
    )
    del dino_model, dino_processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    masks: list[np.ndarray] = []
    if predictions:
        print(f"  Segmenting {len(predictions)} DINO detection(s) with SAM...")
        sam_model, sam_processor = run_sam.load_model(
            str(sam_config["model_id"]), device
        )
        for start in range(0, len(predictions), SAM_BATCH_SIZE):
            batch = predictions[start:start + SAM_BATCH_SIZE]
            masks.extend(run_sam.run_sam(
                sam_model,
                sam_processor,
                image,
                [prediction["bbox"] for prediction in batch],
                device,
                bool(sam_config["multimask_output"]),
            ))
        del sam_model, sam_processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(masks) != len(predictions):
        raise RuntimeError(
            "SAM returned a different number of masks than Grounding DINO "
            f"detections ({len(masks)} masks for {len(predictions)} boxes)."
        )
    for mask in masks:
        if mask.shape != (image.height, image.width):
            raise RuntimeError(
                "SAM mask dimensions do not match the input crop: "
                f"{mask.shape} versus {(image.height, image.width)}."
            )

    annotated = image.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    colors = [
        tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        + (255,)
        for color in BALCONY_COLORS
    ]
    for index, (prediction, mask) in enumerate(zip(predictions, masks)):
        color = colors[index % len(colors)]
        run_sam.draw_mask(annotated, mask, color)
        x, y, width, height = prediction["bbox"]
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=color,
            width=max(2, round(image.width / 1024)),
        )
        draw.text(
            (x, max(0, y - 22)),
            f"Balcony {index + 1}: {prediction['score']:.2f}",
            fill=color,
        )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.convert("RGB").save(overlay_path)

    return predictions, masks, {
        "status": "inferred",
        "image": str(image_path),
        "overlay": str(overlay_path),
        "model_id": str(dino_config["model_id"]),
        "sam_model_id": str(sam_config["model_id"]),
        "text_prompt": str(dino_config["text_prompt"]),
        "box_threshold": float(dino_config["box_threshold"]),
        "text_threshold": float(dino_config["text_threshold"]),
        "device": str(device),
        "detections": [
            {
                "index": index,
                "score": float(prediction["score"]),
                "bbox_xywh_pixels": [
                    float(value) for value in prediction["bbox"]
                ],
                "mask_pixel_count": int(np.count_nonzero(mask)),
            }
            for index, (prediction, mask) in enumerate(
                zip(predictions, masks), start=1
            )
        ],
    }


def _project_points(
    vertices: np.ndarray,
    crop: dict[str, Any],
    camera_height: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    directions = vertices - np.array([0.0, 0.0, camera_height])
    distances = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(distances[:, None], 1e-12)
    bearing = np.arctan2(directions[:, 0], directions[:, 1])
    bearing -= math.radians(float(crop["car_heading"]))
    elevation = np.arctan2(
        directions[:, 2],
        np.hypot(directions[:, 0], directions[:, 1]),
    )

    longitude_component = np.sin(bearing) * np.cos(elevation)
    vertical_component = np.sin(elevation)
    forward_component = np.cos(bearing) * np.cos(elevation)
    yaw = math.radians(float(crop["yaw_deg"]))
    pitch = math.radians(float(crop["pitch_deg"]))

    camera_x = longitude_component * math.cos(yaw) - forward_component * math.sin(yaw)
    pitch_forward = (
        longitude_component * math.sin(yaw)
        + forward_component * math.cos(yaw)
    )
    camera_y = vertical_component * math.cos(pitch) - pitch_forward * math.sin(pitch)
    camera_z = vertical_component * math.sin(pitch) + pitch_forward * math.cos(pitch)

    focal = (width / 2.0) / math.tan(
        math.radians(float(crop["fov_deg"])) / 2.0
    )
    pixels = np.column_stack((
        width / 2.0 + focal * camera_x / camera_z,
        height / 2.0 - focal * camera_y / camera_z,
    ))
    return pixels, camera_z * distances


def _rasterize_building_surfaces(
    meshes: list[dict[str, Any]],
    crop: dict[str, Any],
    camera_height: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    surface_points = np.full((height, width, 3), np.nan, dtype=np.float32)
    for mesh in meshes:
        if mesh.get("layer") != "buildings":
            continue
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        if not len(vertices) or not len(faces):
            continue
        projected, camera_depth = _project_points(
            vertices, crop, camera_height, width, height
        )
        for face in faces:
            face_depths = camera_depth[face]
            if np.any(face_depths <= 0.1):
                continue
            triangle = projected[face]
            denominator = (
                (triangle[1, 1] - triangle[2, 1])
                * (triangle[0, 0] - triangle[2, 0])
                + (triangle[2, 0] - triangle[1, 0])
                * (triangle[0, 1] - triangle[2, 1])
            )
            if abs(denominator) < 1e-8:
                continue
            x_min = max(0, int(math.floor(np.min(triangle[:, 0]))))
            x_max = min(width - 1, int(math.ceil(np.max(triangle[:, 0]))))
            y_min = max(0, int(math.floor(np.min(triangle[:, 1]))))
            y_max = min(height - 1, int(math.ceil(np.max(triangle[:, 1]))))
            if x_min > x_max or y_min > y_max:
                continue
            grid_x, grid_y = np.meshgrid(
                np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
                np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
            )
            weight_0 = (
                (triangle[1, 1] - triangle[2, 1])
                * (grid_x - triangle[2, 0])
                + (triangle[2, 0] - triangle[1, 0])
                * (grid_y - triangle[2, 1])
            ) / denominator
            weight_1 = (
                (triangle[2, 1] - triangle[0, 1])
                * (grid_x - triangle[2, 0])
                + (triangle[0, 0] - triangle[2, 0])
                * (grid_y - triangle[2, 1])
            ) / denominator
            weight_2 = 1.0 - weight_0 - weight_1
            inside = (
                (weight_0 >= -1e-8)
                & (weight_1 >= -1e-8)
                & (weight_2 >= -1e-8)
            )
            reciprocal_depth = (
                weight_0 / face_depths[0]
                + weight_1 / face_depths[1]
                + weight_2 / face_depths[2]
            )
            triangle_depth = 1.0 / np.maximum(reciprocal_depth, 1e-12)
            depth_view = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
            nearer = inside & (triangle_depth < depth_view)
            if not np.any(nearer):
                continue
            perspective_weights = np.stack((
                weight_0 / face_depths[0],
                weight_1 / face_depths[1],
                weight_2 / face_depths[2],
            ), axis=-1)
            perspective_weights /= np.maximum(
                reciprocal_depth[..., None], 1e-12
            )
            hit_points = np.einsum(
                "ijk,kl->ijl", perspective_weights, vertices[face]
            )
            depth_view[nearer] = triangle_depth[nearer]
            surface_points_view = surface_points[
                y_min:y_max + 1, x_min:x_max + 1
            ]
            surface_points_view[nearer] = hit_points[nearer].astype(np.float32)

    return depth_buffer, surface_points


def project_masks_to_buildings(
    predictions: list[dict[str, Any]],
    masks: list[np.ndarray],
    meshes: list[dict[str, Any]],
    crop: dict[str, Any],
    camera_height: float,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project SAM mask pixels onto nearest visible building mesh surfaces."""
    if len(predictions) != len(masks):
        raise ValueError("Each DINO prediction must have exactly one SAM mask.")
    image_width, image_height = image_size
    if (
        int(crop["out_w"]) != image_width
        or int(crop["out_h"]) != image_height
    ):
        raise ValueError(
            "Crop manifest dimensions do not match the inference image: "
            f"{crop['out_w']}x{crop['out_h']} versus "
            f"{image_width}x{image_height}."
        )
    width = min(image_width, PROJECTION_MAX_WIDTH)
    height = max(1, round(image_height * width / image_width))
    projection_crop = {**crop, "out_w": width, "out_h": height}
    depth_buffer, surface_points = _rasterize_building_surfaces(
        meshes,
        projection_crop,
        camera_height,
        width,
        height,
    )
    visible_surface = np.isfinite(depth_buffer)
    projected_meshes = []
    projection_records = []

    for index, (prediction, mask) in enumerate(zip(predictions, masks), start=1):
        if mask.shape != (image_height, image_width):
            raise ValueError(
                f"SAM mask {index} has dimensions {mask.shape}; expected "
                f"{(image_height, image_width)}."
            )
        scaled_mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.Resampling.NEAREST
            )
        ) > 0
        selected = scaled_mask & visible_surface
        row_indices, column_indices = np.nonzero(selected)
        if not len(row_indices):
            projection_records.append({
                "index": index,
                "projected_pixel_count": 0,
                "mesh_triangle_count": 0,
            })
            continue

        vertex_map = np.full((height, width), -1, dtype=np.int64)
        vertex_map[row_indices, column_indices] = np.arange(
            len(row_indices), dtype=np.int64
        )
        mesh_vertices = surface_points[row_indices, column_indices].astype(
            np.float64
        )
        toward_camera = np.array([0.0, 0.0, camera_height]) - mesh_vertices
        toward_camera /= np.maximum(
            np.linalg.norm(toward_camera, axis=1, keepdims=True), 1e-12
        )
        mesh_vertices += SURFACE_OVERLAY_OFFSET_M * toward_camera
        cell_valid = (
            (vertex_map[:-1, :-1] >= 0)
            & (vertex_map[:-1, 1:] >= 0)
            & (vertex_map[1:, 1:] >= 0)
            & (vertex_map[1:, :-1] >= 0)
        )
        depth_cells = np.stack((
            depth_buffer[:-1, :-1],
            depth_buffer[:-1, 1:],
            depth_buffer[1:, 1:],
            depth_buffer[1:, :-1],
        ), axis=-1)
        finite_depth_cells = np.where(np.isfinite(depth_cells), depth_cells, 0.0)
        cell_valid &= (
            np.max(finite_depth_cells, axis=-1)
            - np.min(finite_depth_cells, axis=-1)
            <= PROJECTION_DEPTH_DISCONTINUITY_M
        )
        corners = (
            vertex_map[:-1, :-1],
            vertex_map[:-1, 1:],
            vertex_map[1:, 1:],
            vertex_map[1:, :-1],
        )
        a, b, c, d = (corner[cell_valid] for corner in corners)
        if len(a):
            faces = np.concatenate((
                np.column_stack((a, b, c)),
                np.column_stack((a, c, d)),
            )).astype(np.int64)
        else:
            faces = np.empty((0, 3), dtype=np.int64)
        metadata = {
            "detection_index": index,
            "score": float(prediction["score"]),
            "bbox_xywh_pixels": [
                float(value) for value in prediction["bbox"]
            ],
            "projected_pixel_count": len(row_indices),
            "mesh_triangle_count": len(faces),
        }
        projection_records.append(metadata)
        if len(faces):
            projected_meshes.append({
                "name": f"Balcony segmentation {index} "
                f"(score {prediction['score']:.2f})",
                "layer": "balcony_segmentation",
                "vertices": [
                    (float(x), float(y), float(z))
                    for x, y, z in mesh_vertices
                ],
                "faces": [
                    (int(a), int(b), int(c))
                    for a, b, c in faces
                ],
                "color": BALCONY_COLORS[(index - 1) % len(BALCONY_COLORS)],
                "metadata": metadata,
            })
    return projected_meshes, projection_records
