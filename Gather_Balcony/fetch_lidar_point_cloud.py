#!/usr/bin/env python3
"""Find USGS 3DEP LiDAR tiles and build class-specific vegetation meshes."""

from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import Polygon
from shapely.ops import unary_union


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_LPC_DATASET = "Lidar Point Cloud (LPC)"
USER_AGENT = "VIP-6600-Solar-3DEP-point-cloud/1.0"
CHUNK_SIZE = 250_000
MAX_GROUND_SAMPLES_PER_TILE = 50_000
DEFAULT_MAX_VISUALIZATION_POINTS_PER_CLASS = 2_500
VEGETATION_CLUSTER_CELL_SIZE_METERS = 100.0
VEGETATION_CLUSTER_CELL_BUFFER_METERS = 25.0
MAX_CLUSTERING_POINTS_PER_CELL_CLASS = 5_000
MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS = 3_000
MIN_VEGETATION_HEIGHT_METERS = 0.5
DEFAULT_SHADE_RELEVANT_MIN_HEIGHT_METERS = 2.0
MAX_CANOPY_HEIGHT_METERS = 60.0
TREE_LAS_CLASSIFICATION = 5
MIN_VEGETATION_CLUSTER_POINTS = 20
CANOPY_RASTER_CELL_METERS = 1.0
CANOPY_PEAK_NEIGHBORHOOD_CELLS = 5
MIN_TREE_PEAK_SEPARATION_METERS = 3.0
MAX_TREE_CROWN_ASSIGNMENT_RADIUS_METERS = 15.0
FOLIAGE_GRID_LATITUDE_SEGMENTS = 8
FOLIAGE_GRID_LONGITUDE_SEGMENTS = 16
MIN_INFERRED_CROWN_HEIGHT_RATIO = 0.65
MAX_INFERRED_CROWN_HEIGHT_RATIO = 1.5
FOLIAGE_DENSITY_BY_CLASS = {
    3: 0.25,
    4: 0.35,
    5: 0.50,
}
MIN_CROWN_RADIUS_METERS = 0.75
MIN_CROWN_HEIGHT_METERS = 2.0
POINT_CLASSES = {
    "ground": (2,),
    "low_vegetation": (3,),
    "medium_vegetation": (4,),
    "high_vegetation": (5,),
    "building": (6,),
    "other": (),
}
POINT_CLASS_COLORS = {
    "ground": "#8d6e63",
    "low_vegetation": "#c0ca33",
    "medium_vegetation": "#66bb6a",
    "high_vegetation": "#1b5e20",
    "building": "#ef5350",
    "other": "#78909c",
}


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60.0) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise RuntimeError("USGS TNM returned an unexpected response.")
            return result
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(
                    f"USGS TNM point-cloud search failed with HTTP "
                    f"{error.code}: {error.reason}"
                ) from error
        except URLError as error:
            last_error = error
            if attempt == 2:
                raise RuntimeError(
                    f"Could not reach USGS TNM point-cloud search: "
                    f"{error.reason}"
                ) from error
        time.sleep(2**attempt)
    raise RuntimeError(f"USGS TNM point-cloud search failed: {last_error}")


def _download_tile(
    url: str,
    output_path: Path,
    expected_size: int | None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not (
        parsed.hostname == "rockyweb.usgs.gov"
        or parsed.hostname == "prd-tnm.s3.amazonaws.com"
    ):
        raise RuntimeError(
            f"USGS TNM returned an unexpected point-cloud host: "
            f"{parsed.hostname}"
        )
    temporary_path = output_path.with_name(output_path.name + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=180.0) as response:
                with temporary_path.open("wb") as file:
                    shutil.copyfileobj(response, file)
            actual_size = temporary_path.stat().st_size
            if actual_size == 0 or (
                expected_size is not None and actual_size != expected_size
            ):
                raise RuntimeError(
                    f"Downloaded USGS tile has {actual_size} bytes; expected "
                    f"{expected_size or 'a nonzero file'}: {url}"
                )
            temporary_path.replace(output_path)
            return
        except HTTPError as error:
            temporary_path.unlink(missing_ok=True)
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                break
            retry_after = error.headers.get("Retry-After")
            delay = 30.0 * (2**attempt) if error.code == 429 else 2**(attempt + 1)
            if retry_after:
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(tzinfo=timezone.utc)
                        delay = max(
                            0.0,
                            (retry_at - datetime.now(timezone.utc)).total_seconds(),
                        )
                    except (TypeError, ValueError, OverflowError):
                        pass
            delay = min(delay, 300.0)
            if error.code == 429:
                print(
                    f"  USGS rate-limited {output_path.name}; "
                    f"retrying in {delay:g} seconds "
                    f"(attempt {attempt + 2}/4)."
                )
        except (URLError, OSError) as error:
            temporary_path.unlink(missing_ok=True)
            last_error = error
            if attempt == 3:
                break
            delay = 2**(attempt + 1)
        except RuntimeError:
            temporary_path.unlink(missing_ok=True)
            raise
        time.sleep(delay)
    raise RuntimeError(
        f"Could not download USGS point-cloud tile {url} after retries: "
        f"{last_error}"
    ) from last_error


def _local_polygon(
    points: list[tuple[float, float]],
    latitude: float,
    longitude: float,
) -> Polygon:
    earth_radius = 6_371_000.0
    return Polygon([
        (
            math.radians(point_lon - longitude)
            * earth_radius
            * math.cos(math.radians(latitude)),
            math.radians(point_lat - latitude) * earth_radius,
        )
        for point_lon, point_lat in points
    ])


def _product_polygon(
    product: dict[str, Any],
    latitude: float,
    longitude: float,
) -> Polygon | None:
    bounds = product.get("boundingBox")
    if not isinstance(bounds, dict):
        return None
    keys = ("minX", "minY", "maxX", "maxY")
    if not all(isinstance(bounds.get(key), (int, float)) for key in keys):
        return None
    return _local_polygon(
        [
            (bounds["minX"], bounds["minY"]),
            (bounds["maxX"], bounds["minY"]),
            (bounds["maxX"], bounds["maxY"]),
            (bounds["minX"], bounds["maxY"]),
        ],
        latitude,
        longitude,
    )


def find_lidar_tiles(
    latitude: float,
    longitude: float,
    aoi_boundary_local: list[tuple[float, float]],
    cache_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Find and cache the newest USGS LPC project tiles intersecting the AOI."""
    from shapely.geometry import mapping

    aoi = Polygon(aoi_boundary_local)
    if aoi.is_empty or not aoi.is_valid:
        raise ValueError("The scene terrain boundary is invalid.")

    geographic_boundary = [
        (
            longitude + math.degrees(
                east / (
                    6_371_000.0 * math.cos(math.radians(latitude))
                )
            ),
            latitude + math.degrees(north / 6_371_000.0),
        )
        for east, north in aoi_boundary_local
    ]
    west = min(point[0] for point in geographic_boundary)
    south = min(point[1] for point in geographic_boundary)
    east = max(point[0] for point in geographic_boundary)
    north = max(point[1] for point in geographic_boundary)
    params = urlencode({
        "datasets": TNM_LPC_DATASET,
        "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
        "max": 1000,
        "outputFormat": "JSON",
    })
    search_url = f"{TNM_PRODUCTS_URL}?{params}"
    response = _request_json(search_url)
    products = response.get("items")
    if not isinstance(products, list):
        raise RuntimeError("USGS TNM response did not contain a product list.")

    intersecting: list[tuple[dict[str, Any], Polygon]] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        polygon = _product_polygon(product, latitude, longitude)
        if polygon is not None and polygon.intersects(aoi):
            intersecting.append((product, polygon))

    if not intersecting:
        return [], {
            "status": "no_coverage",
            "source": "USGS 3DEP Lidar Point Cloud (LPC), via TNM Access API",
            "search_url": search_url,
            "products_returned": len(products),
            "tile_count": 0,
            "coverage_fraction_from_tile_bounds": 0.0,
            "aoi_area_square_meters": aoi.area,
            "tiles": [],
        }

    project_groups: dict[str, list[tuple[dict[str, Any], Polygon]]] = {}
    for product, polygon in intersecting:
        title = str(product.get("title", ""))
        project_name = title.removeprefix("USGS Lidar Point Cloud ").rsplit(
            " ", 1
        )[0]
        project_groups.setdefault(project_name, []).append((product, polygon))

    def project_key(
        item: tuple[str, list[tuple[dict[str, Any], Polygon]]],
    ) -> tuple[str, float]:
        _name, entries = item
        publication = max(
            str(product.get("publicationDate") or "")
            for product, _polygon in entries
        )
        coverage = unary_union([polygon for _product, polygon in entries])
        coverage_fraction = coverage.intersection(aoi).area / aoi.area
        return publication, coverage_fraction

    selected_project, selected_entries = max(
        project_groups.items(), key=project_key
    )
    tile_union = unary_union(
        [polygon for _product, polygon in selected_entries]
    )
    coverage_fraction = min(
        1.0, tile_union.intersection(aoi).area / aoi.area
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    tile_records = []
    for product, polygon in selected_entries:
        url = product.get("downloadLazURL") or product.get("downloadURL")
        if not isinstance(url, str):
            continue
        filename = Path(urlparse(url).path).name
        if not filename.lower().endswith(".laz"):
            continue
        output_path = cache_dir / filename
        expected_size = product.get("sizeInBytes")
        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
            or (
                isinstance(expected_size, int)
                and output_path.stat().st_size != expected_size
            )
        ):
            print(f"  Downloading USGS LiDAR tile {filename}...")
            _download_tile(
                url,
                output_path,
                expected_size if isinstance(expected_size, int) else None,
            )
        else:
            print(f"  Using cached USGS LiDAR tile {filename}.")
        paths.append(output_path)
        tile_records.append({
            "title": product.get("title"),
            "publication_date": product.get("publicationDate"),
            "size_bytes": product.get("sizeInBytes"),
            "download_url": url,
            "cached_file": str(output_path),
            "tile_aoi_intersection_fraction": min(
                1.0, polygon.intersection(aoi).area / aoi.area
            ),
        })

    return paths, {
        "status": "tiles_found",
        "source": "USGS 3DEP Lidar Point Cloud (LPC), via TNM Access API",
        "search_url": search_url,
        "products_returned": len(products),
        "selected_project": selected_project,
        "publication_date": project_key(
            (selected_project, selected_entries)
        )[0],
        "tile_count": len(paths),
        "coverage_fraction_from_tile_bounds": coverage_fraction,
        "aoi_area_square_meters": aoi.area,
        "tiles": tile_records,
        "coverage_geometry_local": mapping(tile_union.intersection(aoi)),
        "coverage_note": (
            "Coverage is estimated from catalog tile bounding boxes; actual "
            "classified point density is measured during processing."
        ),
    }


def _las_points_in_dem(
    points: Any,
    transformer: Transformer,
    dataset: rasterio.io.DatasetReader,
    dem: np.ndarray,
    dem_invalid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x, y = transformer.transform(points.x, points.y)
    x = np.asarray(x)
    y = np.asarray(y)
    rows, cols = rowcol(dataset.transform, x, y)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    in_bounds = (
        (rows >= 0)
        & (rows < dataset.height)
        & (cols >= 0)
        & (cols < dataset.width)
    )
    valid = np.flatnonzero(in_bounds)
    if not len(valid):
        empty = np.array([], dtype=np.float64)
        return (
            empty,
            empty,
            empty,
            np.array([], dtype=np.uint8),
            empty,
            np.array([], dtype=np.int64),
        )
    rows_valid = rows[valid]
    cols_valid = cols[valid]
    dem_valid = ~dem_invalid[rows_valid, cols_valid]
    valid = valid[dem_valid]
    rows_valid = rows[valid]
    cols_valid = cols[valid]
    ground = dem[rows_valid, cols_valid]
    z = np.asarray(points.z)[valid]
    classification = np.asarray(points.classification)[valid]
    return x[valid], y[valid], z, classification, ground, valid


def _point_class_masks(classification: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        name: (
            np.isin(classification, codes)
            if codes else ~np.isin(classification, (2, 3, 4, 5, 6))
        )
        for name, codes in POINT_CLASSES.items()
    }
    return masks


def _update_visualization_sample(
    samples: dict[str, np.ndarray],
    sample_keys: dict[str, np.ndarray],
    class_name: str,
    points: np.ndarray,
    maximum_points: int,
    random_generator: np.random.Generator,
) -> None:
    if maximum_points <= 0 or not len(points):
        return
    keys = random_generator.random(len(points))
    if class_name in samples:
        points = np.concatenate((samples[class_name], points), axis=0)
        keys = np.concatenate((sample_keys[class_name], keys))
    if len(points) > maximum_points:
        selected = np.argpartition(keys, maximum_points - 1)[:maximum_points]
        points = points[selected]
        keys = keys[selected]
    samples[class_name] = points
    sample_keys[class_name] = keys


def _porous_crown_surface(
    points_xyz: np.ndarray,
    foliage_density: float,
    random_seed: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Build a porous ellipsoidal crown sized from a LiDAR cluster."""
    if len(points_xyz) < MIN_VEGETATION_CLUSTER_POINTS:
        return [], []

    random_generator = np.random.default_rng(random_seed)
    lower = np.percentile(points_xyz, 5, axis=0)
    upper = np.percentile(points_xyz, 95, axis=0)
    horizontal_radii = np.maximum(
        (upper[:2] - lower[:2]) / 2.0,
        MIN_CROWN_RADIUS_METERS,
    )
    measured_height = max(float(upper[2] - lower[2]), 0.0)
    horizontal_diameter = 2.0 * float(np.max(horizontal_radii))
    inferred_height = min(
        MAX_INFERRED_CROWN_HEIGHT_RATIO * horizontal_diameter,
        max(
            measured_height,
            MIN_CROWN_HEIGHT_METERS,
            MIN_INFERRED_CROWN_HEIGHT_RATIO * horizontal_diameter,
        ),
    )
    crown_top = float(upper[2])
    crown_base = crown_top - inferred_height
    center = np.array((
        (lower[0] + upper[0]) / 2.0,
        (lower[1] + upper[1]) / 2.0,
        (crown_base + crown_top) / 2.0,
    ))
    radii = np.array((
        horizontal_radii[0],
        horizontal_radii[1],
        inferred_height / 2.0,
    ))
    foliage_density = min(max(foliage_density, 0.05), 0.95)

    vertices: list[tuple[float, float, float]] = []
    for latitude_index in range(FOLIAGE_GRID_LATITUDE_SEGMENTS + 1):
        normalized_height = (
            latitude_index / FOLIAGE_GRID_LATITUDE_SEGMENTS
        )
        phi = -math.pi / 2 + math.pi * normalized_height
        shape_factor = math.sin(math.pi * normalized_height) ** 0.55
        for longitude_index in range(FOLIAGE_GRID_LONGITUDE_SEGMENTS):
            theta = 2 * math.pi * (
                longitude_index / FOLIAGE_GRID_LONGITUDE_SEGMENTS
            )
            vertices.append((
                float(center[0] + radii[0] * shape_factor * math.cos(theta)),
                float(center[1] + radii[1] * shape_factor * math.sin(theta)),
                float(crown_base + inferred_height * normalized_height),
            ))

    faces: list[tuple[int, int, int]] = []
    for latitude_index in range(FOLIAGE_GRID_LATITUDE_SEGMENTS):
        for longitude_index in range(FOLIAGE_GRID_LONGITUDE_SEGMENTS):
            keep = random_generator.random() < foliage_density
            if not keep:
                continue
            next_longitude = (
                (longitude_index + 1) % FOLIAGE_GRID_LONGITUDE_SEGMENTS
            )
            first = latitude_index * FOLIAGE_GRID_LONGITUDE_SEGMENTS
            second = (latitude_index + 1) * FOLIAGE_GRID_LONGITUDE_SEGMENTS
            a = first + longitude_index
            b = first + next_longitude
            c = second + next_longitude
            d = second + longitude_index
            faces.extend(((a, b, c), (a, c, d)))

    if not faces:
        return [], []
    return vertices, faces


def _canopy_peak_clusters(
    points_xyz: np.ndarray,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    """Split vegetation returns around local maxima in a canopy-height grid."""
    from scipy.spatial import cKDTree

    grid_coordinates = np.floor(
        points_xyz[:, :2] / CANOPY_RASTER_CELL_METERS
    ).astype(np.int64)
    maximum_by_cell: dict[tuple[int, int], float] = {}
    for cell, elevation in zip(grid_coordinates, points_xyz[:, 2]):
        key = (int(cell[0]), int(cell[1]))
        maximum_by_cell[key] = max(
            maximum_by_cell.get(key, -math.inf),
            float(elevation),
        )

    candidates = []
    neighborhood_radius = CANOPY_PEAK_NEIGHBORHOOD_CELLS // 2
    for (east, north), elevation in maximum_by_cell.items():
        is_peak = all(
            elevation >= maximum_by_cell.get(
                (east + east_delta, north + north_delta),
                -math.inf,
            )
            for east_delta in range(-neighborhood_radius, neighborhood_radius + 1)
            for north_delta in range(
                -neighborhood_radius, neighborhood_radius + 1
            )
        )
        if is_peak:
            candidates.append((elevation, east, north))
    candidates.sort(reverse=True)

    peaks: list[tuple[float, float, float]] = []
    for elevation, east, north in candidates:
        point = np.array([
            (east + 0.5) * CANOPY_RASTER_CELL_METERS,
            (north + 0.5) * CANOPY_RASTER_CELL_METERS,
            elevation,
        ])
        if all(
            np.linalg.norm(point[:2] - np.asarray(existing[:2]))
            >= MIN_TREE_PEAK_SEPARATION_METERS
            for existing in peaks
        ):
            peaks.append(tuple(point))
    if not peaks:
        return [], 0

    peak_xy = np.asarray(peaks)[:, :2]
    _distances, nearest = cKDTree(peak_xy).query(
        points_xyz[:, :2],
        distance_upper_bound=MAX_TREE_CROWN_ASSIGNMENT_RADIUS_METERS,
    )
    clusters = []
    for peak_index, peak in enumerate(peaks):
        member_indices = np.flatnonzero(nearest == peak_index)
        if len(member_indices) >= MIN_VEGETATION_CLUSTER_POINTS:
            clusters.append((peak_index, points_xyz[member_indices]))
    return clusters, len(peaks)


def _vegetation_meshes(
    points_by_class: dict[int, list[np.ndarray]],
    raw_point_counts: dict[int, int] | None = None,
    core_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect canopy peaks and mesh individual crown regions."""
    class_meshes = []
    class_metadata = {}
    for classification in (3, 4, 5):
        class_points = points_by_class[classification]
        raw_point_count = (
            raw_point_counts.get(classification, 0)
            if raw_point_counts is not None else 0
        )
        if not class_points:
            class_metadata[str(classification)] = {
                "input_point_count": raw_point_count,
                "clustering_sample_point_count": 0,
                "clustering_point_limit_per_cell": (
                    MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                ),
                "cluster_count": 0,
                "meshed_cluster_count": 0,
                "canopy_peak_count": 0,
            }
            continue

        xyz = np.concatenate(class_points)
        if len(xyz) < MIN_VEGETATION_CLUSTER_POINTS:
            class_metadata[str(classification)] = {
                "input_point_count": raw_point_count or len(xyz),
                "clustering_sample_point_count": len(xyz),
                "clustering_point_limit_per_cell": (
                    MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                ),
                "points_subsampled": len(xyz) < raw_point_count,
                "cluster_count": 0,
                "meshed_cluster_count": 0,
                "canopy_peak_count": 0,
            }
            continue

        clusters, peak_count = _canopy_peak_clusters(xyz)

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        cluster_records = []
        mesh_failures = []
        for label, cluster_points in clusters:
            point_count = len(cluster_points)
            if core_bounds is not None:
                center_east, center_north = np.mean(
                    cluster_points[:, :2], axis=0
                )
                if not (
                    core_bounds[0] <= center_east < core_bounds[2]
                    and core_bounds[1] <= center_north < core_bounds[3]
                ):
                    continue
            cluster_vertices, cluster_faces = _porous_crown_surface(
                cluster_points,
                FOLIAGE_DENSITY_BY_CLASS[classification],
                random_seed=classification * 1_000_003 + int(label),
            )
            if not cluster_faces:
                mesh_failures.append({
                    "cluster": int(label),
                    "point_count": int(point_count),
                    "reason": "No porous crown surface was produced.",
                })
                continue
            vertex_offset = len(vertices)
            vertices.extend(cluster_vertices)
            faces.extend(
                (a + vertex_offset, b + vertex_offset, c + vertex_offset)
                for a, b, c in cluster_faces
            )
            cluster_records.append({
                "cluster_id": int(label),
                "point_count": int(point_count),
                "mesh_vertex_count": len(cluster_vertices),
                "mesh_triangle_count": len(cluster_faces),
                "mesh_input_point_count": int(point_count),
                "crown_lower_percentile": 5,
                "crown_upper_percentile": 95,
                "observed_vertical_return_thickness_m": float(
                    max(cluster_points[:, 2]) -
                    min(cluster_points[:, 2])
                ),
                "crown_shape": "tapered layered crown",
                "foliage_density": FOLIAGE_DENSITY_BY_CLASS[classification],
            })

        if vertices and faces:
            class_meshes.append({
                "name": f"3DEP_LiDAR_vegetation_class_{classification}",
                "layer": "trees",
                "vertices": vertices,
                "faces": faces,
                "color": {
                    3: "#c0ca33",
                    4: "#66bb6a",
                    5: "#1b5e20",
                }[classification],
                "metadata": {
                    "las_classification": classification,
                    "vegetation_class": {
                        3: "low",
                        4: "medium",
                        5: "high",
                    }[classification],
                    "cluster_count": len(cluster_records),
                    "clusters": cluster_records,
                    "canopy_peak_count": peak_count,
                    "canopy_raster_cell_m": CANOPY_RASTER_CELL_METERS,
                    "peak_neighborhood_cells": (
                        CANOPY_PEAK_NEIGHBORHOOD_CELLS
                    ),
                    "crown_model": "LiDAR-sized porous ellipsoid",
                    "foliage_density": FOLIAGE_DENSITY_BY_CLASS[classification],
                    "crown_percentiles": [5, 95],
                    "minimum_inferred_height_ratio": (
                        MIN_INFERRED_CROWN_HEIGHT_RATIO
                    ),
                    "maximum_inferred_height_ratio": (
                        MAX_INFERRED_CROWN_HEIGHT_RATIO
                    ),
                    "grid_latitude_segments": FOLIAGE_GRID_LATITUDE_SEGMENTS,
                    "grid_longitude_segments": FOLIAGE_GRID_LONGITUDE_SEGMENTS,
                },
            })
        class_metadata[str(classification)] = {
            "input_point_count": raw_point_count or len(xyz),
            "clustering_sample_point_count": len(xyz),
            "clustering_point_limit_per_cell": (
                MAX_CLUSTERING_POINTS_PER_CELL_CLASS
            ),
            "points_subsampled": len(xyz) < raw_point_count,
            "cluster_count": len(clusters),
            "meshed_cluster_count": len(cluster_records),
            "canopy_peak_count": peak_count,
            "canopy_raster_cell_m": CANOPY_RASTER_CELL_METERS,
            "mesh_failures": mesh_failures,
        }
    return class_meshes, class_metadata


def create_canopy_mesh(
    tile_paths: list[Path],
    raster_path: Path,
    latitude: float,
    longitude: float,
    point_output_dir: Path,
    maximum_visualization_points_per_class: int = (
        DEFAULT_MAX_VISUALIZATION_POINTS_PER_CLASS
    ),
    minimum_shade_relevant_height_m: float = (
        DEFAULT_SHADE_RELEVANT_MIN_HEIGHT_METERS
    ),
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[list[float]]]]:
    """Create vegetation-cluster meshes and export classified points."""
    try:
        import laspy
    except ImportError as error:
        raise RuntimeError(
            "USGS LAZ processing requires laspy and lazrs. Install project "
            "requirements with `python -m pip install -r ../requirements.txt`."
        ) from error

    if not tile_paths:
        return [], {
            "status": "no_tiles",
            "vegetation_return_count": 0,
            "vegetation_cluster_count": 0,
            "mesh_triangle_count": 0,
            "vegetation_classes": {},
            "classified_points": {
                "status": "no_tiles",
                "counts": {name: 0 for name in POINT_CLASSES},
                "files": [],
                "clip_definition": (
                    "Points mapping to valid pixels inside the downloaded "
                    "terrain raster."
                ),
            },
        }, {}
    if maximum_visualization_points_per_class < 0:
        raise ValueError(
            "Maximum visualization points per class cannot be negative."
        )
    if (
        not math.isfinite(minimum_shade_relevant_height_m)
        or minimum_shade_relevant_height_m < MIN_VEGETATION_HEIGHT_METERS
        or minimum_shade_relevant_height_m > MAX_CANOPY_HEIGHT_METERS
    ):
        raise ValueError(
            "Minimum shade-relevant height must be finite and between "
            f"{MIN_VEGETATION_HEIGHT_METERS:g} and "
            f"{MAX_CANOPY_HEIGHT_METERS:g} metres."
        )

    with rasterio.open(raster_path) as dataset:
        if dataset.crs is None:
            raise ValueError("The DEM raster has no CRS.")
        dem_masked = dataset.read(1, masked=True)
        dem = np.asarray(np.ma.filled(dem_masked, np.nan), dtype=np.float64)
        dem_invalid = np.ma.getmaskarray(dem_masked) | ~np.isfinite(dem)
        origin_x, origin_y = transform_coordinates(
            "EPSG:4326", dataset.crs, [longitude], [latitude]
        )
        origin_row, origin_col = dataset.index(origin_x[0], origin_y[0])
        if (
            not (0 <= origin_row < dataset.height and 0 <= origin_col < dataset.width)
            or dem_invalid[origin_row, origin_col]
        ):
            raise ValueError(
                "The crop location has no valid elevation in the DEM used to "
                "normalize the LiDAR canopy heights."
            )
        origin_elevation = dem[origin_row, origin_col]
        vegetation_return_count = 0
        class_counts = {str(code): 0 for code in (3, 4, 5)}
        vegetation_cell_points: dict[
            tuple[int, int, int], np.ndarray
        ] = {}
        vegetation_cell_keys: dict[
            tuple[int, int, int], np.ndarray
        ] = {}
        vegetation_point_counts = {code: 0 for code in (3, 4, 5)}
        vegetation_random_generator = np.random.default_rng(20261029)
        tile_stats = []
        classified_point_counts = {name: 0 for name in POINT_CLASSES}
        point_file_records: list[dict[str, Any]] = []
        visualization_samples: dict[str, np.ndarray] = {}
        visualization_sample_keys: dict[str, np.ndarray] = {}
        random_generator = np.random.default_rng(20261028)
        all_ground_differences: list[np.ndarray] = []
        tile_transformers: list[tuple[Path, Transformer]] = []
        point_output_dir.mkdir(parents=True, exist_ok=True)

        for tile_index, tile_path in enumerate(tile_paths):
            print(f"  Reading ground returns from {tile_path.name}...")
            with laspy.open(tile_path) as reader:
                source_crs = reader.header.parse_crs()
                if source_crs is None:
                    raise RuntimeError(
                        f"USGS point-cloud tile has no coordinate reference "
                        f"system: {tile_path.name}"
                    )
                transformer = Transformer.from_crs(
                    source_crs, dataset.crs, always_xy=True
                )
                point_count = 0
                sample_differences = []
                with ExitStack() as writer_stack:
                    tile_writers: dict[str, dict[str, Any]] = {}
                    for chunk in reader.chunk_iterator(CHUNK_SIZE):
                        point_count += len(chunk)
                        x, y, z, classification, ground, source_indices = (
                            _las_points_in_dem(
                                chunk, transformer, dataset, dem, dem_invalid
                            )
                        )
                        masks = _point_class_masks(classification)
                        for class_name, mask in masks.items():
                            selected_indices = source_indices[mask]
                            if not len(selected_indices):
                                continue
                            if class_name not in tile_writers:
                                class_dir = point_output_dir / class_name
                                class_dir.mkdir(parents=True, exist_ok=True)
                                output_path = class_dir / tile_path.name
                                record = {
                                    "class": class_name,
                                    "file": str(output_path),
                                    "source_tile": tile_path.name,
                                    "point_count": 0,
                                }
                                tile_writers[class_name] = {
                                    "writer": writer_stack.enter_context(
                                        laspy.open(
                                            output_path,
                                            mode="w",
                                            header=reader.header.copy(),
                                            do_compress=True,
                                        )
                                    ),
                                    "record": record,
                                }
                                point_file_records.append(record)
                            tile_writers[class_name]["writer"].write_points(
                                chunk[selected_indices]
                            )
                            tile_writers[class_name]["record"]["point_count"] += (
                                len(selected_indices)
                            )
                            classified_point_counts[class_name] += len(
                                selected_indices
                            )

                            if maximum_visualization_points_per_class <= 0:
                                continue

                        ground_points = np.flatnonzero(classification == 2)
                        if len(ground_points):
                            remaining = (
                                MAX_GROUND_SAMPLES_PER_TILE
                                - sum(len(sample) for sample in sample_differences)
                            )
                            if remaining > 0:
                                stride = max(
                                    1,
                                    math.ceil(len(ground_points) / remaining),
                                )
                                chosen = ground_points[::stride][:remaining]
                                sample_differences.append(
                                    z[chosen] - ground[chosen]
                                )
                if sample_differences:
                    differences = np.concatenate(sample_differences)
                else:
                    differences = np.array([], dtype=np.float64)
                all_ground_differences.append(differences)
                tile_transformers.append((tile_path, transformer))
                tile_stats.append({
                    "file": tile_path.name,
                    "point_count": point_count,
                    "coordinate_crs": source_crs.to_string(),
                    "ground_samples_for_vertical_alignment": len(differences),
                })

        usable_differences = [
            difference for difference in all_ground_differences
            if len(difference)
        ]
        if not usable_differences:
            return [], {
                "status": "no_classified_ground_returns",
                "tile_count": len(tile_paths),
                "tiles": tile_stats,
                "vegetation_return_count": 0,
                "vegetation_cluster_count": 0,
                "mesh_triangle_count": 0,
                "note": (
                    "Cannot align point-cloud elevations to the bare-earth "
                    "DEM without class-2 ground returns."
                ),
                "classified_points": {
                    "status": "exported_without_vertical_alignment",
                    "counts": classified_point_counts,
                    "files": point_file_records,
                    "visualization_samples_per_class": {
                        name: 0 for name in POINT_CLASSES
                    },
                    "clip_definition": (
                        "Points mapping to valid pixels inside the downloaded "
                        "terrain raster."
                    ),
                    "note": (
                        "Class-separated LAZ files are available, but point "
                        "samples are omitted from the scene view because "
                        "class-2 returns were unavailable for vertical "
                        "alignment."
                    ),
                },
            }, {}
        all_offsets = np.concatenate(usable_differences)
        global_vertical_offset = float(np.median(all_offsets))
        vertical_offsets = [
            (
                float(np.median(difference))
                if len(difference) else global_vertical_offset
            )
            for difference in all_ground_differences
        ]
        for tile_index, tile_stat in enumerate(tile_stats):
            tile_stat["vertical_alignment_offset_m"] = vertical_offsets[tile_index]
            tile_stat["used_project_median_fallback"] = not bool(
                len(all_ground_differences[tile_index])
            )
        visualization_points = {}
        for class_name, samples in visualization_samples.items():
            aligned_samples = samples.copy()
            sample_tile_indices = aligned_samples[:, 3].astype(np.int64)
            aligned_samples[:, 2] -= (
                np.asarray(vertical_offsets)[sample_tile_indices]
                + origin_elevation
            )
            visualization_points[class_name] = (
                aligned_samples[:, :3].tolist()
            )

        for tile_index, (tile_path, transformer) in enumerate(tile_transformers):
            tile_offset = vertical_offsets[tile_index]
            print(f"  Collecting vegetation classes from {tile_path.name}...")
            with laspy.open(tile_path) as reader:
                for chunk in reader.chunk_iterator(CHUNK_SIZE):
                    x, y, z, classification, ground, _source_indices = (
                        _las_points_in_dem(
                            chunk, transformer, dataset, dem, dem_invalid
                        )
                    )
                    if not len(z):
                        continue
                    height_above_ground = z - tile_offset - ground
                    vegetation = (
                        np.isin(classification, (3, 4, 5))
                        & (
                            height_above_ground
                            >= minimum_shade_relevant_height_m
                        )
                        & (height_above_ground <= MAX_CANOPY_HEIGHT_METERS)
                    )
                    if not np.any(vegetation):
                        continue
                    if maximum_visualization_points_per_class > 0:
                        point_lons, point_lats = transform_coordinates(
                            dataset.crs,
                            "EPSG:4326",
                            x[vegetation].tolist(),
                            y[vegetation].tolist(),
                        )
                        local_east = (
                            np.radians(np.asarray(point_lons) - longitude)
                            * 6_371_000.0
                            * math.cos(math.radians(latitude))
                        )
                        local_north = (
                            np.radians(np.asarray(point_lats) - latitude)
                            * 6_371_000.0
                        )
                        for code, class_name in (
                            (3, "low_vegetation"),
                            (4, "medium_vegetation"),
                            (5, "high_vegetation"),
                        ):
                            class_mask = (
                                vegetation
                                & (classification == code)
                            )
                            if not np.any(class_mask):
                                continue
                            samples = np.column_stack((
                                local_east[class_mask[vegetation]],
                                local_north[class_mask[vegetation]],
                                z[class_mask],
                                np.full(np.count_nonzero(class_mask), tile_index),
                            ))
                            _update_visualization_sample(
                                visualization_samples,
                                visualization_sample_keys,
                                class_name,
                                samples,
                                maximum_visualization_points_per_class,
                                random_generator,
                            )
                    veg_x = x[vegetation]
                    veg_y = y[vegetation]
                    veg_z = z[vegetation]
                    veg_classifications = classification[vegetation]
                    point_lons, point_lats = transform_coordinates(
                        dataset.crs,
                        "EPSG:4326",
                        veg_x.tolist(),
                        veg_y.tolist(),
                    )
                    local_east = (
                        np.radians(np.asarray(point_lons) - longitude)
                        * 6_371_000.0
                        * math.cos(math.radians(latitude))
                    )
                    local_north = (
                        np.radians(np.asarray(point_lats) - latitude)
                        * 6_371_000.0
                    )
                    local_z = veg_z - tile_offset - origin_elevation
                    for code in (3, 4, 5):
                        class_mask = veg_classifications == code
                        count = int(np.count_nonzero(class_mask))
                        if not count:
                            continue
                        class_counts[str(code)] += count
                        vegetation_point_counts[code] += count
                        chunk_points = np.column_stack((
                            local_east[class_mask],
                            local_north[class_mask],
                            local_z[class_mask],
                        ))
                        cell_coordinates = np.floor(
                            chunk_points[:, :2]
                            / VEGETATION_CLUSTER_CELL_SIZE_METERS
                        ).astype(np.int64)
                        unique_cells, inverse_cells = np.unique(
                            cell_coordinates,
                            axis=0,
                            return_inverse=True,
                        )
                        for cell_index, (cell_east, cell_north) in enumerate(
                            unique_cells
                        ):
                            cell_mask = inverse_cells == cell_index
                            cell_points = chunk_points[cell_mask]
                            cell_key = (
                                code,
                                int(cell_east),
                                int(cell_north),
                            )
                            candidate_keys = (
                                vegetation_random_generator.random(
                                    len(cell_points)
                                )
                            )
                            existing_points = vegetation_cell_points.get(
                                cell_key,
                                np.empty((0, 3), dtype=np.float64),
                            )
                            combined_points = np.concatenate(
                                (existing_points, cell_points), axis=0
                            )
                            combined_keys = np.concatenate((
                                vegetation_cell_keys.get(
                                    cell_key,
                                    np.array([], dtype=np.float64),
                                ),
                                candidate_keys,
                            ))
                            if (
                                len(combined_points)
                                > MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                            ):
                                retained_indices = np.argpartition(
                                    combined_keys,
                                    -MAX_CLUSTERING_POINTS_PER_CELL_CLASS,
                                )[-MAX_CLUSTERING_POINTS_PER_CELL_CLASS:]
                                combined_points = combined_points[
                                    retained_indices
                                ]
                                combined_keys = combined_keys[
                                    retained_indices
                                ]
                            vegetation_cell_points[cell_key] = combined_points
                            vegetation_cell_keys[cell_key] = combined_keys
                    vegetation_return_count += len(veg_z)
        cluster_cells = sorted({
            (east_index, north_index)
            for _code, east_index, north_index in vegetation_cell_points
        })
        retained_by_class = {
            code: sum(
                len(points)
                for (point_class, _east_index, _north_index), points
                in vegetation_cell_points.items()
                if point_class == code
            )
            for code in (3, 4, 5)
        }
        print(
            "  Vegetation samples retained for clustering: "
            + ", ".join(
                f"class {code}: {retained_by_class[code]:,}/"
                f"{vegetation_point_counts[code]:,}"
                for code in (3, 4, 5)
            ),
            flush=True,
        )
        print(
            f"  Clustering vegetation in {len(cluster_cells)} bounded "
            "spatial cells...",
            flush=True,
        )
        vegetation_mesh_accumulators: dict[int, dict[str, Any]] = {}
        vegetation_class_metadata = {
            str(code): {
                "input_point_count": vegetation_point_counts[code],
                "clustering_sample_point_count": retained_by_class[code],
                "peak_segmentation_sample_point_count": 0,
                "peak_segmentation_points_omitted_at_fit": 0,
                "processing_cell_count": len(cluster_cells),
                "clustering_point_limit_per_cell": (
                    MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                ),
                "peak_segmentation_point_limit_per_cell": (
                    MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS
                ),
                "points_subsampled": (
                    retained_by_class[code] < vegetation_point_counts[code]
                ),
                "cluster_count": 0,
                "meshed_cluster_count": 0,
                "canopy_peak_count": 0,
                "mesh_failures": [],
            }
            for code in (3, 4, 5)
        }
        for cell_number, (east_index, north_index) in enumerate(cluster_cells, 1):
            core_min_east = (
                east_index * VEGETATION_CLUSTER_CELL_SIZE_METERS
            )
            core_min_north = (
                north_index * VEGETATION_CLUSTER_CELL_SIZE_METERS
            )
            core_bounds = (
                core_min_east,
                core_min_north,
                core_min_east + VEGETATION_CLUSTER_CELL_SIZE_METERS,
                core_min_north + VEGETATION_CLUSTER_CELL_SIZE_METERS,
            )
            search_bounds = (
                core_bounds[0] - VEGETATION_CLUSTER_CELL_BUFFER_METERS,
                core_bounds[1] - VEGETATION_CLUSTER_CELL_BUFFER_METERS,
                core_bounds[2] + VEGETATION_CLUSTER_CELL_BUFFER_METERS,
                core_bounds[3] + VEGETATION_CLUSTER_CELL_BUFFER_METERS,
            )
            points_by_class: dict[int, list[np.ndarray]] = {}
            for code in (3, 4, 5):
                neighboring_points = [
                    vegetation_cell_points[(code, neighbor_east, neighbor_north)]
                    for neighbor_east in range(east_index - 1, east_index + 2)
                    for neighbor_north in range(north_index - 1, north_index + 2)
                    if (code, neighbor_east, neighbor_north)
                    in vegetation_cell_points
                ]
                if not neighboring_points:
                    points_by_class[code] = []
                    continue
                class_xyz = np.concatenate(neighboring_points, axis=0)
                in_buffer = (
                    (class_xyz[:, 0] >= search_bounds[0])
                    & (class_xyz[:, 1] >= search_bounds[1])
                    & (class_xyz[:, 0] <= search_bounds[2])
                    & (class_xyz[:, 1] <= search_bounds[3])
                )
                class_xyz = class_xyz[in_buffer]
                class_details = vegetation_class_metadata[str(code)]
                if len(class_xyz) > MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS:
                    class_details["peak_segmentation_points_omitted_at_fit"] += (
                        len(class_xyz) -
                        MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS
                    )
                    class_xyz = class_xyz[
                        vegetation_random_generator.choice(
                            len(class_xyz),
                            size=MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS,
                            replace=False,
                        )
                    ]
                class_details["peak_segmentation_sample_point_count"] += (
                    len(class_xyz)
                )
                points_by_class[code] = [class_xyz]
            cell_meshes, cell_metadata = _vegetation_meshes(
                points_by_class,
                core_bounds=core_bounds,
            )
            for code in (3, 4, 5):
                class_details = cell_metadata[str(code)]
                total_details = vegetation_class_metadata[str(code)]
                total_details["cluster_count"] += class_details["cluster_count"]
                total_details["meshed_cluster_count"] += (
                    class_details["meshed_cluster_count"]
                )
                if total_details["peak_segmentation_points_omitted_at_fit"]:
                    total_details["points_subsampled"] = True
                total_details["mesh_failures"].extend(
                    class_details.get("mesh_failures", [])
                )
            for cell_mesh in cell_meshes:
                code = int(cell_mesh["metadata"]["las_classification"])
                accumulator = vegetation_mesh_accumulators.setdefault(
                    code,
                    {
                        "vertices": [],
                        "faces": [],
                        "clusters": [],
                    },
                )
                vertex_offset = len(accumulator["vertices"])
                accumulator["vertices"].extend(cell_mesh["vertices"])
                accumulator["faces"].extend(
                    (a + vertex_offset, b + vertex_offset, c + vertex_offset)
                    for a, b, c in cell_mesh["faces"]
                )
                accumulator["clusters"].extend(
                    cell_mesh["metadata"]["clusters"]
                )
            if cell_number % 10 == 0 or cell_number == len(cluster_cells):
                print(
                    f"    Processed vegetation cell {cell_number}/"
                    f"{len(cluster_cells)}.",
                    flush=True,
                )
        vegetation_meshes = []
        for code, accumulator in sorted(
            vegetation_mesh_accumulators.items()
        ):
            vegetation_meshes.append({
                "name": f"3DEP_LiDAR_vegetation_class_{code}",
                "layer": "trees",
                "vertices": accumulator["vertices"],
                "faces": accumulator["faces"],
                "color": {3: "#c0ca33", 4: "#66bb6a", 5: "#1b5e20"}[code],
                "metadata": {
                    "las_classification": code,
                    "vegetation_class": {
                        3: "low",
                        4: "medium",
                        5: "high",
                    }[code],
                    "cluster_count": len(accumulator["clusters"]),
                    "clusters": accumulator["clusters"],
                    "crown_model": "LiDAR-sized porous ellipsoid",
                    "foliage_density": FOLIAGE_DENSITY_BY_CLASS[code],
                    "crown_percentiles": [5, 95],
                    "grid_latitude_segments": FOLIAGE_GRID_LATITUDE_SEGMENTS,
                    "grid_longitude_segments": FOLIAGE_GRID_LONGITUDE_SEGMENTS,
                    "cluster_cell_size_m": (
                        VEGETATION_CLUSTER_CELL_SIZE_METERS
                    ),
                    "cluster_cell_buffer_m": (
                        VEGETATION_CLUSTER_CELL_BUFFER_METERS
                    ),
                    "clustering_point_limit_per_cell": (
                        MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                    ),
                    "peak_segmentation_point_limit_per_cell": (
                        MAX_PEAK_SEGMENTATION_POINTS_PER_CELL_CLASS
                    ),
                },
            })
        mesh_triangle_count = sum(
            len(mesh["faces"]) for mesh in vegetation_meshes
        )
        meshed_cluster_count = sum(
            metadata["meshed_cluster_count"]
            for metadata in vegetation_class_metadata.values()
        )
        classified_points_metadata = {
            "status": "exported",
            "classification_groups": {
                name: list(codes) if codes else "all classes except 2-6"
                for name, codes in POINT_CLASSES.items()
            },
            "counts": classified_point_counts,
            "files": point_file_records,
            "visualization_samples_per_class": {
                name: len(visualization_points.get(name, []))
                for name in POINT_CLASSES
            },
            "maximum_visualization_points_per_class": (
                maximum_visualization_points_per_class
            ),
            "clip_definition": (
                "Points mapping to valid pixels inside the downloaded terrain "
                "raster. LAZ coordinates, elevations, and LAS classification "
                "values are preserved in each output file."
            ),
            "file_coordinate_reference": (
                "Each clipped LAZ remains in its source tile CRS and vertical "
                "datum; scene-relative coordinates are used only for the HTML "
                "samples."
            ),
        }
        return vegetation_meshes, {
            "status": (
                "vegetation_found"
                if meshed_cluster_count else "no_vegetation_clusters"
            ),
            "tile_count": len(tile_paths),
            "tiles": tile_stats,
            "vegetation_classes": class_counts,
            "tree_source_classifications": [3, 4, 5],
            "minimum_shade_relevant_height_m": (
                minimum_shade_relevant_height_m
            ),
            "vegetation_return_count": vegetation_return_count,
            "vegetation_cluster_count": meshed_cluster_count,
            "mesh_triangle_count": mesh_triangle_count,
            "vegetation_class_processing": vegetation_class_metadata,
            "vertical_alignment_offset_per_tile_m": vertical_offsets,
            "vertical_alignment_method": (
                "Median class-2 LAS ground elevation minus DEM elevation; "
                "applied per tile before calculating canopy heights."
            ),
            "z_unit_assumption": (
                "LAS Z coordinates are treated as metres, consistent with the "
                "selected USGS 3DEP project; confirm from tile metadata before "
                "survey-grade use."
            ),
            "vegetation_filter": (
                "LAS classifications 3-5 (low, medium, and high vegetation) "
                "are used for vegetation meshes and point previews. "
                "Short vegetation is excluded from those representations. "
                "Candidate points have DEM-relative height between "
                f"{minimum_shade_relevant_height_m:g} and "
                f"{MAX_CANOPY_HEIGHT_METERS:g} m."
            ),
            "mesh_method": (
                "DBSCAN clusters each vegetation classification separately, "
                "using twice the mean distance to the three nearest neighbors "
                "measured from streamed LAS chunks as eps and min_samples=5. "
                "To bound memory, clustering is spatially partitioned into "
                f"{VEGETATION_CLUSTER_CELL_SIZE_METERS:g} m cells, with a "
                f"{VEGETATION_CLUSTER_CELL_BUFFER_METERS:g} m overlap and at "
                f"most {MAX_CLUSTERING_POINTS_PER_CELL_CLASS:,} sampled points "
                f"per class per cell and {MAX_DBSCAN_POINTS_PER_CELL_CLASS:,} "
                "points per class per buffered DBSCAN fit. Each cluster is "
                "represented by a LiDAR-sized ellipsoidal crown using its "
                "5th-95th percentile extents. Crown surface patches are "
                "deterministically sampled at class-specific foliage "
                "densities, leaving gaps for partial light transmission. "
                "The clusters are separate disconnected components grouped "
                "into one render mesh per vegetation class."
            ),
            "crown_model": {
                "type": "porous_ellipsoid",
                "extent_percentiles": [5, 95],
                "foliage_density_by_las_class": FOLIAGE_DENSITY_BY_CLASS,
                "minimum_radius_m": MIN_CROWN_RADIUS_METERS,
                "minimum_height_m": MIN_CROWN_HEIGHT_METERS,
                "surface_grid": [
                    FOLIAGE_GRID_LATITUDE_SEGMENTS,
                    FOLIAGE_GRID_LONGITUDE_SEGMENTS,
                ],
                "interpretation": (
                    "Density is a geometric foliage-occupancy approximation, "
                    "not a calibrated leaf-area-index measurement."
                ),
            },
            "clustering_memory_limits": {
                "cell_size_m": VEGETATION_CLUSTER_CELL_SIZE_METERS,
                "cell_overlap_m": VEGETATION_CLUSTER_CELL_BUFFER_METERS,
                "max_points_per_class_per_cell": (
                    MAX_CLUSTERING_POINTS_PER_CELL_CLASS
                ),
                "max_dbscan_points_per_class_per_cell": (
                    MAX_DBSCAN_POINTS_PER_CELL_CLASS
                ),
                "point_selection": (
                    "Deterministic-seed reservoir sampling within each "
                    "class and spatial cell."
                ),
            },
            "classified_points": classified_points_metadata,
        }, visualization_points
