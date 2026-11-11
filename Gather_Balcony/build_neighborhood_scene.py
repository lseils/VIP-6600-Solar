#!/usr/bin/env python3
"""Combine 3DEP terrain and Overture buildings into a reviewable scene.

The scene is centered on a selected crop. Horizontal coordinates are local
east/north metres; vertical coordinates are metres above the bare-earth
elevation at the crop location. Exports include an interactive HTML view, an
OBJ mesh for Rhino, clipped class-separated LiDAR LAZ files, and JSON
describing sources and scene layers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as transform_coordinates

if __package__:
    from . import fetch_lidar_elevation
    from . import fetch_lidar_point_cloud
    from . import visualize_building_camera as building_data
else:
    import fetch_lidar_elevation
    import fetch_lidar_point_cloud
    import visualize_building_camera as building_data


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "neighborhood_scenes"
DEFAULT_TERRAIN_MAX_DIMENSION = 250
LIDAR_BUILDING_MATCH_DISTANCE_M = 8.0
MIN_LIDAR_ONLY_BUILDING_HEIGHT_M = 3.0
MIN_LIDAR_ONLY_BUILDING_POINTS = 20
MIN_LIDAR_ONLY_BUILDING_AREA_M2 = 8.0
LIDAR_FOOTPRINT_CONCAVE_RATIO = 0.35
LIDAR_FOOTPRINT_SIMPLIFY_METERS = 0.35
LIDAR_ROOF_ELEVATION_QUANTILE = 0.95
ROOF_LEVEL_MERGE_TOLERANCE_M = 0.5
LAYER_COLORS = {
    "terrain": "#8d9f76",
    "buildings": "#90a4ae",
    "balconies": "#ef6c00",
    "trees": "#2e7d32",
    "other": "#7e57c2",
}


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "mesh"


def lidar_cluster_footprint(cluster_xy: np.ndarray) -> Any:
    """Build a simplified concave outline around a LiDAR roof-point cluster."""
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    footprint = concave_hull(
        MultiPoint(cluster_xy),
        ratio=LIDAR_FOOTPRINT_CONCAVE_RATIO,
        allow_holes=False,
    )
    if footprint.geom_type == "Polygon":
        footprint = footprint.simplify(
            LIDAR_FOOTPRINT_SIMPLIFY_METERS,
            preserve_topology=True,
        )
    return footprint


def scene_origin_elevation(
    dataset: rasterio.io.DatasetReader,
    latitude: float,
    longitude: float,
) -> float:
    xs, ys = transform_coordinates(
        "EPSG:4326", dataset.crs, [longitude], [latitude]
    )
    row, col = dataset.index(xs[0], ys[0])
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        raise ValueError(
            "The crop location is outside the returned USGS elevation raster."
        )
    value = dataset.read(1, window=((row, row + 1), (col, col + 1)), masked=True)[0, 0]
    if np.ma.is_masked(value) or not np.isfinite(value):
        raise ValueError(
            "The crop location has no valid bare-earth elevation in the "
            "USGS raster; the local scene cannot be vertically anchored."
        )
    return float(value)


def terrain_mesh(
    raster_path: Path,
    latitude: float,
    longitude: float,
    max_dimension: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read/downsample a DEM and make a triangulated local terrain surface."""
    if max_dimension < 2:
        raise ValueError("Terrain mesh maximum dimension must be at least 2.")
    with rasterio.open(raster_path) as dataset:
        if dataset.count != 1:
            raise ValueError(
                f"Expected a single-band elevation raster, got {dataset.count}."
            )
        if dataset.crs is None:
            raise ValueError("The elevation raster has no CRS.")
        origin_elevation = scene_origin_elevation(
            dataset, latitude, longitude
        )
        scale = max(
            1,
            math.ceil(
                max(dataset.width, dataset.height) / max_dimension
            ),
        )
        out_height = max(2, math.ceil(dataset.height / scale))
        out_width = max(2, math.ceil(dataset.width / scale))
        elevation = dataset.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.bilinear,
        )
        transform = dataset.transform * Affine.scale(
            dataset.width / out_width,
            dataset.height / out_height,
        )
        rows, cols = np.indices((out_height, out_width))
        projected_x, projected_y = transform * (
            cols + 0.5,
            rows + 0.5,
        )
        longitudes, latitudes = transform_coordinates(
            dataset.crs,
            "EPSG:4326",
            projected_x.ravel().tolist(),
            projected_y.ravel().tolist(),
        )
        east = np.empty(len(longitudes), dtype=np.float64)
        north = np.empty(len(latitudes), dtype=np.float64)
        for index, (point_lat, point_lon) in enumerate(
            zip(latitudes, longitudes)
        ):
            east[index], north[index] = building_data.local_xy(
                point_lat, point_lon, latitude, longitude
            )

        values = np.ma.filled(elevation, np.nan).astype(np.float64)
        values -= origin_elevation
        valid = (~np.ma.getmaskarray(elevation)) & np.isfinite(values)
        grid_indices = np.full((out_height, out_width), -1, dtype=np.int64)
        valid_positions = np.flatnonzero(valid.ravel())
        grid_indices.ravel()[valid_positions] = np.arange(
            len(valid_positions), dtype=np.int64
        )
        vertices = [
            (
                float(east[index]),
                float(north[index]),
                float(values.ravel()[index]),
            )
            for index in valid_positions
        ]
        faces: list[tuple[int, int, int]] = []
        for row in range(out_height - 1):
            for col in range(out_width - 1):
                a = grid_indices[row, col]
                b = grid_indices[row, col + 1]
                c = grid_indices[row + 1, col + 1]
                d = grid_indices[row + 1, col]
                if min(a, b, c, d) < 0:
                    continue
                faces.extend(((int(a), int(b), int(c)), (int(a), int(c), int(d))))
        if not vertices or not faces:
            raise ValueError(
                "The DEM did not contain enough valid cells to create a "
                "terrain surface."
            )
        mesh = {
            "name": "3DEP_bare_earth_terrain",
            "layer": "terrain",
            "vertices": vertices,
            "faces": faces,
            "color": LAYER_COLORS["terrain"],
            "grid_indices": grid_indices.tolist(),
        }
        details = {
            "source": "USGS 3DEP Bare Earth DEM Dynamic service",
            "raster": str(raster_path),
            "crs": dataset.crs.to_string(),
            "width_pixels": dataset.width,
            "height_pixels": dataset.height,
            "native_output_pixel_size": {
                "x_meters": abs(dataset.transform.a),
                "y_meters": abs(dataset.transform.e),
            },
            "mesh_grid": {"width": out_width, "height": out_height},
            "scene_origin_bare_earth_elevation_m": origin_elevation,
            "aoi_boundary_local": [
                (float(east[0]), float(north[0])),
                (
                    float(east[out_width - 1]),
                    float(north[out_width - 1]),
                ),
                (
                    float(east[-1]),
                    float(north[-1]),
                ),
                (
                    float(east[-out_width]),
                    float(north[-out_width]),
                ),
            ],
            "valid_vertices": len(vertices),
            "triangles": len(faces),
            "vertical_reference": (
                "Elevation values are offset by the terrain elevation sampled "
                "at the scene origin; source vertical datum varies by DEM."
            ),
        }
    return mesh, details


def sample_ground_elevation(
    dataset: rasterio.io.DatasetReader,
    latitude: float,
    longitude: float,
    feature_id: str,
) -> float:
    xs, ys = transform_coordinates(
        "EPSG:4326", dataset.crs, [longitude], [latitude]
    )
    row, col = dataset.index(xs[0], ys[0])
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        nearest_row = min(max(row, 0), dataset.height - 1)
        nearest_col = min(max(col, 0), dataset.width - 1)
        pixel_diagonal = math.hypot(
            dataset.transform.a,
            dataset.transform.e,
        )
        pixel_x, pixel_y = dataset.xy(nearest_row, nearest_col)
        distance = math.hypot(xs[0] - pixel_x, ys[0] - pixel_y)
        if distance > pixel_diagonal * 2:
            raise ValueError(
                f"Overture feature {feature_id} has no ground sample inside "
                "the downloaded DEM extent."
            )
        row, col = nearest_row, nearest_col
    value = dataset.read(1, window=((row, row + 1), (col, col + 1)), masked=True)[0, 0]
    if np.ma.is_masked(value) or not np.isfinite(value):
        raise ValueError("A building footprint has no valid DEM ground sample.")
    return float(value)


def lidar_search_boundary(
    scene_area_polygon: list[tuple[float, float]],
    buildings: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    buffer_meters: float = LIDAR_BUILDING_MATCH_DISTANCE_M,
) -> list[tuple[float, float]]:
    """Bound scene and full selected footprints with room for nearby returns."""
    scene_easts = [point[0] for point in scene_area_polygon]
    scene_norths = [point[1] for point in scene_area_polygon]
    east_values = list(scene_easts)
    north_values = list(scene_norths)
    for building in buildings:
        for ring in building["rings"]:
            for point_latitude, point_longitude in ring:
                east, north = building_data.local_xy(
                    point_latitude,
                    point_longitude,
                    latitude,
                    longitude,
                )
                east_values.append(east)
                north_values.append(north)
    west = min(east_values) - buffer_meters
    east = max(east_values) + buffer_meters
    south = min(north_values) - buffer_meters
    north = max(north_values) + buffer_meters
    return [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
    ]


def _lidar_only_building_candidates(
    point_xy: np.ndarray,
    point_z: np.ndarray,
    height_above_ground: np.ndarray,
    candidate_point_indices: np.ndarray,
    existing_footprints: list[Any],
    scene_area_polygon: list[tuple[float, float]],
    latitude: float,
    longitude: float,
) -> list[dict[str, Any]]:
    """Find elevated, supported class-6 clusters without mapped footprints."""
    from shapely import points
    from shapely.geometry import Polygon
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    scene_polygon = Polygon(scene_area_polygon)
    if scene_polygon.is_empty or not scene_polygon.is_valid:
        raise ValueError("The LiDAR-only building search area is invalid.")

    candidate_indices = candidate_point_indices[
        np.isfinite(height_above_ground[candidate_point_indices])
        & (
            height_above_ground[candidate_point_indices]
            >= MIN_LIDAR_ONLY_BUILDING_HEIGHT_M
        )
    ]
    if not len(candidate_indices):
        return []

    inside_scene = scene_polygon.covers(
        points(point_xy[candidate_indices, 0], point_xy[candidate_indices, 1])
    )
    candidate_indices = candidate_indices[np.asarray(inside_scene)]
    if not len(candidate_indices):
        return []

    cell_size = fetch_lidar_point_cloud.VEGETATION_CLUSTER_CELL_SIZE_METERS
    cell_buffer = fetch_lidar_point_cloud.VEGETATION_CLUSTER_CELL_BUFFER_METERS
    max_fit_points = fetch_lidar_point_cloud.MAX_DBSCAN_POINTS_PER_CELL_CLASS
    by_cell: dict[tuple[int, int], np.ndarray] = {}
    candidate_xy = point_xy[candidate_indices]
    cell_keys = np.floor(candidate_xy / cell_size).astype(np.int64)
    for cell_key in np.unique(cell_keys, axis=0):
        selected = np.all(cell_keys == cell_key, axis=1)
        by_cell[(int(cell_key[0]), int(cell_key[1]))] = candidate_indices[selected]

    random_generator = np.random.default_rng(20261103)
    candidates = []
    for east_index, north_index in sorted(by_cell):
        core_bounds = (
            east_index * cell_size,
            north_index * cell_size,
            (east_index + 1) * cell_size,
            (north_index + 1) * cell_size,
        )
        search_bounds = (
            core_bounds[0] - cell_buffer,
            core_bounds[1] - cell_buffer,
            core_bounds[2] + cell_buffer,
            core_bounds[3] + cell_buffer,
        )
        neighboring_indices = [
            by_cell[(neighbor_east, neighbor_north)]
            for neighbor_east in range(east_index - 1, east_index + 2)
            for neighbor_north in range(north_index - 1, north_index + 2)
            if (neighbor_east, neighbor_north) in by_cell
        ]
        if not neighboring_indices:
            continue
        indices = np.concatenate(neighboring_indices)
        xy = point_xy[indices]
        in_buffer = (
            (xy[:, 0] >= search_bounds[0])
            & (xy[:, 1] >= search_bounds[1])
            & (xy[:, 0] <= search_bounds[2])
            & (xy[:, 1] <= search_bounds[3])
        )
        indices = indices[in_buffer]
        if len(indices) < MIN_LIDAR_ONLY_BUILDING_POINTS:
            continue
        if len(indices) > max_fit_points:
            indices = indices[
                random_generator.choice(
                    len(indices), size=max_fit_points, replace=False
                )
            ]
        cluster_xy = point_xy[indices]
        if len(cluster_xy) < MIN_LIDAR_ONLY_BUILDING_POINTS:
            continue
        neighbor_distances, _ = NearestNeighbors(
            n_neighbors=4,
            algorithm="kd_tree",
            n_jobs=1,
        ).fit(cluster_xy).kneighbors(cluster_xy)
        epsilon = min(
            2.0,
            max(1.0, 2.0 * float(np.mean(neighbor_distances[:, 1:]))),
        )
        if not math.isfinite(epsilon) or epsilon <= 0:
            continue
        labels = DBSCAN(
            eps=epsilon,
            min_samples=5,
            algorithm="kd_tree",
            n_jobs=1,
        ).fit_predict(cluster_xy)
        cluster_labels, cluster_sizes = np.unique(
            labels[labels >= 0], return_counts=True
        )
        for label, point_count in zip(cluster_labels, cluster_sizes):
            if point_count < MIN_LIDAR_ONLY_BUILDING_POINTS:
                continue
            member_indices = np.flatnonzero(labels == label)
            cluster_xy = point_xy[indices[member_indices]]
            center_east, center_north = np.mean(cluster_xy, axis=0)
            if not (
                core_bounds[0] <= center_east < core_bounds[2]
                and core_bounds[1] <= center_north < core_bounds[3]
            ):
                continue
            median_height = float(np.median(
                height_above_ground[indices[member_indices]]
            ))
            if median_height < MIN_LIDAR_ONLY_BUILDING_HEIGHT_M:
                continue
            hull = lidar_cluster_footprint(cluster_xy)
            if hull.geom_type != "Polygon" or hull.area < (
                MIN_LIDAR_ONLY_BUILDING_AREA_M2
            ):
                continue
            if any(
                hull.intersection(footprint).area / hull.area >= 0.5
                for footprint in existing_footprints
                if hull.intersects(footprint)
            ):
                continue

            roof_elevation = float(np.quantile(
                point_z[indices[member_indices]],
                LIDAR_ROOF_ELEVATION_QUANTILE,
            ))
            local_ring = [
                (float(east), float(north))
                for east, north in hull.exterior.coords
            ]
            geographic_ring = [
                (
                    latitude + math.degrees(
                        north / building_data.EARTH_RADIUS_METERS
                    ),
                    longitude + math.degrees(
                        east / (
                            building_data.EARTH_RADIUS_METERS
                            * math.cos(math.radians(latitude))
                        )
                    ),
                )
                for east, north in local_ring
            ]
            candidate_id = f"lidar-only-{len(candidates) + 1:04d}"
            candidates.append({
                "osm_type": "USGS 3DEP",
                "osm_id": candidate_id,
                "tags": {},
                "rings": [geographic_ring],
                "geometry_source": (
                    "USGS 3DEP class-6 LiDAR-only building candidate"
                ),
                "lidar_only": True,
                "lidar_building_point_count": int(point_count),
                "lidar_building_inlier_count": int(point_count),
                "lidar_building_cluster_count": 1,
                "lidar_median_height_above_ground_m": median_height,
                "lidar_dbscan_epsilon_m": epsilon,
                "lidar_roof_regions": [{
                    "footprint_local": local_ring,
                    "roof_elevation_m": roof_elevation,
                    "median_elevation_m": float(np.median(
                        point_z[indices[member_indices]]
                    )),
                    "point_count": int(point_count),
                    "area_m2": float(hull.area),
                }],
            })
    return candidates


def adjust_buildings_with_lidar(
    buildings: list[dict[str, Any]],
    point_cloud_paths: list[Path],
    tile_vertical_offsets: dict[str, float],
    latitude: float,
    longitude: float,
    raster_path: Path | None = None,
    scene_area_polygon: list[tuple[float, float]] | None = None,
    maximum_footprint_distance_m: float = LIDAR_BUILDING_MATCH_DISTANCE_M,
    minimum_cluster_points: int = 20,
) -> dict[str, Any]:
    """Adjust mapped footprints and add supported LiDAR-only building candidates."""
    if not point_cloud_paths:
        return {
            "status": "no_building_points",
            "adjusted_buildings": 0,
            "unmatched_building_point_count": 0,
            "maximum_footprint_distance_m": maximum_footprint_distance_m,
        }
    if scene_area_polygon is None or raster_path is None:
        raise ValueError(
            "LiDAR building detection requires the terrain raster and "
            "scene boundary."
        )

    try:
        import laspy
        from pyproj import Transformer
        from shapely import distance, points
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
    except ImportError as error:
        raise RuntimeError(
            "LiDAR-based building adjustment requires laspy, Shapely, and "
            "scikit-learn from the project requirements."
        ) from error

    local_polygons = []
    for building in buildings:
        parts = [
            Polygon([
                building_data.local_xy(
                    point_lat,
                    point_lon,
                    latitude,
                    longitude,
                )
                for point_lat, point_lon in ring
            ])
            for ring in building["rings"]
        ]
        footprint = unary_union(parts)
        if footprint.is_empty or not footprint.is_valid or footprint.area <= 0:
            raise ValueError(
                f"Invalid footprint geometry for Overture feature "
                f"{building['osm_id']} during LiDAR matching."
            )
        local_polygons.append(footprint)

    point_xy_parts = []
    point_z_parts = []
    vertical_alignment_available = True
    search_polygons = [
        Polygon(scene_area_polygon),
        *local_polygons,
    ]
    point_bounds = (
        min(polygon.bounds[0] for polygon in search_polygons)
        - maximum_footprint_distance_m,
        min(polygon.bounds[1] for polygon in search_polygons)
        - maximum_footprint_distance_m,
        max(polygon.bounds[2] for polygon in search_polygons)
        + maximum_footprint_distance_m,
        max(polygon.bounds[3] for polygon in search_polygons)
        + maximum_footprint_distance_m,
    )
    for point_cloud_path in point_cloud_paths:
        source_tile = point_cloud_path.name
        tile_offset = tile_vertical_offsets.get(source_tile)
        if tile_offset is None:
            vertical_alignment_available = False
        with laspy.open(point_cloud_path) as reader:
            source_crs = reader.header.parse_crs()
            if source_crs is None:
                raise ValueError(
                    f"LiDAR building points have no CRS: "
                    f"{point_cloud_path}"
                )
            transformer = Transformer.from_crs(
                source_crs, "EPSG:4326", always_xy=True
            )
            for chunk in reader.chunk_iterator(
                fetch_lidar_point_cloud.CHUNK_SIZE
            ):
                building_indices = np.flatnonzero(
                    np.asarray(chunk.classification) == 6
                )
                if not len(building_indices):
                    continue
                point_longitudes, point_latitudes = transformer.transform(
                    np.asarray(chunk.x)[building_indices],
                    np.asarray(chunk.y)[building_indices],
                )
                east = (
                    np.radians(
                        np.asarray(point_longitudes) - longitude
                    )
                    * building_data.EARTH_RADIUS_METERS
                    * math.cos(math.radians(latitude))
                )
                north = (
                    np.radians(
                        np.asarray(point_latitudes) - latitude
                    )
                    * building_data.EARTH_RADIUS_METERS
                )
                in_bounds = (
                    (east >= point_bounds[0])
                    & (north >= point_bounds[1])
                    & (east <= point_bounds[2])
                    & (north <= point_bounds[3])
                )
                if not np.any(in_bounds):
                    continue
                point_xy_parts.append(
                    np.column_stack((east[in_bounds], north[in_bounds]))
                )
                point_z_parts.append(
                    np.asarray(chunk.z, dtype=np.float64)[building_indices][
                        in_bounds
                    ]
                    - (tile_offset if tile_offset is not None else 0.0)
                )

    if not point_xy_parts:
        return {
            "status": "no_class_6_points",
            "adjusted_buildings": 0,
            "unmatched_building_point_count": 0,
            "maximum_footprint_distance_m": maximum_footprint_distance_m,
        }
    if not vertical_alignment_available:
        return {
            "status": "vertical_alignment_unavailable",
            "adjusted_buildings": 0,
            "class_6_point_count": sum(
                len(part) for part in point_xy_parts
            ),
            "maximum_footprint_distance_m": maximum_footprint_distance_m,
            "note": (
                "Class-6 points were not used because one or more source "
                "tiles lacked a ground-derived vertical alignment offset."
            ),
        }

    point_xy = np.concatenate(point_xy_parts)
    point_z = np.concatenate(point_z_parts)
    point_building_pairs = []
    query_chunk_size = 20_000
    assigned_point_indices = np.array([], dtype=np.int64)
    points_by_building: dict[int, np.ndarray] = {}
    if local_polygons:
        spatial_index = STRtree(local_polygons)
        for start in range(0, len(point_xy), query_chunk_size):
            end = min(start + query_chunk_size, len(point_xy))
            point_geometries = points(
                point_xy[start:end, 0], point_xy[start:end, 1]
            )
            pairs = spatial_index.query(
                point_geometries,
                predicate="dwithin",
                distance=maximum_footprint_distance_m,
            )
            if not pairs.size:
                continue
            point_indices = pairs[0] + start
            building_indices = pairs[1]
            pair_distances = distance(
                point_geometries[pairs[0]],
                np.asarray(local_polygons, dtype=object)[building_indices],
            )
            point_building_pairs.append(
                np.column_stack((
                    point_indices, building_indices, pair_distances
                ))
            )

    if point_building_pairs:
        pairs = np.concatenate(point_building_pairs)
        order = np.lexsort((pairs[:, 2], pairs[:, 0]))
        ordered_pairs = pairs[order]
        assigned_point_indices, first_positions = np.unique(
            ordered_pairs[:, 0].astype(np.int64),
            return_index=True,
        )
        nearest_buildings = ordered_pairs[first_positions, 1].astype(np.int64)
        points_by_building = {
            int(building_index): assigned_point_indices[
                nearest_buildings == building_index
            ]
            for building_index in np.unique(nearest_buildings)
        }
    assigned_point_count = len(assigned_point_indices)

    adjusted_count = 0
    clustered_point_count = 0
    total_cluster_count = 0
    for building_index, point_indices in points_by_building.items():
        if len(point_indices) < minimum_cluster_points:
            continue
        building_xy = point_xy[point_indices]
        sample = building_xy
        if len(sample) > 10_000:
            sample_indices = np.linspace(
                0, len(sample) - 1, num=10_000, dtype=np.int64
            )
            sample = sample[sample_indices]
        neighbor_distances, _ = NearestNeighbors(
            n_neighbors=4,
            algorithm="kd_tree",
            n_jobs=-1,
        ).fit(sample).kneighbors(sample)
        epsilon = 2.0 * float(np.mean(neighbor_distances[:, 1:]))
        if not math.isfinite(epsilon) or epsilon <= 0:
            continue
        building_xyz = np.column_stack((building_xy, point_z[point_indices]))
        labels = DBSCAN(
            eps=epsilon,
            min_samples=5,
            algorithm="kd_tree",
            n_jobs=-1,
        ).fit_predict(building_xyz)
        cluster_labels, cluster_sizes = np.unique(
            labels[labels >= 0],
            return_counts=True,
        )
        usable_clusters = [
            int(label)
            for label, count in zip(cluster_labels, cluster_sizes)
            if count >= minimum_cluster_points
        ]
        if not usable_clusters:
            continue
        cluster_regions = []
        inlier_indices = []
        for label in usable_clusters:
            member_indices = np.flatnonzero(labels == label)
            hull = lidar_cluster_footprint(building_xy[member_indices])
            if hull.geom_type == "Polygon" and hull.area > 1.0:
                region_points = point_indices[member_indices]
                cluster_regions.append({
                    "footprint_local": [
                        [float(east), float(north)]
                        for east, north in hull.exterior.coords
                    ],
                    "roof_elevation_m": float(np.quantile(
                        point_z[region_points],
                        LIDAR_ROOF_ELEVATION_QUANTILE,
                    )),
                    "median_elevation_m": float(
                        np.median(point_z[region_points])
                    ),
                    "point_count": len(member_indices),
                    "area_m2": float(hull.area),
                })
                inlier_indices.extend(member_indices.tolist())
        if not cluster_regions or not inlier_indices:
            continue

        building = buildings[building_index]
        combined_geometry = unary_union((
            local_polygons[building_index],
            *[
                Polygon(region["footprint_local"])
                for region in cluster_regions
            ],
        ))
        building["lidar_roof_regions"] = cluster_regions
        building["lidar_building_point_count"] = len(point_indices)
        building["lidar_building_inlier_count"] = len(inlier_indices)
        building["lidar_building_cluster_count"] = len(cluster_regions)
        building["lidar_dbscan_epsilon_m"] = epsilon
        building["lidar_footprint_added_area_m2"] = max(
            0.0,
            combined_geometry.area - local_polygons[building_index].area,
        )
        building["geometry_source"] = (
            "Overture footprint with USGS 3DEP class-6 height regions"
        )
        adjusted_count += 1
        clustered_point_count += len(inlier_indices)
        total_cluster_count += len(usable_clusters)

    candidate_point_indices = np.setdiff1d(
        np.arange(len(point_xy), dtype=np.int64),
        assigned_point_indices,
        assume_unique=True,
    )
    lidar_only_buildings = []
    if len(candidate_point_indices):
        point_lons = longitude + np.degrees(
            point_xy[:, 0]
            / (
                building_data.EARTH_RADIUS_METERS
                * math.cos(math.radians(latitude))
            )
        )
        point_lats = latitude + np.degrees(
            point_xy[:, 1] / building_data.EARTH_RADIUS_METERS
        )
        with rasterio.open(raster_path) as dataset:
            raster_x, raster_y = transform_coordinates(
                "EPSG:4326",
                dataset.crs,
                point_lons.tolist(),
                point_lats.tolist(),
            )
            rows, cols = rasterio.transform.rowcol(
                dataset.transform, raster_x, raster_y
            )
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)
            in_raster = (
                (rows >= 0)
                & (rows < dataset.height)
                & (cols >= 0)
                & (cols < dataset.width)
            )
            ground = np.full(len(point_z), np.nan, dtype=np.float64)
            dem = dataset.read(1, masked=True)
            valid = np.flatnonzero(in_raster)
            if len(valid):
                values = dem[rows[valid], cols[valid]]
                valid_dem = ~np.ma.getmaskarray(values)
                ground[valid[valid_dem]] = np.asarray(values)[valid_dem]
        height_above_ground = point_z - ground
        lidar_only_buildings = _lidar_only_building_candidates(
            point_xy,
            point_z,
            height_above_ground,
            candidate_point_indices,
            local_polygons,
            scene_area_polygon,
            latitude,
            longitude,
        )
        buildings.extend(lidar_only_buildings)

    return {
        "status": (
            "adjusted"
            if adjusted_count
            else (
                "lidar_only_candidates_detected"
                if lidar_only_buildings else "no_reliable_clusters"
            )
        ),
        "adjusted_buildings": adjusted_count,
        "lidar_only_building_count": len(lidar_only_buildings),
        "lidar_only_buildings": [
            {
                "id": building["osm_id"],
                "point_count": building["lidar_building_point_count"],
                "footprint_area_m2": building["lidar_roof_regions"][0][
                    "area_m2"
                ],
                "roof_elevation_m": building["lidar_roof_regions"][0][
                    "roof_elevation_m"
                ],
                "median_height_above_ground_m": building[
                    "lidar_median_height_above_ground_m"
                ],
                "dbscan_epsilon_m": building["lidar_dbscan_epsilon_m"],
            }
            for building in lidar_only_buildings
        ],
        "buildings_with_assigned_points": len(points_by_building),
        "class_6_point_count": len(point_xy),
        "points_near_footprints": assigned_point_count,
        "unmatched_building_point_count": len(point_xy) - assigned_point_count,
        "clustered_inlier_point_count": clustered_point_count,
        "cluster_count": total_cluster_count,
        "maximum_footprint_distance_m": maximum_footprint_distance_m,
        "vertical_alignment_available": vertical_alignment_available,
        "dbscan": {
            "epsilon_method": (
                "Per-building epsilon is twice the mean distance to each "
                "point's three nearest horizontal neighbors; DBSCAN is then "
                "run in east/north/elevation coordinates."
            ),
            "minimum_samples": 5,
            "minimum_cluster_points": minimum_cluster_points,
            "footprint_method": (
                "Simplified concave hull per usable 3D class-6 cluster."
            ),
            "vertical_method": (
                f"Each cluster uses its {LIDAR_ROOF_ELEVATION_QUANTILE:.0%} "
                "inlier elevation. Higher regions take precedence over "
                "overlapping lower regions; uncovered Overture footprint "
                "areas keep their original height."
            ),
        },
    }


def extrude_local_polygon(
    polygon: Any,
    base_elevation: float,
    roof_elevation: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Create a closed stepped-volume mesh for a polygon with optional holes."""
    try:
        from shapely import constrained_delaunay_triangles
    except ImportError:
        from shapely.ops import triangulate

        roof_triangles = (
            triangle for triangle in triangulate(polygon)
            if polygon.covers(triangle)
        )
    else:
        roof_triangles = constrained_delaunay_triangles(polygon).geoms

    rings = [polygon.exterior, *polygon.interiors]
    boundary_coordinates = [
        [(float(x), float(y)) for x, y in ring.coords[:-1]]
        for ring in rings
    ]
    flattened = [point for ring in boundary_coordinates for point in ring]
    coordinate_indices = {
        (round(x, 8), round(y, 8)): index
        for index, (x, y) in enumerate(flattened)
    }
    count = len(flattened)
    vertices = [
        (x, y, base_elevation) for x, y in flattened
    ] + [
        (x, y, roof_elevation) for x, y in flattened
    ]
    faces: list[tuple[int, int, int]] = []
    for triangle in roof_triangles:
        triangle_indices = [
            coordinate_indices[(round(x, 8), round(y, 8))]
            for x, y in list(triangle.exterior.coords)[:3]
        ]
        a, b, c = triangle_indices
        faces.extend((
            (a, c, b),
            (a + count, b + count, c + count),
        ))
    for ring_coordinates in boundary_coordinates:
        for index, (x, y) in enumerate(ring_coordinates):
            next_x, next_y = ring_coordinates[
                (index + 1) % len(ring_coordinates)
            ]
            start = coordinate_indices[(round(x, 8), round(y, 8))]
            end = coordinate_indices[
                (round(next_x, 8), round(next_y, 8))
            ]
            faces.extend((
                (start, end, end + count),
                (start, end + count, start + count),
            ))
    return vertices, faces


def building_wireframe(
    polygon: Any,
    base_elevation: float,
    roof_elevation: float,
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
    list[float],
    list[float],
    list[float],
]:
    """Return polygon boundary edges and corner points for scene display."""
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    corner_x: list[float] = []
    corner_y: list[float] = []
    corner_z: list[float] = []
    for ring in (polygon.exterior, *polygon.interiors):
        coordinates = [
            (float(x), float(y)) for x, y in ring.coords[:-1]
        ]
        if not coordinates:
            continue
        closed_coordinates = [*coordinates, coordinates[0]]
        for elevation in (base_elevation, roof_elevation):
            line_x.extend([x for x, _y in closed_coordinates] + [None])
            line_y.extend([y for _x, y in closed_coordinates] + [None])
            line_z.extend([elevation] * len(closed_coordinates) + [None])
            corner_x.extend(x for x, _y in coordinates)
            corner_y.extend(y for _x, y in coordinates)
            corner_z.extend([elevation] * len(coordinates))
        for x, y in coordinates:
            line_x.extend((x, x, None))
            line_y.extend((y, y, None))
            line_z.extend((base_elevation, roof_elevation, None))
    return line_x, line_y, line_z, corner_x, corner_y, corner_z


def building_meshes(
    buildings: list[dict[str, Any]],
    raster_path: Path,
    latitude: float,
    longitude: float,
    origin_elevation: float,
    target_reference: str | None,
    scene_area_polygon: list[tuple[float, float]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Extrude Overture and LiDAR roof regions at their own elevations."""
    from shapely.geometry import GeometryCollection, Polygon, mapping
    from shapely.ops import unary_union

    meshes = []
    records = []
    comparison_features = []
    with rasterio.open(raster_path) as dataset:
        raster_local_area = Polygon(scene_area_polygon)
        if raster_local_area.is_empty or not raster_local_area.is_valid:
            raise ValueError("The terrain mesh has invalid AOI boundary geometry.")
        for building in buildings:
            overture_height, overture_height_source = (
                building_data.building_height(building)
            )
            local_polygons = []
            for ring in building["rings"]:
                polygon = Polygon([
                    building_data.local_xy(
                        point_lat,
                        point_lon,
                        latitude,
                        longitude,
                    )
                    for point_lat, point_lon in ring
                ])
                if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                    raise ValueError(
                        f"Invalid footprint geometry for Overture feature "
                        f"{building['osm_id']}."
                    )
                local_polygons.append(polygon)
            footprint = unary_union(local_polygons)
            ground_sample_area = footprint.intersection(raster_local_area)
            if ground_sample_area.is_empty:
                raise ValueError(
                    f"Overture feature {building['osm_id']} has no footprint "
                    "area inside the downloaded DEM extent."
                )
            point = ground_sample_area.representative_point()
            sample_latitude = latitude + math.degrees(
                point.y / building_data.EARTH_RADIUS_METERS
            )
            sample_longitude = longitude + math.degrees(
                point.x
                / (
                    building_data.EARTH_RADIUS_METERS
                    * math.cos(math.radians(latitude))
                )
            )
            ground_elevation = sample_ground_elevation(
                dataset,
                sample_latitude,
                sample_longitude,
                building["osm_id"],
            )
            source = building.get("geometry_source", "Overture Maps")
            is_target = target_reference in building.get(
                "osm_references", []
            ) if target_reference else False
            lidar_regions = []
            for region_index, region in enumerate(
                building.get("lidar_roof_regions", [])
            ):
                roof_elevation = float(region["roof_elevation_m"])
                if roof_elevation <= ground_elevation:
                    continue
                region_polygon = Polygon(region["footprint_local"])
                if region_polygon.is_empty or not region_polygon.is_valid:
                    continue
                if region_polygon.is_empty or region_polygon.area <= 1.0:
                    continue
                lidar_regions.append((region_index, region, region_polygon))
            lidar_regions.sort(
                key=lambda item: float(item[1]["roof_elevation_m"]),
                reverse=True,
            )
            lidar_footprint = unary_union([
                region_polygon for _index, _region, region_polygon in lidar_regions
            ]) if lidar_regions else GeometryCollection()
            lidar_only_candidate = bool(building.get("lidar_only", False))
            comparison_features.append({
                "id": str(building["osm_id"]),
                "lidar_only_candidate": lidar_only_candidate,
                "overture_footprint_local": (
                    None if lidar_only_candidate else mapping(footprint)
                ),
                "lidar_footprint_local": (
                    mapping(lidar_footprint)
                    if not lidar_footprint.is_empty else None
                ),
                "ground_elevation_m": ground_elevation,
                "overture_height_m": (
                    None if lidar_only_candidate else overture_height
                ),
                "lidar_roof_elevations_m": [
                    float(region["roof_elevation_m"])
                    for _index, region, _polygon in lidar_regions
                ],
                "lidar_point_count": int(building.get(
                    "lidar_building_inlier_count",
                    building.get("lidar_building_point_count", 0),
                ) or 0),
            })

            covered_by_lidar: Any = GeometryCollection()
            components: list[tuple[Any, float, str, dict[str, Any]]] = []
            for region_index, region, region_polygon in lidar_regions:
                visible_region = region_polygon.difference(covered_by_lidar)
                parts = (
                    list(visible_region.geoms)
                    if visible_region.geom_type == "MultiPolygon"
                    else [visible_region]
                )
                roof_elevation = float(region["roof_elevation_m"])
                for part in parts:
                    if part.geom_type != "Polygon" or part.area <= 1.0:
                        continue
                    components.append((
                        part,
                        roof_elevation,
                        (
                            "USGS 3DEP class-6 95th-percentile inlier elevation "
                            f"(region {region_index + 1})"
                        ),
                        region,
                    ))
                covered_by_lidar = unary_union(
                    (covered_by_lidar, region_polygon)
                )

            overture_only_geometry = footprint.difference(covered_by_lidar)
            overture_parts = (
                list(overture_only_geometry.geoms)
                if overture_only_geometry.geom_type == "MultiPolygon"
                else [overture_only_geometry]
            )
            for part in overture_parts:
                if part.geom_type == "Polygon" and part.area > 1.0:
                    components.append((
                        part,
                        ground_elevation + overture_height,
                        overture_height_source,
                        {},
                    ))

            roof_level_groups: list[dict[str, Any]] = []
            for component in sorted(components, key=lambda item: item[1]):
                part, roof_elevation, height_source, region = component
                matching_groups = [
                    group for group in roof_level_groups
                    if abs(
                        roof_elevation
                        - float(np.median(group["roof_elevations"]))
                    ) <= ROOF_LEVEL_MERGE_TOLERANCE_M
                ]
                if matching_groups:
                    group = min(
                        matching_groups,
                        key=lambda candidate: abs(
                            roof_elevation
                            - float(np.median(candidate["roof_elevations"]))
                        ),
                    )
                else:
                    group = {
                        "parts": [],
                        "roof_elevations": [],
                        "height_sources": [],
                        "regions": [],
                        "has_overture_geometry": False,
                    }
                    roof_level_groups.append(group)
                group["parts"].append(part)
                group["roof_elevations"].append(roof_elevation)
                group["height_sources"].append(height_source)
                if region:
                    group["regions"].append(region)
                else:
                    group["has_overture_geometry"] = True

            cleaned_components: list[
                tuple[Any, float, str, dict[str, Any]]
            ] = []
            for group in roof_level_groups:
                merged_geometry = unary_union(group["parts"])
                merged_parts = (
                    list(merged_geometry.geoms)
                    if merged_geometry.geom_type == "MultiPolygon"
                    else [merged_geometry]
                )
                roof_elevation = float(np.median(group["roof_elevations"]))
                lidar_regions = group["regions"]
                height_sources = group["height_sources"]
                if lidar_regions and group["has_overture_geometry"]:
                    height_source = (
                        "Merged USGS 3DEP LiDAR and Overture roof regions"
                    )
                elif lidar_regions:
                    height_source = (
                        "USGS 3DEP class-6 95th-percentile inlier elevation "
                        "(merged roof regions)"
                    )
                else:
                    height_source = height_sources[0]
                region = (
                    {
                        "median_elevation_m": float(np.median([
                            item["median_elevation_m"]
                            for item in lidar_regions
                            if "median_elevation_m" in item
                        ])),
                        "point_count": sum(
                            int(item.get("point_count", 0))
                            for item in lidar_regions
                        ),
                    }
                    if lidar_regions else {}
                )
                for part in merged_parts:
                    if part.geom_type == "Polygon" and part.area > 1.0:
                        cleaned_components.append((
                            part,
                            roof_elevation,
                            height_source,
                            region,
                        ))

            for part_index, (
                part, roof_elevation, height_source, region
            ) in enumerate(
                cleaned_components
            ):
                height = roof_elevation - ground_elevation
                vertices, faces = extrude_local_polygon(
                    part,
                    ground_elevation - origin_elevation,
                    roof_elevation - origin_elevation,
                )
                wireframe = building_wireframe(
                    part,
                    ground_elevation - origin_elevation,
                    roof_elevation - origin_elevation,
                )
                name = sanitize_name(
                    f"building_{building['osm_id']}_roof_region_{part_index}"
                )
                region_details = {
                    "roof_elevation_m": roof_elevation,
                    "roof_median_elevation_m": region.get(
                        "median_elevation_m"
                    ),
                    "lidar_region_point_count": region.get("point_count"),
                    "footprint_area_m2": part.area,
                }
                mesh_metadata = {
                    "building_feature_id": building["osm_id"],
                    "overture_feature_id": (
                        None if building.get("lidar_only")
                        else building["osm_id"]
                    ),
                    "lidar_only_candidate": bool(
                        building.get("lidar_only", False)
                    ),
                    "geometry_source": (
                        source if region else "Overture footprint"
                    ),
                    "height_m": height,
                    "height_source": height_source,
                    "lidar_building_point_count": building.get(
                        "lidar_building_point_count"
                    ),
                    "lidar_building_inlier_count": building.get(
                        "lidar_building_inlier_count"
                    ),
                    "lidar_building_cluster_count": building.get(
                        "lidar_building_cluster_count"
                    ),
                    "lidar_dbscan_epsilon_m": building.get(
                        "lidar_dbscan_epsilon_m"
                    ),
                    "lidar_footprint_added_area_m2": building.get(
                        "lidar_footprint_added_area_m2"
                    ),
                    **region_details,
                    "ground_elevation_m": ground_elevation,
                    "osm_references": building.get("osm_references", []),
                    "is_target": is_target,
                    "target_reference": (
                        target_reference if is_target else None
                    ),
                }
                meshes.append({
                    "name": name,
                    "layer": "buildings",
                    "vertices": vertices,
                    "faces": faces,
                    "color": LAYER_COLORS["buildings"],
                    "metadata": mesh_metadata,
                    "wireframe": {
                        "line_x": wireframe[0],
                        "line_y": wireframe[1],
                        "line_z": wireframe[2],
                        "corner_x": wireframe[3],
                        "corner_y": wireframe[4],
                        "corner_z": wireframe[5],
                    },
                })
                records.append({
                    "id": building["osm_id"],
                    "lidar_only_candidate": bool(
                        building.get("lidar_only", False)
                    ),
                    "geometry_source": mesh_metadata["geometry_source"],
                    "height_m": height,
                    "height_source": height_source,
                    "lidar_building_point_count": building.get(
                        "lidar_building_point_count"
                    ),
                    "lidar_building_inlier_count": building.get(
                        "lidar_building_inlier_count"
                    ),
                    "lidar_building_cluster_count": building.get(
                        "lidar_building_cluster_count"
                    ),
                    "lidar_dbscan_epsilon_m": building.get(
                        "lidar_dbscan_epsilon_m"
                    ),
                    "lidar_footprint_added_area_m2": building.get(
                        "lidar_footprint_added_area_m2"
                    ),
                    **region_details,
                    "ground_elevation_m": ground_elevation,
                    "local_base_elevation_m": ground_elevation - origin_elevation,
                    "osm_references": building.get("osm_references", []),
                    "is_target": is_target,
                })
    return meshes, records, comparison_features


def load_extra_meshes(path: Path | None) -> list[dict[str, Any]]:
    """Load future balcony/tree meshes in scene-local XYZ metres."""
    if path is None:
        return []
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("--extra-meshes JSON must contain a list of mesh objects.")
    meshes = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Extra mesh {index} must be a JSON object.")
        layer = item.get("layer")
        if layer not in {"balconies", "trees", "other"}:
            raise ValueError(
                f"Extra mesh {index} has unsupported layer {layer!r}; "
                "use balconies, trees, or other."
            )
        vertices = item.get("vertices")
        faces = item.get("faces")
        if not isinstance(vertices, list) or not isinstance(faces, list):
            raise ValueError(
                f"Extra mesh {index} must include vertices and faces arrays."
            )
        parsed_vertices = []
        for vertex in vertices:
            if (
                not isinstance(vertex, list)
                or len(vertex) != 3
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in vertex
                )
            ):
                raise ValueError(
                    f"Extra mesh {index} vertices must be finite [east, north, "
                    "relative-elevation] triples in metres."
                )
            parsed_vertices.append(tuple(map(float, vertex)))
        parsed_faces = []
        for face in faces:
            if (
                not isinstance(face, list)
                or len(face) != 3
                or not all(
                    isinstance(value, int)
                    and 0 <= value < len(parsed_vertices)
                    for value in face
                )
            ):
                raise ValueError(
                    f"Extra mesh {index} faces must be zero-based vertex-index "
                    "triples."
                )
            parsed_faces.append(tuple(face))
        if not parsed_vertices or not parsed_faces:
            raise ValueError(f"Extra mesh {index} must not be empty.")
        meshes.append({
            "name": sanitize_name(str(item.get("name", f"{layer}_{index}"))),
            "layer": layer,
            "vertices": parsed_vertices,
            "faces": parsed_faces,
            "color": LAYER_COLORS[layer],
            "metadata": item.get("metadata", {}),
        })
    return meshes


def write_obj(meshes: list[dict[str, Any]], output_path: Path) -> None:
    """Write all scene meshes as a single OBJ with named layer groups."""
    vertex_offset = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("# VIP-6600-Solar neighborhood scene; units are metres\n")
        file.write("# XYZ: local east, local north, elevation above scene origin\n")
        for mesh in meshes:
            name = sanitize_name(mesh["name"])
            file.write(f"\ng {mesh['layer']}_{name}\n")
            file.write(f"o {name}\n")
            for x, y, z in mesh["vertices"]:
                file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            if not mesh["faces"]:
                for index in range(len(mesh["vertices"])):
                    file.write(f"p {vertex_offset + index + 1}\n")
            for a, b, c in mesh["faces"]:
                file.write(
                    f"f {a + vertex_offset + 1} "
                    f"{b + vertex_offset + 1} "
                    f"{c + vertex_offset + 1}\n"
                )
            vertex_offset += len(mesh["vertices"])


def make_scene_figure(
    meshes: list[dict[str, Any]],
    crop: dict[str, Any],
    maximum_terrain_triangles: int,
    camera_height: float,
    hit_record: dict[str, Any] | None,
    lidar_point_samples: dict[str, list[list[float]]],
    vegetation_point_preview: bool = False,
) -> go.Figure:
    figure = go.Figure()
    shown_terrain = False
    shown_other_layers: set[str] = set()
    building_edge_x: list[float | None] = []
    building_edge_y: list[float | None] = []
    building_edge_z: list[float | None] = []
    building_corner_x: list[float] = []
    building_corner_y: list[float] = []
    building_corner_z: list[float] = []
    for mesh in meshes:
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        layer = mesh["layer"]
        if layer == "terrain" and len(faces) > maximum_terrain_triangles:
            stride = math.ceil(
                math.sqrt(len(faces) / maximum_terrain_triangles)
            )
            grid_indices = mesh["grid_indices"]
            rows, cols = len(grid_indices), len(grid_indices[0])
            row_steps = list(range(0, rows - 1, stride)) + [rows - 1]
            col_steps = list(range(0, cols - 1, stride)) + [cols - 1]
            reduced_faces = []
            for row_index in range(len(row_steps) - 1):
                row = row_steps[row_index]
                next_row = row_steps[row_index + 1]
                for col_index in range(len(col_steps) - 1):
                    col = col_steps[col_index]
                    next_col = col_steps[col_index + 1]
                    a = grid_indices[row][col]
                    b = grid_indices[row][next_col]
                    c = grid_indices[next_row][next_col]
                    d = grid_indices[next_row][col]
                    if min(a, b, c, d) < 0:
                        continue
                    reduced_faces.extend(((a, b, c), (a, c, d)))
            figure.add_trace(go.Mesh3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                i=[face[0] for face in reduced_faces],
                j=[face[1] for face in reduced_faces],
                k=[face[2] for face in reduced_faces],
                name="3DEP bare-earth terrain",
                color=mesh["color"],
                opacity=0.92,
                flatshading=True,
                showlegend=not shown_terrain,
                hoverinfo="name",
            ))
            shown_terrain = True
            continue
        if layer == "terrain":
            name = "3DEP bare-earth terrain"
            showlegend = not shown_terrain
            shown_terrain = True
        elif layer == "buildings":
            is_target = bool(mesh.get("metadata", {}).get("is_target"))
            name = "Target building" if is_target else "Overture buildings"
            showlegend = (
                not any(
                    trace.name == name and trace.showlegend
                    for trace in figure.data
                )
            )
        elif layer == "balcony_segmentation":
            name = mesh["name"]
            showlegend = True
        else:
            name = layer.title()
            showlegend = layer not in shown_other_layers
            shown_other_layers.add(layer)
        metadata = mesh.get("metadata", {})
        if layer == "buildings":
            wireframe = mesh.get("wireframe", {})
            building_edge_x.extend(wireframe.get("line_x", []))
            building_edge_y.extend(wireframe.get("line_y", []))
            building_edge_z.extend(wireframe.get("line_z", []))
            building_corner_x.extend(wireframe.get("corner_x", []))
            building_corner_y.extend(wireframe.get("corner_y", []))
            building_corner_z.extend(wireframe.get("corner_z", []))
        hover_lines = [name, f"Mesh: {mesh['name']}"]
        for key in (
            "height_m",
            "height_source",
            "ground_elevation_m",
            "lidar_building_point_count",
            "lidar_building_inlier_count",
            "lidar_building_cluster_count",
            "lidar_dbscan_epsilon_m",
            "lidar_footprint_added_area_m2",
            "score",
            "bbox_xywh_pixels",
            "projected_pixel_count",
            "mesh_triangle_count",
        ):
            if key in metadata:
                hover_lines.append(f"{key.replace('_', ' ').title()}: {metadata[key]}")
        if metadata.get("is_target"):
            hover_lines.append(
                f"Matched target: {metadata.get('target_reference')}"
            )
        if layer == "trees" and not faces:
            figure.add_trace(go.Scatter3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                mode="markers",
                name=name,
                marker={"size": 2, "color": mesh["color"]},
                showlegend=showlegend,
                hoverinfo="name",
            ))
        else:
            figure.add_trace(go.Mesh3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                i=[face[0] for face in faces],
                j=[face[1] for face in faces],
                k=[face[2] for face in faces],
                name=name,
                color="#ff7043" if metadata.get("is_target") else mesh["color"],
                opacity=(
                    1.0 if layer == "buildings"
                    else 0.78 if layer == "balcony_segmentation"
                    else 0.95
                ),
                flatshading=True,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.8,
                    "specular": 0.15,
                    "roughness": 0.9,
                    "fresnel": 0.05,
                },
                lightposition={"x": 1000, "y": 1200, "z": 2000},
                showlegend=showlegend,
                hovertext="<br>".join(map(str, hover_lines)),
                hoverinfo="text",
            ))

    if building_edge_x:
        figure.add_trace(go.Scatter3d(
            x=building_edge_x,
            y=building_edge_y,
            z=building_edge_z,
            mode="lines",
            name="Building edges",
            line={"color": "#37474f", "width": 2},
            showlegend=True,
            hoverinfo="skip",
        ))
    if building_corner_x:
        figure.add_trace(go.Scatter3d(
            x=building_corner_x,
            y=building_corner_y,
            z=building_corner_z,
            mode="markers",
            name="Building vertices",
            marker={
                "size": 1.8,
                "color": "#263238",
                "opacity": 0.95,
            },
            showlegend=True,
            hoverinfo="skip",
        ))

    point_class_labels = {
        "ground": "Ground (class 2)",
        "low_vegetation": "Low vegetation (class 3)",
        "medium_vegetation": "Medium vegetation (class 4)",
        "high_vegetation": "High vegetation (class 5)",
        "building": "Building (class 6)",
        "other": "Other classifications",
    }
    for class_name, label in point_class_labels.items():
        points = lidar_point_samples.get(class_name, [])
        if not points:
            continue
        figure.add_trace(go.Scatter3d(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            z=[point[2] for point in points],
            mode="markers",
            name=f"LiDAR: {label}",
            marker={
                "size": 2,
                "color": fetch_lidar_point_cloud.POINT_CLASS_COLORS[class_name],
                "opacity": 0.75,
            },
            visible=(
                "legendonly"
                if (
                    class_name in {"low_vegetation", "medium_vegetation"}
                    and not vegetation_point_preview
                )
                else True
            ),
            showlegend=True,
            hovertemplate=(
                f"LiDAR {label}<br>East: %{{x:.1f}} m"
                "<br>North: %{y:.1f} m<br>Elevation: %{z:.1f} m"
                "<extra></extra>"
            ),
        ))

    center_bearing, center_elevation = building_data.pixel_to_bearing_elevation(
        float(crop["out_w"]) / 2.0,
        float(crop["out_h"]) / 2.0,
        crop,
    )
    camera_xyz = (0.0, 0.0, camera_height)

    def endpoint(
        bearing: float,
        elevation: float,
        distance: float,
    ) -> tuple[float, float, float]:
        return building_data.ray_endpoint(
            camera_xyz[0],
            camera_xyz[1],
            camera_xyz[2],
            bearing,
            elevation,
            distance,
        )

    frame_distance = min(
        60.0,
        max(10.0, float(crop.get("view_distance_m", 30.0))),
    )
    frame_corners = [
        endpoint(
            *building_data.pixel_to_bearing_elevation(x, y, crop),
            frame_distance,
        )
        for x, y in (
            (0.0, 0.0),
            (float(crop["out_w"]), 0.0),
            (float(crop["out_w"]), float(crop["out_h"])),
            (0.0, float(crop["out_h"])),
        )
    ]
    frustum_x: list[float | None] = []
    frustum_y: list[float | None] = []
    frustum_z: list[float | None] = []

    def add_frustum_segment(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> None:
        frustum_x.extend((start[0], end[0], None))
        frustum_y.extend((start[1], end[1], None))
        frustum_z.extend((start[2], end[2], None))

    for corner_index, corner in enumerate(frame_corners):
        next_corner = frame_corners[(corner_index + 1) % len(frame_corners)]
        add_frustum_segment(camera_xyz, corner)
        add_frustum_segment(corner, next_corner)
    figure.add_trace(go.Scatter3d(
        x=frustum_x,
        y=frustum_y,
        z=frustum_z,
        mode="lines",
        name="Camera field of view",
        line={"color": "#42a5f5", "width": 3},
        showlegend=True,
        hoverinfo="name",
    ))
    figure.add_trace(go.Scatter3d(
        x=[camera_xyz[0]],
        y=[camera_xyz[1]],
        z=[camera_xyz[2]],
        mode="markers+text",
        name="Estimated camera",
        text=["Camera"],
        textposition="top center",
        marker={"size": 7, "color": "#1565c0", "symbol": "diamond"},
        showlegend=True,
        hovertemplate="Estimated camera<br>Height: %{z:.1f} m<extra></extra>",
    ))

    hit = hit_record.get("hit", {}) if hit_record else {}
    if not isinstance(hit, dict):
        hit = {}
    bearing = (
        float(hit_record.get("bearing", center_bearing))
        if hit_record else center_bearing
    )
    elevation = (
        float(hit_record.get("elevation", center_elevation))
        if hit_record else center_elevation
    )
    distance = float(hit.get("distance_m") or frame_distance)
    target_xyz = endpoint(bearing, elevation, distance)
    target_reference = None
    if hit.get("osm_type") and hit.get("osm_id"):
        target_reference = f"{hit['osm_type']}/{hit['osm_id']}"
    target_found = any(
        mesh.get("metadata", {}).get("is_target") for mesh in meshes
    )
    target_label = (
        f"Target {target_reference or '(no OSM ID)'}"
        + (" — Overture match" if target_found else " — no Overture match")
    )
    figure.add_trace(go.Scatter3d(
        x=[camera_xyz[0], target_xyz[0]],
        y=[camera_xyz[1], target_xyz[1]],
        z=[camera_xyz[2], target_xyz[2]],
        mode="lines",
        name="Target tracker",
        line={"color": "#ff1744", "width": 5},
        showlegend=True,
        hovertemplate=(
            f"{target_label}<br>Bearing: {bearing:.1f}°"
            f"<br>Distance: {distance:.1f} m<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[target_xyz[0]],
        y=[target_xyz[1]],
        z=[target_xyz[2]],
        mode="markers+text",
        name="Target point",
        text=[target_label],
        textposition="top center",
        marker={"size": 7, "color": "#ff1744", "symbol": "x"},
        showlegend=True,
        hovertemplate=(
            f"{target_label}<br>Bearing: {bearing:.1f}°"
            f"<br>Distance: {distance:.1f} m<extra></extra>"
        ),
    ))

    figure.update_layout(
        title=(
            f"{crop.get('crop_name', 'Selected crop')} — USGS 3DEP terrain "
            "with Overture buildings (LiDAR-adjusted where supported) and "
            "camera/target tracker"
            "<br><sup>Local east/north metres; elevation relative to bare "
            "earth at the crop location. Terrain view may be decimated; OBJ "
            "retains the configured terrain mesh.</sup>"
        ),
        scene={
            "xaxis_title": "East (m from crop)",
            "yaxis_title": "North (m from crop)",
            "zaxis_title": "Elevation above crop ground (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.25}},
        },
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 0, "r": 0, "t": 100, "b": 15},
        hovermode="closest",
    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", help="Crop filename; defaults to a center crop.")
    parser.add_argument(
        "--manifest", type=Path,
        default=fetch_lidar_elevation.DEFAULT_MANIFEST,
    )
    parser.add_argument("--radius", type=float, default=500.0)
    parser.add_argument(
        "--hits",
        type=Path,
        default=BASE_DIR / "building_hits.json",
        help="Target-hit JSON produced by locate_building.py.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=building_data.DEFAULT_CAMERA_HEIGHT,
        help="Estimated camera height above local ground, in metres.",
    )
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Requested DEM output pixel size in metres.")
    parser.add_argument(
        "--terrain-max-dimension",
        type=int,
        default=DEFAULT_TERRAIN_MAX_DIMENSION,
        help="Maximum mesh grid width/height for terrain visualization/export.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--extra-meshes",
        type=Path,
        help="Optional JSON meshes (for future segmented balconies/trees).",
    )
    parser.add_argument("--overture-release",
                        help="Overture release; defaults to latest.")
    parser.add_argument(
        "--maximum-visualization-terrain-triangles",
        type=int,
        default=60_000,
        help="Decimate terrain only in the HTML view above this triangle count.",
    )
    parser.add_argument(
        "--max-lidar-visualization-points",
        type=int,
        default=fetch_lidar_point_cloud.DEFAULT_MAX_VISUALIZATION_POINTS_PER_CLASS,
        help=(
            "Maximum sampled points per LAS classification group in the HTML "
            "view; use 0 to omit point traces while still exporting clipped "
            "LAZ files."
        ),
    )
    parser.add_argument(
        "--minimum-shade-relevant-height",
        type=float,
        default=(
            fetch_lidar_point_cloud.DEFAULT_SHADE_RELEVANT_MIN_HEIGHT_METERS
        ),
        help=(
            "Minimum vegetation height above the DEM to include in tree "
            "meshes and the point-cloud preview, in metres."
        ),
    )
    parser.add_argument(
        "--vegetation-point-preview",
        action="store_true",
        help=(
            "Show sampled class 3-5 LiDAR returns as visible point traces in "
            "the HTML view; short vegetation remains filtered out."
        ),
    )
    parser.add_argument(
        "--segment-balconies",
        action="store_true",
        help=(
            "Run Grounding DINO and SAM on the selected perspective crop and "
            "project balcony masks onto visible building mesh surfaces."
        ),
    )
    parser.add_argument(
        "--grounding-dino-config",
        type=Path,
        default=(
            BASE_DIR.parent / "Grounding_DINO" / "grounding_dino_config.json"
        ),
        help="Grounding DINO/SAM model and prompt configuration JSON.",
    )
    parser.add_argument(
        "--segmentation-image",
        type=Path,
        help=(
            "Perspective crop image to segment; searches Gather_Balcony/"
            "perspective_crops/ and the repository-root perspective_crops/ "
            "by default."
        ),
    )
    args = parser.parse_args()
    if args.radius <= 0 or args.resolution <= 0 or args.camera_height <= 0:
        parser.error(
            "--radius, --resolution, and --camera-height must be greater "
            "than zero."
        )
    if args.maximum_visualization_terrain_triangles <= 0:
        parser.error("--maximum-visualization-terrain-triangles must be positive.")
    if args.max_lidar_visualization_points < 0:
        parser.error("--max-lidar-visualization-points cannot be negative.")
    if not (
        fetch_lidar_point_cloud.MIN_VEGETATION_HEIGHT_METERS
        <= args.minimum_shade_relevant_height
        <= fetch_lidar_point_cloud.MAX_CANOPY_HEIGHT_METERS
    ):
        parser.error(
            "--minimum-shade-relevant-height must be between "
            f"{fetch_lidar_point_cloud.MIN_VEGETATION_HEIGHT_METERS:g} and "
            f"{fetch_lidar_point_cloud.MAX_CANOPY_HEIGHT_METERS:g} metres."
        )

    manifest = fetch_lidar_elevation.load_manifest(args.manifest)
    crop = fetch_lidar_elevation.select_crop(manifest, args.crop)
    latitude = float(crop["lat"])
    longitude = float(crop["lng"])
    stem = Path(str(crop["crop_name"])).stem
    hit_record = None
    if args.hits.exists():
        hits = building_data.load_json(args.hits)
        if not isinstance(hits, list):
            raise ValueError(f"Expected a list of target records in {args.hits}.")
        hit_record = next(
            (
                item for item in hits
                if isinstance(item, dict)
                and item.get("crop_name") == crop.get("crop_name")
            ),
            None,
        )
    if hit_record is None:
        print(
            f"[WARN] No target hit for {crop.get('crop_name')} in {args.hits}; "
            "the crop center direction will be shown instead."
        )
    hit = hit_record.get("hit", {}) if hit_record else {}
    target_reference = (
        f"{hit['osm_type']}/{hit['osm_id']}"
        if isinstance(hit, dict) and hit.get("osm_type") and hit.get("osm_id")
        else None
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dem_path = output_dir / f"{stem}_3dep_bare_earth.tif"
    dem_metadata_path = dem_path.with_suffix(".metadata.json")
    scene_json_path = output_dir / f"{stem}_neighborhood_scene.json"
    obj_path = output_dir / f"{stem}_neighborhood_scene.obj"
    html_path = output_dir / f"{stem}_neighborhood_scene.html"
    segmentation_module = None
    segmentation_predictions: list[dict[str, Any]] = []
    segmentation_masks: list[np.ndarray] = []
    segmentation_metadata: dict[str, Any] = {"status": "not_requested"}
    if args.segment_balconies:
        if args.segmentation_image:
            image_path = args.segmentation_image
        else:
            crop_name = str(crop["crop_name"])
            image_candidates = (
                BASE_DIR / "perspective_crops" / crop_name,
                BASE_DIR.parent / "perspective_crops" / crop_name,
            )
            image_path = next(
                (path for path in image_candidates if path.is_file()),
                image_candidates[0],
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Perspective crop for segmentation was not found: "
                f"{image_path}. Supply --segmentation-image or create the "
                "perspective crops first."
            )
        if not args.grounding_dino_config.is_file():
            raise FileNotFoundError(
                "Grounding DINO configuration was not found: "
                f"{args.grounding_dino_config}"
            )
        if __package__:
            from . import balcony_segmentation
        else:
            import balcony_segmentation
        segmentation_module = balcony_segmentation
        print("Running Grounding DINO and SAM on the selected crop...")
        (
            segmentation_predictions,
            segmentation_masks,
            segmentation_metadata,
        ) = segmentation_module.infer_balcony_masks(
            image_path,
            args.grounding_dino_config,
            output_dir / f"{stem}_balcony_segmentation.png",
        )
        print(
            f"  DINO detections and SAM masks: "
            f"{len(segmentation_predictions)}."
        )

    print("Step 1/5: Fetching USGS 3DEP bare-earth elevation...")
    dem_metadata = fetch_lidar_elevation.export_dem(
        latitude, longitude, args.radius, args.resolution, dem_path
    )
    dem_metadata["crop_name"] = crop.get("crop_name")
    dem_metadata_path.write_text(
        json.dumps(dem_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Saved DEM: {dem_path}")

    print("Step 2/5: Fetching Overture building footprints and heights...")
    overture_features = building_data.fetch_overture_buildings(
        latitude, longitude, args.radius, args.overture_release
    )
    buildings, building_stats = building_data.overture_scene_buildings(
        overture_features,
        latitude,
        longitude,
        args.radius,
    )
    print(
        f"  Selected {building_stats['included']} full building footprints "
        f"intersecting the {args.radius:g} m scene radius."
    )
    print("Step 3/5: Creating terrain and checking USGS LiDAR coverage...")

    terrain, terrain_metadata = terrain_mesh(
        dem_path,
        latitude,
        longitude,
        args.terrain_max_dimension,
    )
    origin_elevation = terrain_metadata["scene_origin_bare_earth_elevation_m"]
    scene_area_polygon = [
        tuple(point)
        for point in terrain_metadata["aoi_boundary_local"]
    ]
    point_cloud_search_boundary = lidar_search_boundary(
        scene_area_polygon,
        buildings,
        latitude,
        longitude,
    )
    point_cloud_cache = output_dir / "point_cloud"
    point_cloud_paths, point_cloud_metadata = (
        fetch_lidar_point_cloud.find_lidar_tiles(
            latitude,
            longitude,
            point_cloud_search_boundary,
            point_cloud_cache,
        )
    )
    print(
        "  USGS LiDAR point-cloud coverage: "
        f"{point_cloud_metadata['coverage_fraction_from_tile_bounds']:.1%} "
        "of expanded search area by tile bounds; "
        f"{point_cloud_metadata['tile_count']} newest-project tiles."
    )
    lidar_points_dir = output_dir / "lidar_points"
    (
        canopy_meshes,
        canopy_metadata,
        lidar_point_samples,
    ) = fetch_lidar_point_cloud.create_canopy_mesh(
        point_cloud_paths,
        dem_path,
        latitude,
        longitude,
        lidar_points_dir,
        args.max_lidar_visualization_points,
        args.minimum_shade_relevant_height,
    )
    classified_points_metadata = canopy_metadata["classified_points"]
    print("  Classified LiDAR points within valid terrain cells:")
    for class_name, count in classified_points_metadata["counts"].items():
        print(f"    {class_name.replace('_', ' ').title()}: {count:,}")
    print(
        f"  Saved {len(classified_points_metadata['files'])} clipped, "
        f"class-separated LAZ files under {lidar_points_dir}."
    )
    if canopy_meshes:
        print(
            f"  Vegetation returns: "
            f"{canopy_metadata['vegetation_return_count']:,}; "
            f"{canopy_metadata['vegetation_cluster_count']:,} meshed "
            "vegetation clusters; "
            f"{canopy_metadata['mesh_triangle_count']:,} canopy triangles."
        )
    else:
        print(
            "  No canopy mesh produced: "
            f"{canopy_metadata.get('status', 'unknown status')}."
        )
    print("Step 4/5: Building buildings and assembling scene geometry...")
    scene_area_bounds = (
        min(point[0] for point in scene_area_polygon),
        min(point[1] for point in scene_area_polygon),
        max(point[0] for point in scene_area_polygon),
        max(point[1] for point in scene_area_polygon),
    )
    print(
        f"  {len(overture_features)} features downloaded; "
        f"{building_stats['included']} full footprints selected within the "
        "scene radius; "
        f"{building_stats['direct_heights']} direct heights; "
        f"{building_stats['estimated_heights']} height estimates; "
        f"{building_stats['fallback_heights']} fallback heights."
    )
    tile_vertical_offsets = {
        str(tile["file"]): float(tile["vertical_alignment_offset_m"])
        for tile in canopy_metadata.get("tiles", [])
        if isinstance(tile.get("vertical_alignment_offset_m"), (int, float))
    }
    building_lidar_adjustment = adjust_buildings_with_lidar(
        buildings,
        point_cloud_paths,
        tile_vertical_offsets,
        latitude,
        longitude,
        raster_path=dem_path,
        scene_area_polygon=scene_area_polygon,
    )
    print(
        "  LiDAR class-6 building adjustment: "
        f"{building_lidar_adjustment['adjusted_buildings']} Overture "
        "footprints adjusted; "
        f"{building_lidar_adjustment.get('clustered_inlier_point_count', 0):,} "
        "clustered points used; "
        f"{building_lidar_adjustment.get('lidar_only_building_count', 0)} "
        "unmapped LiDAR building clusters added."
    )
    (
        building_mesh_data,
        building_records,
        building_comparison_features,
    ) = building_meshes(
        buildings, dem_path, latitude, longitude, origin_elevation,
        target_reference, scene_area_polygon,
    )
    balcony_segmentation_meshes = []
    balcony_projection_records = []
    if segmentation_module is not None and segmentation_predictions:
        balcony_segmentation_meshes, balcony_projection_records = (
            segmentation_module.project_masks_to_buildings(
                segmentation_predictions,
                segmentation_masks,
                building_mesh_data,
                crop,
                args.camera_height,
                (int(crop["out_w"]), int(crop["out_h"])),
            )
        )
        segmentation_metadata["projection"] = {
            "surface_source": "nearest visible existing building-mesh faces",
            "raster_width_pixels": segmentation_module.PROJECTION_MAX_WIDTH,
            "surface_offset_toward_camera_m": (
                segmentation_module.SURFACE_OVERLAY_OFFSET_M
            ),
            "generated_mesh_count": len(balcony_segmentation_meshes),
            "detections": balcony_projection_records,
        }
        for detection, projection in zip(
            segmentation_metadata.get("detections", []),
            balcony_projection_records,
        ):
            detection["projection"] = projection
        print(
            f"  Projected {len(balcony_segmentation_meshes)} balcony masks "
            "onto building geometry."
        )
    elif segmentation_module is not None:
        segmentation_metadata["projection"] = {
            "surface_source": "nearest visible existing building-mesh faces",
            "generated_mesh_count": 0,
            "detections": [],
        }
    target_found = any(
        mesh.get("metadata", {}).get("is_target") for mesh in building_mesh_data
    )
    if target_reference:
        print(
            f"  Target {target_reference}: "
            f"{'matched and highlighted in Overture' if target_found else 'not matched in Overture'}."
        )
    print(
        f"  Camera tracker: origin (0, 0, {args.camera_height:g} m), "
        "crop field-of-view frustum, target bearing/range."
    )
    extra_mesh_data = load_extra_meshes(args.extra_meshes)
    meshes = [
        terrain,
        *building_mesh_data,
        *balcony_segmentation_meshes,
        *canopy_meshes,
        *extra_mesh_data,
    ]

    scene_metadata = {
        "schema_version": 1,
        "crop_name": crop.get("crop_name"),
        "origin": {
            "latitude": latitude,
            "longitude": longitude,
            "bare_earth_elevation_m": origin_elevation,
        },
        "coordinate_system": {
            "horizontal": "local east/north metres centered on crop",
            "vertical": "metres relative to bare-earth elevation at crop",
            "source_raster_crs": terrain_metadata["crs"],
        },
        "sources": {
            "terrain": dem_metadata,
            "buildings": {
                "dataset": "Overture Maps building theme",
                "feature_count_downloaded": len(overture_features),
                "feature_count_selected_within_radius": building_stats["included"],
                "local_aoi_bounds_m": scene_area_bounds,
                "geometry_note": (
                    "Buildings whose footprints intersect the requested "
                    "radius are rendered with their full Overture footprints. "
                    "Nearby USGS 3DEP class-6 returns are clustered in "
                    "east/north/elevation coordinates. Simplified concave "
                    "cluster outlines use the 95th-percentile inlier "
                    "elevation; higher roof regions take precedence where "
                    "they overlap. Nearby roof regions at similar heights "
                    "are unioned, while uncovered mapped footprint areas "
                    "retain their Overture extrusion height. Unmatched "
                    "class-6 clusters are constructed as buildings in the "
                    "same exported layer."
                ),
                "comparison_features": building_comparison_features,
                "comparison_geometry_note": (
                    "Local east/north Overture footprints and the union of "
                    "detected USGS class-6 roof-region footprints. These are "
                    "independent local inputs for comparison; they do not "
                    "contain or derive geometry from Google Photorealistic "
                    "3D Tiles. LiDAR roof-region coverage may be incomplete."
                ),
                "roof_geometry_cleanup": {
                    "lidar_footprint_method": "Shapely concave hull",
                    "concave_hull_ratio": LIDAR_FOOTPRINT_CONCAVE_RATIO,
                    "simplify_tolerance_m": (
                        LIDAR_FOOTPRINT_SIMPLIFY_METERS
                    ),
                    "roof_elevation_quantile": (
                        LIDAR_ROOF_ELEVATION_QUANTILE
                    ),
                    "merge_roof_levels_within_m": (
                        ROOF_LEVEL_MERGE_TOLERANCE_M
                    ),
                    "roof_triangulation": (
                        "Constrained Delaunay where available, with "
                        "polygon-filtered Delaunay fallback."
                    ),
                },
                "lidar_adjustment": building_lidar_adjustment,
                "lidar_only_candidate_filter": {
                    "minimum_height_above_ground_m": (
                        MIN_LIDAR_ONLY_BUILDING_HEIGHT_M
                    ),
                    "minimum_cluster_points": (
                        MIN_LIDAR_ONLY_BUILDING_POINTS
                    ),
                    "minimum_footprint_area_m2": (
                        MIN_LIDAR_ONLY_BUILDING_AREA_M2
                    ),
                    "matched_footprint_exclusion_distance_m": (
                        LIDAR_BUILDING_MATCH_DISTANCE_M
                    ),
                    "footprint_overlap_suppression_fraction": 0.5,
                    "dbscan": {
                        "minimum_samples": 5,
                        "epsilon_m": (
                            "Twice the mean distance to the three nearest "
                            "horizontal neighbors, constrained to 1-2 m."
                        ),
                        "cell_size_m": (
                            fetch_lidar_point_cloud.VEGETATION_CLUSTER_CELL_SIZE_METERS
                        ),
                        "cell_buffer_m": (
                            fetch_lidar_point_cloud.VEGETATION_CLUSTER_CELL_BUFFER_METERS
                        ),
                        "maximum_fit_points_per_cell": (
                            fetch_lidar_point_cloud.MAX_DBSCAN_POINTS_PER_CELL_CLASS
                        ),
                    },
                    "note": (
                        "Class-6 points assigned to mapped footprints are "
                        "excluded from unmatched-cluster construction to "
                        "avoid duplicate nearby shells."
                    ),
                },
                "lidar_only_candidate_count": (
                    building_lidar_adjustment.get(
                        "lidar_only_building_count", 0
                    )
                ),
                "lidar_search_boundary_local": point_cloud_search_boundary,
                "features": building_records,
            },
            "balconies": {
                "source": str(args.extra_meshes) if args.extra_meshes else None,
                "mesh_count": sum(
                    mesh["layer"] == "balconies" for mesh in extra_mesh_data
                ),
            },
            "balcony_detection_segmentation": {
                **segmentation_metadata,
                "mesh_count": len(balcony_segmentation_meshes),
                "mesh_layer": "balcony_segmentation",
            },
            "trees": {
                "source": point_cloud_metadata["source"],
                "coverage": point_cloud_metadata,
                "processing": canopy_metadata,
                "mesh_count": (
                    sum(mesh["layer"] == "trees" for mesh in extra_mesh_data)
                    + len(canopy_meshes)
                ),
            },
            "lidar_points": {
                "source": point_cloud_metadata["source"],
                "processing": classified_points_metadata,
                "export_directory": str(lidar_points_dir),
                "visualization": (
                    "Deterministic random sample of clipped points, separated "
                    "by LAS classification group. Toggle each group in the "
                    "HTML legend."
                ),
            },
            "additional_meshes": [
                {
                    "name": mesh["name"],
                    "layer": mesh["layer"],
                    "vertex_count": len(mesh["vertices"]),
                    "triangle_count": len(mesh["faces"]),
                    "metadata": mesh.get("metadata", {}),
                }
                for mesh in extra_mesh_data
            ],
        },
        "terrain_mesh": terrain_metadata,
        "camera_tracker": {
            "camera_height_m": args.camera_height,
            "target_reference": target_reference,
            "target_matched_to_overture": target_found,
            "target_bearing_deg": (
                hit_record.get("bearing") if hit_record else None
            ),
            "target_elevation_deg": (
                hit_record.get("elevation") if hit_record else None
            ),
            "target_distance_m": (
                hit.get("distance_m") if isinstance(hit, dict) else None
            ),
        },
        "export": {
            "obj": obj_path.name,
            "html": html_path.name,
            "terrain_mesh_grid": terrain_metadata["mesh_grid"],
            "units": "metres",
        },
    }

    print(
        f"  Terrain: {len(terrain['vertices']):,} vertices, "
        f"{len(terrain['faces']):,} triangles; "
        f"{len(building_mesh_data)} building meshes; "
        f"{sum(mesh['layer'] == 'balconies' for mesh in extra_mesh_data)} "
        "extra balcony meshes; "
        f"{len(balcony_segmentation_meshes)} DINO/SAM balcony surface overlays; "
        f"{sum(mesh['layer'] == 'trees' for mesh in extra_mesh_data) + len(canopy_meshes)} "
        "tree meshes."
    )
    print("Step 5/5: Exporting Rhino mesh and verification visualization...")
    write_obj(meshes, obj_path)
    figure = make_scene_figure(
        meshes,
        crop,
        args.maximum_visualization_terrain_triangles,
        args.camera_height,
        hit_record,
        lidar_point_samples,
        args.vegetation_point_preview,
    )
    html_path.write_text(
        figure.to_html(
            full_html=True,
            include_plotlyjs=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["zoomIn3d", "zoomOut3d"],
            },
        ),
        encoding="utf-8",
    )
    scene_json_path.write_text(
        json.dumps(scene_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  Rhino-importable OBJ: {obj_path}")
    print(f"  Interactive verification view: {html_path}")
    print(f"  Scene metadata: {scene_json_path}")
    print(
        "  Vegetation meshes are class-specific DBSCAN cluster surfaces. "
        "Balcony segmentation is opt-in with --segment-balconies; manual "
        "meshes can also be supplied with --extra-meshes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
