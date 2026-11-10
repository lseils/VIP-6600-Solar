#!/usr/bin/env python3
"""Create an interactive 3D view of buildings and an estimated camera.

Run locate_building.py first. The visualization uses crop_manifest.json for
the camera/view estimate and building_hits.json to identify the target OSM
building in its default mode. Nearby footprints are extruded using mapped
heights or clearly labeled estimates. The optional Overture LiDAR-height mode
does not query OSM. An available perspective crop (or source panorama) is
embedded as a reference image; it is not projected onto the 3D geometry.


If you get a 502 error from running the script, just run the script again.

"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parent
EARTH_RADIUS_METERS = 6_371_000.0
DEFAULT_CAMERA_HEIGHT = 2.5
DEFAULT_BUILDING_HEIGHT = 9.0
DEFAULT_FLOOR_HEIGHT = 3.0
DEFAULT_ROOF_LEVEL_HEIGHT = 2.5
DEFAULT_RADIUS_METERS = 500.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERTURE_TIMEOUT_SECONDS = 180


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def local_xy(latitude: float, longitude: float, origin_latitude: float,
             origin_longitude: float) -> tuple[float, float]:
    """Project a small local area to east/north metres around the camera."""
    east = math.radians(longitude - origin_longitude) * EARTH_RADIUS_METERS
    east *= math.cos(math.radians(origin_latitude))
    north = math.radians(latitude - origin_latitude) * EARTH_RADIUS_METERS
    return east, north


def pixel_to_bearing_elevation(x: float, y: float,
                               crop: dict[str, Any]) -> tuple[float, float]:
    """Convert crop pixels to compass bearing and elevation."""
    width = float(crop["out_w"])
    height = float(crop["out_h"])
    focal = (width / 2.0) / math.tan(math.radians(float(crop["fov_deg"])) / 2.0)
    rx = (x - width / 2.0) / focal
    ry = -(y - height / 2.0) / focal
    rz = 1.0
    yaw = math.radians(float(crop["yaw_deg"]))
    pitch = math.radians(float(crop["pitch_deg"]))

    # Rotation 1: pitch around the camera's X axis (positive = look up)
    ry2 =  ry * math.cos(pitch) + rz * math.sin(pitch)
    rz2 = -ry * math.sin(pitch) + rz * math.cos(pitch)

    # Rotation 2: yaw around the vertical axis
    rx3 =  rx * math.cos(yaw) + rz2 * math.sin(yaw)
    rz3 = -rx * math.sin(yaw) + rz2 * math.cos(yaw)

    longitude = math.atan2(rx3, rz3)
    latitude = math.atan2(ry2, math.hypot(rx3, rz3))
    bearing = (float(crop["car_heading"]) + math.degrees(longitude)) % 360.0
    return bearing, math.degrees(latitude)


def parse_length_meters(value: Any) -> float | None:
    """Parse common OSM length values, returning metres when the value is positive."""
    if value is None:
        return None
    text = str(value).strip().lower()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    number = float(match.group())
    if number <= 0:
        return None
    if "ft" in text or "'" in text:
        number *= 0.3048
    elif "cm" in text:
        number /= 100.0
    elif "mm" in text:
        number /= 1000.0
    return number


def parse_positive_number(value: Any) -> float | None:
    """Parse a positive numeric OSM count tag."""
    if value is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(value).strip())
    if not match:
        return None
    number = float(match.group())
    return number if number > 0 else None


def parse_height(tags: dict[str, Any]) -> tuple[float, str]:
    """Estimate total height from OSM height, floor, and roof tags.

    Explicit total heights take priority. Otherwise, above-ground floor counts
    are converted using a 3 m/floor estimate, with mapped roof height or roof
    levels added. Offset tags are included where available. If only roof or
    offset information exists, the total still needs a labeled base-height
    estimate.
    """
    for key in ("height", "building:height"):
        height = parse_length_meters(tags.get(key))
        if height is not None:
            source = "OSM height" if key == "height" else "OSM building:height"
            return height, source

    roof_height = parse_length_meters(tags.get("roof:height"))
    roof_levels = parse_positive_number(tags.get("roof:levels"))
    roof_addition = (
        roof_height if roof_height is not None
        else roof_levels * DEFAULT_ROOF_LEVEL_HEIGHT if roof_levels is not None
        else 0.0
    )

    offset = parse_length_meters(tags.get("min_height"))
    if offset is None:
        min_level = parse_positive_number(tags.get("building:min_level"))
        if min_level is not None:
            offset = min_level * DEFAULT_FLOOR_HEIGHT
    offset = offset or 0.0

    levels = parse_positive_number(tags.get("building:levels"))
    if levels is not None:
        height = (
            offset
            + levels * DEFAULT_FLOOR_HEIGHT
            + roof_addition
        )
        sources = ["OSM levels × 3 m estimate"]
        if tags.get("min_height") and offset:
            sources.append("OSM min_height")
        elif tags.get("building:min_level") and offset:
            sources.append("OSM building:min_level × 3 m estimate")
        if roof_height is not None:
            sources.append("OSM roof:height")
        elif roof_levels is not None:
            sources.append("OSM roof:levels × 2.5 m estimate")
        return height, " + ".join(sources)

    if roof_addition or offset:
        height = DEFAULT_BUILDING_HEIGHT + offset + roof_addition
        sources = [f"default base estimate ({DEFAULT_BUILDING_HEIGHT:g} m)"]
        if offset:
            sources.append("OSM min_height" if tags.get("min_height")
                           else "OSM building:min_level × 3 m estimate")
        if roof_height is not None:
            sources.append("OSM roof:height")
        elif roof_levels is not None:
            sources.append("OSM roof:levels × 2.5 m estimate")
        return height, " + ".join(sources)

    return DEFAULT_BUILDING_HEIGHT, "default estimate (9 m)"


def parse_explicit_osm_height(tags: dict[str, Any]) -> tuple[float, str] | None:
    for key in ("height", "building:height"):
        height = parse_length_meters(tags.get(key))
        if height is not None:
            source = "OSM height" if key == "height" else "OSM building:height"
            return height, source
    return None


def parse_overture_height(feature: dict[str, Any]) -> tuple[float, str] | None:
    """Extract Overture's explicit height or estimate from its height fields."""
    properties = feature.get("properties", {})
    height = parse_positive_number(properties.get("height"))
    sources = properties.get("sources") or []
    height_sources = [
        source for source in sources
        if isinstance(source, dict)
        and source.get("property") == "/properties/height"
    ]
    datasets = sorted({
        str(source["dataset"])
        for source in height_sources
        if source.get("dataset")
    })
    confidences = [
        source["confidence"]
        for source in height_sources
        if isinstance(source.get("confidence"), (int, float))
        and math.isfinite(source["confidence"])
    ]
    if height is not None:
        provenance = ", ".join(datasets) if datasets else "source unspecified"
        confidence = (
            f", confidence {max(confidences):.2f}" if confidences else ""
        )
        return height, f"Overture height ({provenance}{confidence})"

    floors = parse_positive_number(properties.get("num_floors"))
    roof_height = parse_positive_number(properties.get("roof_height"))
    if floors is not None:
        estimate = floors * DEFAULT_FLOOR_HEIGHT + (roof_height or 0.0)
        return estimate, "Overture floors × 3 m estimate"

    min_height = parse_positive_number(properties.get("min_height"))
    if min_height is not None or roof_height is not None:
        estimate = (
            DEFAULT_BUILDING_HEIGHT
            + (min_height or 0.0)
            + (roof_height or 0.0)
        )
        return estimate, "Overture height tags + 9 m base estimate"
    return None


def overture_lidar_height(feature: dict[str, Any]) -> tuple[float, str] | None:
    """Return a direct Overture height explicitly attributed to USGS LiDAR."""
    properties = feature.get("properties") or {}
    height = parse_positive_number(properties.get("height"))
    if height is None:
        return None
    sources = [
        source for source in (properties.get("sources") or [])
        if isinstance(source, dict)
        and source.get("property") == "/properties/height"
        and isinstance(source.get("dataset"), str)
        and "usgs" in source["dataset"].casefold()
        and "lidar" in source["dataset"].casefold()
    ]
    if not sources:
        return None
    dataset_names = sorted({str(source["dataset"]) for source in sources})
    confidences = [
        source["confidence"]
        for source in sources
        if isinstance(source.get("confidence"), (int, float))
        and math.isfinite(source["confidence"])
    ]
    provenance = ", ".join(dataset_names)
    if confidences:
        provenance += f", confidence {max(confidences):.2f}"
    return height, f"USGS LiDAR height ({provenance})"


def overture_bbox(latitude: float, longitude: float,
                  radius_meters: float) -> tuple[float, float, float, float]:
    latitude_delta = radius_meters / 111_320.0
    longitude_delta = latitude_delta / math.cos(math.radians(latitude))
    return (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )


def fetch_overture_buildings(latitude: float, longitude: float,
                             radius_meters: float,
                             release: str | None = None) -> list[dict[str, Any]]:
    """Download only the Overture building features within the requested bbox."""
    west, south, east, north = overture_bbox(latitude, longitude, radius_meters)
    with tempfile.TemporaryDirectory(prefix="vip6600-overture-") as temp_dir:
        output_path = Path(temp_dir) / "buildings.geojsonl"
        command = [
            sys.executable, "-m", "overturemaps", "download",
            f"--bbox={west:.8f},{south:.8f},{east:.8f},{north:.8f}",
            "-f", "geojsonseq",
            "-t", "building",
            "-o", str(output_path),
            "--connect_timeout", "15",
            "--request_timeout", "60",
        ]
        if release:
            command.extend(["--release", release])
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=OVERTURE_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Overture Maps client is not installed. Install it with "
                "`python -m pip install overturemaps` or install the project "
                "requirements."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Overture download exceeded {OVERTURE_TIMEOUT_SECONDS} seconds "
                "for the requested area."
            ) from error
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or "").strip()
            if "No module named overturemaps" in details:
                python_command = shlex.quote(sys.executable)
                raise RuntimeError(
                    "The Overture Maps client is not installed in the Python "
                    f"environment used by this script ({sys.executable}). "
                    f"Install it with `{python_command} -m pip install "
                    "overturemaps==1.0.2` and rerun the script."
                ) from error
            raise RuntimeError(
                f"Overture building download failed (exit {error.returncode}): "
                f"{details or 'no error details returned'}"
            ) from error

        if result.stderr.strip():
            print(f"Overture: {result.stderr.strip().splitlines()[-1]}")
        if not output_path.exists():
            return []
        with output_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]


def overture_osm_reference_ids(feature: dict[str, Any]) -> set[tuple[str, str]]:
    """Read OSM element type/ID references attached to an Overture feature."""
    references = set()
    for source in feature.get("properties", {}).get("sources") or []:
        if not isinstance(source, dict):
            continue
        if source.get("dataset") != "OpenStreetMap":
            continue
        record_id = source.get("record_id")
        if not isinstance(record_id, str):
            continue
        match = re.match(r"^([nwr])(\d+)(?:@|$)", record_id)
        if not match:
            continue
        osm_type = {"n": "node", "w": "way", "r": "relation"}[match.group(1)]
        references.add((osm_type, match.group(2)))
    return references


def apply_overture_heights(buildings: list[dict[str, Any]],
                           features: list[dict[str, Any]]) -> tuple[int, int]:
    """Attach Overture height data only to its explicitly referenced OSM feature."""
    buildings_by_id = {
        (building["osm_type"], building["osm_id"]): building
        for building in buildings
    }
    matched = set()
    height_count = 0
    for feature in features:
        references = overture_osm_reference_ids(feature)
        targets = {
            key: buildings_by_id[key]
            for key in references if key in buildings_by_id
        }
        if not targets:
            continue
        matched.update(targets)
        overture_height_result = parse_overture_height(feature)
        if overture_height_result is None:
            continue
        height, source = overture_height_result
        for building in targets.values():
            if parse_explicit_osm_height(building["tags"]) is not None:
                continue
            building["overture_height_m"] = height
            building["overture_height_source"] = source
            height_count += 1
    return len(matched), height_count


def building_height(building: dict[str, Any]) -> tuple[float, str]:
    """Prefer explicit OSM, then Overture, then OSM tag estimates, then default."""
    explicit_osm = parse_explicit_osm_height(building["tags"])
    if explicit_osm is not None:
        return explicit_osm
    if building.get("overture_height_m") is not None:
        return (
            float(building["overture_height_m"]),
            str(building["overture_height_source"]),
        )
    return parse_height(building["tags"])


def merge_overture_footprints(
    buildings: list[dict[str, Any]],
    features: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    radius_meters: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Append Overture footprints not represented by OSM within the radius."""
    try:
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union
    except ImportError as error:
        raise RuntimeError(
            "Spatial comparison of Overture and OSM footprints requires "
            "Shapely. Install the project requirements with "
            "`python -m pip install -r ../requirements.txt`."
        ) from error

    def local_polygon(ring: list[tuple[float, float]]) -> Any:
        points = [
            local_xy(point_lat, point_lon, latitude, longitude)
            for point_lat, point_lon in ring
        ]
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon

    osm_keys = {
        (building["osm_type"], building["osm_id"])
        for building in buildings
    }
    osm_polygons = [
        local_polygon(ring)
        for building in buildings
        for ring in building.get("rings", [])
    ]
    osm_polygons = [
        polygon for polygon in osm_polygons
        if not polygon.is_empty and polygon.area > 0
    ]
    added: list[dict[str, Any]] = []
    stats = {
        "outside_radius": 0,
        "matched_by_osm_id": 0,
        "duplicate_geometry": 0,
        "missing_geometry": 0,
        "invalid_geometry": 0,
        "added": 0,
    }
    seen_overture_ids: set[str] = set()

    for feature in features:
        properties = feature.get("properties") or {}
        feature_id = str(feature.get("id") or "")
        references = overture_osm_reference_ids(feature)
        if any(reference in osm_keys for reference in references):
            stats["matched_by_osm_id"] += 1
            continue
        if feature_id and feature_id in seen_overture_ids:
            continue
        if feature_id:
            seen_overture_ids.add(feature_id)

        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if geometry_type == "Polygon":
            polygon_rings = [coordinates[0]] if coordinates else []
        elif geometry_type == "MultiPolygon":
            polygon_rings = [
                polygon[0] for polygon in coordinates if polygon
            ]
        else:
            polygon_rings = []
        rings: list[list[tuple[float, float]]] = []
        for coordinate_ring in polygon_rings:
            ring = []
            for coordinate in coordinate_ring:
                if (
                    not isinstance(coordinate, (list, tuple))
                    or len(coordinate) < 2
                ):
                    continue
                lon, lat = coordinate[0], coordinate[1]
                if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                    continue
                if not math.isfinite(lon) or not math.isfinite(lat):
                    continue
                ring.append((float(lat), float(lon)))
            if len(ring) >= 4 and ring[0] == ring[-1]:
                rings.append(ring)
        if not rings:
            stats["missing_geometry"] += 1
            continue

        overture_polygons = [
            local_polygon(ring) for ring in rings
        ]
        overture_polygons = [
            polygon for polygon in overture_polygons
            if not polygon.is_empty and polygon.area > 0
        ]
        if not overture_polygons:
            stats["missing_geometry"] += 1
            continue
        valid_rings = []
        for ring in rings:
            local_points = [
                local_xy(point_lat, point_lon, latitude, longitude)
                for point_lat, point_lon in ring
            ]
            try:
                triangulate_ring(local_points)
            except ValueError:
                stats["invalid_geometry"] += 1
                continue
            valid_rings.append(ring)
        if not valid_rings:
            continue
        rings = valid_rings
        overture_polygons = [
            local_polygon(ring) for ring in rings
        ]
        overture_shape = unary_union(overture_polygons)
        if overture_shape.distance(Point(0.0, 0.0)) > radius_meters:
            stats["outside_radius"] += 1
            continue

        # A substantial shared footprint is treated as a duplicate even when
        # Overture's source metadata does not include an OSM element ID.
        if osm_polygons:
            is_duplicate = any(
                min(
                    overture_shape.intersection(osm_polygon).area
                    / min(overture_shape.area, osm_polygon.area),
                    1.0,
                ) >= 0.5
                for osm_polygon in osm_polygons
                if overture_shape.intersects(osm_polygon)
            )
            if is_duplicate:
                stats["duplicate_geometry"] += 1
                continue

        height_result = parse_overture_height(feature)
        building = {
            "osm_type": "Overture",
            "osm_id": feature_id or "unknown",
            "tags": {},
            "rings": rings,
            "geometry_source": "Overture Maps",
            "overture_height_m": height_result[0] if height_result else None,
            "overture_height_source": height_result[1] if height_result else None,
            "overture_properties": properties,
        }
        added.append(building)
        stats["added"] += 1

    buildings.extend(added)
    return added, stats


def overture_scene_buildings(
    features: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    radius_meters: float,
    lidar_only: bool = False,
    square_area: bool = False,
    area_bounds: tuple[float, float, float, float] | None = None,
    area_polygon: list[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a scene using only Overture footprints and height attributes.

    The default filter retains full footprints that intersect a circular
    radius. Set ``square_area``, ``area_bounds``, or ``area_polygon`` to select
    and clip footprints to a bounded extent.
    """
    try:
        from shapely.geometry import Point, Polygon, box
        from shapely.ops import unary_union
    except ImportError as error:
        raise RuntimeError(
            "The Overture-only comparison requires Shapely. Install the project "
            "requirements with `python -m pip install -r ../requirements.txt`."
        ) from error

    stats = {
        "without_height": 0,
        "missing_geometry": 0,
        "outside_radius": 0,
        "invalid_geometry": 0,
        "included": 0,
        "direct_heights": 0,
        "estimated_heights": 0,
        "fallback_heights": 0,
    }
    buildings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for feature in features:
        height_result = (
            overture_lidar_height(feature)
            if lidar_only else parse_overture_height(feature)
        )
        if height_result is None:
            stats["without_height"] += 1
            if lidar_only:
                continue

        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if geometry_type == "Polygon":
            polygon_rings = [coordinates[0]] if coordinates else []
        elif geometry_type == "MultiPolygon":
            polygon_rings = [polygon[0] for polygon in coordinates if polygon]
        else:
            polygon_rings = []

        rings: list[list[tuple[float, float]]] = []
        polygons = []
        invalid_count = 0
        for coordinates_ring in polygon_rings:
            ring = []
            for coordinate in coordinates_ring:
                if (
                    not isinstance(coordinate, (list, tuple))
                    or len(coordinate) < 2
                ):
                    continue
                lon, lat = coordinate[0], coordinate[1]
                if (
                    not isinstance(lon, (int, float))
                    or not isinstance(lat, (int, float))
                    or not math.isfinite(lon)
                    or not math.isfinite(lat)
                ):
                    continue
                ring.append((float(lat), float(lon)))
            if len(ring) < 4 or ring[0] != ring[-1]:
                invalid_count += 1
                continue
            local_points = [
                local_xy(lat, lon, latitude, longitude) for lat, lon in ring
            ]
            try:
                triangulate_ring(local_points)
            except ValueError:
                invalid_count += 1
                continue
            polygon = Polygon(local_points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                invalid_count += 1
                continue
            rings.append(ring)
            polygons.append(polygon)

        stats["invalid_geometry"] += invalid_count
        if not polygons:
            stats["missing_geometry"] += 1
            continue
        footprint = unary_union(polygons)
        area = Polygon(area_polygon) if area_polygon is not None else (
            box(*area_bounds)
            if area_bounds is not None
            else (
                box(-radius_meters, -radius_meters, radius_meters, radius_meters)
                if square_area else None
            )
        )
        outside_area = (
            not footprint.intersects(area)
            if area is not None
            else footprint.distance(Point(0.0, 0.0)) > radius_meters
        )
        if outside_area:
            stats["outside_radius"] += 1
            continue
        if area is not None:
            clipped_footprint = footprint.intersection(area)
            clipped_rings = []
            clipped_polygons = (
                list(clipped_footprint.geoms)
                if hasattr(clipped_footprint, "geoms")
                else [clipped_footprint]
            )
            for clipped_polygon in clipped_polygons:
                if clipped_polygon.geom_type != "Polygon":
                    continue
                clipped_ring = []
                for east, north in clipped_polygon.exterior.coords:
                    clipped_ring.append((
                        latitude + math.degrees(
                            north / EARTH_RADIUS_METERS
                        ),
                        longitude + math.degrees(
                            east / (
                                EARTH_RADIUS_METERS
                                * math.cos(math.radians(latitude))
                            )
                        ),
                    ))
                if len(clipped_ring) >= 4:
                    clipped_rings.append(clipped_ring)
            if not clipped_rings:
                stats["invalid_geometry"] += 1
                continue
            rings = clipped_rings
        feature_id = str(feature.get("id") or "")
        if feature_id and feature_id in seen_ids:
            continue
        if feature_id:
            seen_ids.add(feature_id)
        buildings.append({
            "osm_type": "Overture",
            "osm_id": feature_id or f"overture-{len(buildings) + 1}",
            "tags": {},
            "rings": rings,
            "geometry_source": (
                "Overture Maps footprint "
                "(geometry source may include OpenStreetMap)"
                + ("; clipped to scene AOI" if area is not None else "")
            ),
            "osm_references": sorted(
                f"{osm_type}/{osm_id}"
                for osm_type, osm_id in overture_osm_reference_ids(feature)
            ),
            "overture_height_m": height_result[0] if height_result else None,
            "overture_height_source": height_result[1] if height_result else None,
        })
        stats["included"] += 1
        if height_result is None:
            stats["fallback_heights"] += 1
        elif height_result[1].startswith(
            ("Overture height (", "USGS LiDAR height (")
        ):
            stats["direct_heights"] += 1
        else:
            stats["estimated_heights"] += 1

    return buildings, stats


def overpass_buildings(endpoint: str, latitude: float, longitude: float,
                       radius: float, target_id: str | None = None,
                       target_type: str | None = None) -> list[dict[str, Any]]:
    target_query = ""
    if target_id and target_id.isdigit():
        if target_type in {"way", "relation"}:
            target_query = f"\n  {target_type}({target_id});"
        else:
            # Legacy building_hits.json files omit the OSM element type. OSM
            # IDs can collide between ways and relations, so try both.
            target_query = f"\n  way({target_id});\n  relation({target_id});"
    query = f"""
[out:json][timeout:60];
(
  way["building"](around:{radius},{latitude},{longitude});
  relation["building"](around:{radius},{latitude},{longitude});
  {target_query}
);
out tags geom;
"""
    request = Request(
        endpoint,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "VIP-6600-Solar-building-camera-visualizer/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read())

    buildings = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        if not tags.get("building"):
            continue
        rings: list[list[tuple[float, float]]] = []

        # Plain ways: the geometry is a single closed ring.
        geometry = element.get("geometry", [])
        if geometry:
            ring = [(float(point["lat"]), float(point["lon"]))
                    for point in geometry if "lat" in point and "lon" in point]
            if len(ring) >= 4:
                rings.append(ring)

        # Relation outer rings may be split across multiple member ways.
        segments: list[list[tuple[float, float]]] = []
        for member in element.get("members", []):
            if member.get("role") != "outer":
                continue
            member_geometry = member.get("geometry", [])
            coords = [(float(point["lon"]), float(point["lat"]))
                      for point in member_geometry
                      if "lat" in point and "lon" in point]
            if len(coords) < 2:
                continue
            segments.append(coords)

        while segments:
            ring = segments.pop(0)
            while ring[0] != ring[-1]:
                for index, segment in enumerate(segments):
                    if ring[-1] == segment[0]:
                        ring.extend(segment[1:])
                    elif ring[-1] == segment[-1]:
                        ring.extend(reversed(segment[:-1]))
                    elif ring[0] == segment[-1]:
                        ring = segment[:-1] + ring
                    elif ring[0] == segment[0]:
                        ring = list(reversed(segment[1:])) + ring
                    else:
                        continue
                    del segments[index]
                    break
                else:
                    break
            if len(ring) >= 4 and ring[0] == ring[-1]:
                rings.append([(lat, lon) for lon, lat in ring])

        if rings:
            buildings.append({
                "osm_type": element.get("type", "way"),
                "osm_id": str(element.get("id")),
                "tags": tags,
                "rings": rings,
            })
    return buildings


def osm_api_target_building(target_id: str, target_type: str | None,
                            timeout: float = 20.0) -> dict[str, Any] | None:
    """Fetch a target building directly, including relation member geometry."""
    element_types = [target_type] if target_type in {"way", "relation"} else ["relation", "way"]
    for element_type in element_types:
        url = f"https://api.openstreetmap.org/api/0.6/{element_type}/{target_id}/full.json"
        request = Request(
            url,
            headers={"User-Agent": "VIP-6600-Solar-building-camera-visualizer/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())

        elements = payload.get("elements", [])
        element = next(
            (item for item in elements
             if item.get("type") == element_type
             and str(item.get("id")) == target_id),
            None,
        )
        if element is None:
            continue
        tags = element.get("tags", {})
        if not tags.get("building"):
            continue

        nodes = {
            int(item["id"]): (float(item["lat"]), float(item["lon"]))
            for item in elements
            if item.get("type") == "node"
            and "lat" in item and "lon" in item
        }
        ways = {
            int(item["id"]): item
            for item in elements
            if item.get("type") == "way"
        }
        rings: list[list[tuple[float, float]]] = []

        if element_type == "way":
            coordinates = [
                nodes[ref] for ref in element.get("nodes", []) if ref in nodes
            ]
            if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
                rings.append(coordinates)
        else:
            segments = [
                [int(ref) for ref in ways[int(member["ref"])].get("nodes", [])]
                for member in element.get("members", [])
                if member.get("type") == "way"
                and member.get("role") == "outer"
                and int(member["ref"]) in ways
                and len(ways[int(member["ref"])].get("nodes", [])) >= 2
            ]
            while segments:
                ring = segments.pop(0)
                while ring[0] != ring[-1]:
                    for index, segment in enumerate(segments):
                        if ring[-1] == segment[0]:
                            ring.extend(segment[1:])
                        elif ring[-1] == segment[-1]:
                            ring.extend(reversed(segment[:-1]))
                        elif ring[0] == segment[-1]:
                            ring = segment[:-1] + ring
                        elif ring[0] == segment[0]:
                            ring = list(reversed(segment[1:])) + ring
                        else:
                            continue
                        del segments[index]
                        break
                    else:
                        break
                if len(ring) >= 4 and ring[0] == ring[-1]:
                    coordinates = [nodes[ref] for ref in ring if ref in nodes]
                    if len(coordinates) >= 4:
                        rings.append(coordinates)

        if rings:
            return {
                "osm_type": element_type,
                "osm_id": target_id,
                "tags": tags,
                "rings": rings,
            }
    return None


def polygon_area(points: list[tuple[float, float]]) -> float:
    return sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ) / 2.0


def cross(a: tuple[float, float], b: tuple[float, float],
          c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_triangle(point: tuple[float, float], a: tuple[float, float],
                      b: tuple[float, float], c: tuple[float, float],
                      orientation: float) -> bool:
    epsilon = 1e-9
    return (
        orientation * cross(a, b, point) >= -epsilon
        and orientation * cross(b, c, point) >= -epsilon
        and orientation * cross(c, a, point) >= -epsilon
    )


def triangulate_ring(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clip a simple OSM footprint ring for a flat roof surface."""
    ring = points[:-1] if points[0] == points[-1] else points[:]
    if len(ring) < 3:
        return []
    orientation = 1.0 if polygon_area(ring) > 0 else -1.0
    remaining = list(range(len(ring)))
    triangles = []

    while len(remaining) > 3:
        ear_found = False
        for i, current in enumerate(remaining):
            previous = remaining[i - 1]
            following = remaining[(i + 1) % len(remaining)]
            a, b, c = ring[previous], ring[current], ring[following]
            if orientation * cross(a, b, c) <= 1e-10:
                continue
            if any(
                point_in_triangle(ring[index], a, b, c, orientation)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[i]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Could not triangulate an OSM footprint ring.")

    triangles.append(tuple(remaining))
    return triangles


def building_mesh(ring_latlon: list[tuple[float, float]],
                  origin_latitude: float, origin_longitude: float,
                  height: float) -> tuple[list[float], list[float], list[float],
                                          list[int], list[int], list[int]]:
    points = [
        local_xy(lat, lon, origin_latitude, origin_longitude)
        for lat, lon in ring_latlon
    ]
    if points[0] == points[-1]:
        points.pop()
    triangles = triangulate_ring(points)
    count = len(points)
    xs = [point[0] for point in points] * 2
    ys = [point[1] for point in points] * 2
    zs = [0.0] * count + [height] * count
    faces = []

    for a, b, c in triangles:
        faces.append((a, c, b))
        faces.append((a + count, b + count, c + count))
    for index in range(count):
        next_index = (index + 1) % count
        faces.extend([
            (index, next_index, next_index + count),
            (index, next_index + count, index + count),
        ])

    i, j, k = zip(*faces)
    return xs, ys, zs, list(i), list(j), list(k)


def ray_endpoint(camera_x: float, camera_y: float, camera_z: float,
                 bearing: float, elevation: float,
                 horizontal_distance: float) -> tuple[float, float, float]:
    bearing_rad = math.radians(bearing)
    elevation_rad = math.radians(elevation)
    return (
        camera_x + horizontal_distance * math.sin(bearing_rad),
        camera_y + horizontal_distance * math.cos(bearing_rad),
        camera_z + horizontal_distance * math.tan(elevation_rad),
    )


def embedded_image(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.is_file():
        return None, None
    content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}", path.name


def find_image(filename: str, directory: Path | None,
               default_directories: list[Path]) -> tuple[Path | None, list[Path]]:
    directories = [directory] if directory is not None else default_directories
    candidates = []
    for candidate_dir in directories:
        candidate = candidate_dir / filename
        if candidate not in candidates:
            candidates.append(candidate)
    return next((candidate for candidate in candidates if candidate.is_file()), None), candidates


def make_figure(crop: dict[str, Any], hit_record: dict[str, Any] | None,
                buildings: list[dict[str, Any]], camera_height: float,
                view_distance: float,
                data_description: str | None = None) -> go.Figure:
    latitude = float(crop["lat"])
    longitude = float(crop["lng"])
    camera_x = camera_y = 0.0
    camera_z = camera_height
    hit = (hit_record or {}).get("hit") or {}
    target_id = str(hit.get("osm_id", ""))
    target_type = hit.get("osm_type")
    target_height = None
    target_legend_added = False
    context_legend_added = False

    figure = go.Figure()
    for building in buildings:
        height, height_source = building_height(building)
        is_target = (
            bool(target_id)
            and building["osm_id"] == target_id
            and (not target_type or building["osm_type"] == target_type)
        )
        if is_target:
            target_height = height
        color = "#ef6c00" if is_target else "#90a4ae"
        opacity = 0.78 if is_target else 0.30
        geometry_source = building.get("geometry_source", "OpenStreetMap")
        label = (
            f"{html.escape(geometry_source)} "
            f"{html.escape(building['osm_type'])}/{html.escape(building['osm_id'])}"
            f"<br>Height: {height:.1f} m ({html.escape(height_source)})"
        )
        if building["tags"].get("name"):
            label += f"<br>{html.escape(str(building['tags']['name']))}"

        for ring in building["rings"]:
            xs, ys, zs, i, j, k = building_mesh(
                ring, latitude, longitude, height
            )
            figure.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs, i=i, j=j, k=k,
                name=(
                    f"{geometry_source} "
                    f"{building['osm_type']}/{building['osm_id']}"
                ),
                legendgroup="target" if is_target else "context",
                showlegend=(
                    not target_legend_added if is_target
                    else not context_legend_added
                ),
                color=color,
                opacity=opacity,
                flatshading=True,
                hovertext=label,
                hoverinfo="text",
            ))
            if is_target:
                target_legend_added = True
            else:
                context_legend_added = True

    center_bearing, center_elevation = pixel_to_bearing_elevation(
        float(crop["out_w"]) / 2.0,
        float(crop["out_h"]) / 2.0,
        crop,
    )
    corner_pixels = [
        (0.0, 0.0),
        (float(crop["out_w"]), 0.0),
        (float(crop["out_w"]), float(crop["out_h"])),
        (0.0, float(crop["out_h"])),
    ]
    corners = [
        ray_endpoint(
            camera_x, camera_y, camera_z,
            *pixel_to_bearing_elevation(x, y, crop),
            view_distance,
        )
        for x, y in corner_pixels
    ]

    figure.add_trace(go.Mesh3d(
        x=[camera_x] + [corner[0] for corner in corners],
        y=[camera_y] + [corner[1] for corner in corners],
        z=[camera_z] + [corner[2] for corner in corners],
        i=[0, 0, 0, 0, 1, 1],
        j=[1, 2, 3, 4, 2, 3],
        k=[2, 3, 4, 1, 3, 4],
        name="Estimated camera view frustum",
        color="#fdd835",
        opacity=0.12,
        hoverinfo="name",
        showlegend=True,
    ))
    edge_x, edge_y, edge_z = [], [], []
    for corner in corners:
        edge_x.extend((camera_x, corner[0], None))
        edge_y.extend((camera_y, corner[1], None))
        edge_z.extend((camera_z, corner[2], None))
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        edge_x.extend((start[0], end[0], None))
        edge_y.extend((start[1], end[1], None))
        edge_z.extend((start[2], end[2], None))
    figure.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line={"color": "#f9a825", "width": 4},
        name="View direction / FOV",
        hoverinfo="name",
    ))
    figure.add_trace(go.Scatter3d(
        x=[camera_x], y=[camera_y], z=[camera_z],
        mode="markers+text",
        marker={"size": 7, "color": "#1565c0", "symbol": "diamond"},
        text=["Camera"],
        textposition="top center",
        name="Estimated camera",
        hovertemplate=(
            f"Camera<br>Lat {latitude:.6f}, lon {longitude:.6f}"
            f"<br>Height {camera_height:.1f} m (assumed)<extra></extra>"
        ),
    ))

    crop_name = str(crop["crop_name"])
    lidar_comparison = bool(
        data_description and "USGS LiDAR-attributed heights" in data_description
    )
    target_label = (
        f"OSM {target_id}" if target_id
        else "USGS LiDAR height filter" if lidar_comparison
        else "Overture Maps only" if data_description
        else "No building hit recorded"
    )
    if target_height is not None:
        mapped_height = f"{target_height:.1f} m"
    elif data_description:
        mapped_height = "per-building values shown on hover"
    else:
        mapped_height = f"fallback {DEFAULT_BUILDING_HEIGHT:.1f} m if untagged"
    scene_note = (
        "Reported Overture heights are filtered by USGS LiDAR provenance; "
        "footprint geometry may use other sources."
        if lidar_comparison else
        "Overture building features and height attributes are shown; source "
        "provenance may include OpenStreetMap."
        if data_description else
        "Building geometry, extrusion, and camera height are estimates."
    )
    figure.update_layout(
        title=(
            f"{html.escape(crop_name)} — {target_label}"
            f"<br><sup>View bearing {center_bearing:.1f}°, elevation "
            f"{center_elevation:.1f}°; target height {mapped_height}. "
            f"{scene_note}</sup>"
        ),
        scene={
            "xaxis_title": "East (m from camera)",
            "yaxis_title": "North (m from camera)",
            "zaxis_title": "Height above local ground (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.25}},
        },
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 0, "r": 0, "t": 105, "b": 15},
        hovermode="closest",
    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", help="Crop filename from crop_manifest.json; defaults to the first center crop.")
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "crop_manifest.json")
    parser.add_argument("--hits", type=Path, default=BASE_DIR / "building_hits.json")
    parser.add_argument("--crop-dir", type=Path,
                        help="Directory containing crop images (auto-detected by default).")
    parser.add_argument("--panorama-dir", type=Path,
                        help="Directory containing source panoramas (auto-detected by default).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_METERS,
                        help="Building context radius around the selected camera, in metres.")
    parser.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT,
                        help="Assumed camera height above local ground, in metres.")
    parser.add_argument("--view-distance", type=float,
                        help="Frustum length in metres; defaults to the recorded hit distance or 30 m.")
    parser.add_argument("--overpass-url", default=OVERPASS_URL)
    parser.add_argument("--overture-release",
                        help="Overture release to use; defaults to the latest release.")
    parser.add_argument("--no-overture", action="store_true",
                        help="Skip Overture lookup (only with --include-osm).")
    parser.add_argument(
        "--include-osm",
        action="store_true",
        help=(
            "Use the previous combined OSM-footprints plus Overture-heights "
            "visualization instead of Overture-only (the default)."
        ),
    )
    parser.add_argument(
        "--overture-lidar-only",
        action="store_true",
        help=(
            "Compare using only Overture buildings with direct heights "
            "attributed to USGS LiDAR; does not query OSM."
        ),
    )
    args = parser.parse_args()
    if args.radius <= 0:
        raise ValueError("--radius must be greater than zero.")
    if args.no_overture and not args.include_osm:
        parser.error("--no-overture can only be used with --include-osm.")
    if args.overture_lidar_only and args.include_osm:
        parser.error("--overture-lidar-only cannot be combined with --include-osm.")
    if args.overture_lidar_only and args.no_overture:
        parser.error("--overture-lidar-only cannot be combined with --no-overture.")

    manifest = load_json(args.manifest)
    if not manifest:
        raise ValueError(f"No crops found in {args.manifest}")

    if args.crop:
        crop = next((entry for entry in manifest
                     if entry["crop_name"] == args.crop), None)
        if crop is None:
            available = ", ".join(entry["crop_name"] for entry in manifest[:10])
            raise ValueError(f"Crop {args.crop!r} is not in {args.manifest}. Examples: {available}")
    else:
        crop = next(
            (entry for entry in manifest if "_center." in entry["crop_name"]),
            manifest[0],
        )

    latitude = float(crop["lat"])
    longitude = float(crop["lng"])
    target_id = ""
    hit_record = None
    data_description = None
    if not args.include_osm:
        print(
            f"Fetching Overture buildings within {args.radius:g} m"
            + (
                " for the USGS LiDAR-height comparison..."
                if args.overture_lidar_only else "..."
            )
        )
        overture_features = fetch_overture_buildings(
            latitude, longitude, args.radius, args.overture_release
        )
        buildings, lidar_stats = overture_scene_buildings(
            overture_features, latitude, longitude, args.radius,
            lidar_only=args.overture_lidar_only,
        )
        missing_height_label = (
            "without direct USGS LiDAR-attributed height"
            if args.overture_lidar_only
            else "without usable height metadata"
        )
        print(
            f"  {len(overture_features)} Overture features; "
            f"{lidar_stats['included']} in-radius buildings added; "
            f"{lidar_stats['direct_heights']} direct Overture heights; "
            f"{lidar_stats['estimated_heights']} Overture tag estimates; "
            f"{lidar_stats['fallback_heights']} using the 9 m fallback; "
            f"{lidar_stats['without_height']} {missing_height_label}; "
            f"{lidar_stats['outside_radius']} outside the radius; "
            f"{lidar_stats['missing_geometry']} without usable geometry; "
            f"{lidar_stats['invalid_geometry']} invalid rings."
        )
        print("  No OSM API or Overpass data was queried.")
        data_description = (
            "Overture-only scene filtered to direct USGS LiDAR-attributed heights."
            if args.overture_lidar_only else
            "Overture-only building features and heights."
        )
    else:
        hit_records = load_json(args.hits) if args.hits.is_file() else []
        hit_record = next(
            (entry for entry in hit_records
             if entry.get("crop_name") == crop["crop_name"]),
            None,
        )
        hit = (hit_record or {}).get("hit") or {}
        target_id = str(hit.get("osm_id", ""))
        target_type = hit.get("osm_type")
        print(f"Fetching OSM building footprints within {args.radius:g} m...")
        buildings = overpass_buildings(
            args.overpass_url, latitude, longitude, args.radius,
            target_id or None, target_type,
        )
        target_found = any(
            building["osm_id"] == target_id
            and (not target_type or building["osm_type"] == target_type)
            for building in buildings
        )
        if target_id and not target_found:
            try:
                target_building = osm_api_target_building(target_id, target_type)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                print(
                    f"[WARN] Could not retrieve target OSM "
                    f"{target_type or 'object'}/{target_id} directly: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                if target_building is not None:
                    buildings.append(target_building)
                    print(
                        f"[OK] Recovered target OSM "
                        f"{target_building['osm_type']}/{target_id} directly "
                        "from the OSM API."
                    )
                else:
                    print(
                        f"[WARN] Target OSM {target_type or 'object'}/{target_id} "
                        "exists but did not resolve to a building footprint with "
                        "usable geometry."
                    )
        print(f"  {len(buildings)} OSM building footprints")
        if args.no_overture:
            print("  Overture height lookup skipped (--no-overture).")
        else:
            print("Fetching Overture Maps buildings and heights...")
            overture_features = fetch_overture_buildings(
                latitude, longitude, args.radius, args.overture_release
            )
            matched_count, overture_height_count = apply_overture_heights(
                buildings, overture_features
            )
            print(
                f"  {len(overture_features)} Overture buildings downloaded; "
                f"{matched_count} matched to OSM by source ID; "
                f"{overture_height_count} Overture heights available "
                "(explicit OSM heights retained)."
            )
            _, footprint_stats = merge_overture_footprints(
                buildings, overture_features, latitude, longitude, args.radius
            )
            print(
                "  Overture footprint comparison: "
                f"{footprint_stats['matched_by_osm_id']} matched by OSM ID, "
                f"{footprint_stats['duplicate_geometry']} duplicates by geometry, "
                f"{footprint_stats['outside_radius']} outside {args.radius:g} m, "
                f"{footprint_stats['added']} Overture-only buildings added, "
                f"{footprint_stats['missing_geometry']} without usable polygon geometry, "
                f"{footprint_stats['invalid_geometry']} with untriangulatable rings."
            )
            print(f"  {len(buildings)} total buildings in the scene.")
        height_sources = [building_height(building)[1] for building in buildings]
        explicit_heights = sum(
            source in {"OSM height", "OSM building:height"}
            for source in height_sources
        )
        overture_heights = sum(source.startswith("Overture height (") for source in height_sources)
        overture_estimates = sum(source.startswith("Overture ") for source in height_sources) - overture_heights
        tag_estimates = sum(
            source != "default estimate (9 m)"
            and source not in {"OSM height", "OSM building:height"}
            and not source.startswith("Overture ")
            for source in height_sources
        )
        default_heights = (
            len(buildings) - explicit_heights - overture_heights
            - overture_estimates - tag_estimates
        )
        print(
            f"  Heights: {explicit_heights} explicit OSM heights, "
            f"{overture_heights} direct Overture heights, "
            f"{overture_estimates} Overture tag estimates, "
            f"{tag_estimates} estimates using OSM level/roof/offset tags, "
            f"{default_heights} using only the 9 m default."
        )
        if target_id:
            target_building = next(
                (
                    building for building in buildings
                    if building["osm_id"] == target_id
                    and (not target_type or building["osm_type"] == target_type)
                ),
                None,
            )
            if target_building is not None:
                target_height, target_height_source = building_height(target_building)
                print(
                    f"  Target {target_building['osm_type']}/{target_id}: "
                    f"{target_height:.2f} m ({target_height_source})."
                )

    hit_distance = (hit_record or {}).get("hit", {}).get("distance_m")
    if args.view_distance is not None:
        view_distance = args.view_distance
    else:
        view_distance = (
            max(float(hit_distance) + 5.0, 10.0)
            if hit_distance is not None else 30.0
        )
    if view_distance <= 0:
        raise ValueError("--view-distance must be greater than zero.")

    figure = make_figure(
        crop, hit_record, buildings, args.camera_height, view_distance,
        data_description,
    )
    crop_path, crop_candidates = find_image(
        crop["crop_name"],
        args.crop_dir,
        [
            Path.cwd() / "perspective_crops",
            BASE_DIR / "perspective_crops",
            BASE_DIR.parent / "perspective_crops",
        ],
    )
    if crop_path is not None:
        image_path = crop_path
        image_caption = "Perspective crop"
    else:
        pano_stem = crop["crop_name"].rsplit("_", 1)[0]
        image_path, panorama_candidates = find_image(
            f"{pano_stem}.jpg",
            args.panorama_dir,
            [
                Path.cwd() / "street_images",
                BASE_DIR / "street_images",
                BASE_DIR.parent / "street_images",
            ],
        )
        image_caption = "Source panorama (equirectangular)" if image_path else None
    image_uri, image_name = embedded_image(image_path)

    output_path = args.output or (
        BASE_DIR / "3d_outputs" / f"building_camera_{Path(crop['crop_name']).stem}.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_html = figure.to_html(full_html=False, include_plotlyjs=True)
    if image_uri:
        image_panel = (
            f'<section><h2>{html.escape(image_caption or "Image")}</h2>'
            f'<img src="{image_uri}" alt="{html.escape(image_name or "Input image")}">'
            f'<p>{html.escape(image_name or "")}</p></section>'
        )
    else:
        expected_crop = html.escape(" or ".join(map(str, crop_candidates)))
        expected_pano = html.escape(" or ".join(map(str, panorama_candidates)))
        image_panel = (
            "<section><h2>Image preview unavailable</h2>"
            f"<p>Expected crop: {expected_crop}</p>"
            f"<p>Expected panorama: {expected_pano}</p></section>"
        )

    if not args.include_osm and args.overture_lidar_only:
        attribution = (
            "Building data © Overture Maps Foundation. Included heights are "
            "filtered for USGS LiDAR provenance; footprint geometry source "
            "may include OpenStreetMap."
        )
    elif not args.include_osm:
        attribution = (
            "Building data © Overture Maps Foundation; source data may include "
            '<a href="https://www.openstreetmap.org/copyright">'
            "OpenStreetMap contributors</a>."
        )
    else:
        overture_attribution = (
            "; height data © Overture Maps Foundation"
            if not args.no_overture else ""
        )
        attribution = (
            "Building footprint data © "
            '<a href="https://www.openstreetmap.org/copyright">'
            f"OpenStreetMap contributors</a>{overture_attribution}."
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Building and camera visualization</title>
  <style>
    body {{ margin: 0; font: 15px sans-serif; color: #263238; }}
    main {{ display: grid; grid-template-columns: minmax(280px, 34%) 1fr; min-height: 100vh; }}
    section {{ padding: 16px; background: #f5f7f8; overflow: auto; }}
    section img {{ display: block; width: 100%; height: auto; }}
    section p {{ overflow-wrap: anywhere; font-size: 12px; color: #546e7a; }}
    .plot {{ min-width: 0; }}
    footer {{ padding: 8px 16px; font-size: 12px; color: #546e7a; }}
    @media (max-width: 760px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    {image_panel}
    <div class="plot">{plot_html}</div>
  </main>
  <footer>{attribution}</footer>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    print(f"Saved interactive visualization: {output_path}")
    if args.include_osm and (not hit_record or not target_id):
        print("[WARN] No OSM hit is recorded for this crop; run locate_building.py to identify its target.")
    if args.overture_lidar_only:
        print(
            "The listed heights are direct Overture values with USGS LiDAR "
            "provenance; footprint geometry may use other sources. Camera "
            "height and view distance are estimates."
        )
    elif not args.include_osm:
        print(
            "Buildings and height attributes are from Overture Maps; source "
            "provenance may include OpenStreetMap. Camera height and view "
            "distance are estimates. The image is a reference, not projected "
            "onto the model."
        )
    else:
        print(
            "Camera height, view distance, and buildings without OSM "
            "height/levels are estimates. The image is shown as a reference, "
            "not projected onto the model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
