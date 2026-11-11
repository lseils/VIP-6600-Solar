"""Match saved Google Street View panoramas to nearby OSM buildings.

The current Google metadata contains panorama position and car heading, but no
perspective crop yaw/FOV. Results are therefore ranked nearby-building
candidates, not definitive facade identifications.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EARTH_RADIUS_METERS = 6_371_000.0


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def request_overpass(endpoint: str, latitude: float, longitude: float, radius: float) -> list[dict[str, Any]]:
    query = f"""
[out:json][timeout:60];
(
  way[building](around:{radius},{latitude},{longitude});
  relation[building](around:{radius},{latitude},{longitude});
);
out tags center geom;
"""
    body = urlencode({"data": query}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "VIP-6600-Solar/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read())
    return payload.get("elements", [])


def local_xy(latitude: float, longitude: float, origin_latitude: float, origin_longitude: float) -> tuple[float, float]:
    scale = math.cos(math.radians(origin_latitude))
    east = math.radians(longitude - origin_longitude) * EARTH_RADIUS_METERS * scale
    north = math.radians(latitude - origin_latitude) * EARTH_RADIUS_METERS
    return east, north


def geometry_points(element: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(point["lat"]), float(point["lon"]))
        for point in element.get("geometry", [])
        if "lat" in point and "lon" in point
    ]


def candidate_record(element: dict[str, Any], latitude: float, longitude: float, heading: float) -> dict[str, Any] | None:
    points = geometry_points(element)
    if not points:
        center = element.get("center", {})
        if "lat" not in center or "lon" not in center:
            return None
        points = [(float(center["lat"]), float(center["lon"]))]

    centroid_latitude = sum(point[0] for point in points) / len(points)
    centroid_longitude = sum(point[1] for point in points) / len(points)
    east, north = local_xy(centroid_latitude, centroid_longitude, latitude, longitude)
    distance = math.hypot(east, north)
    bearing = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    angular_error = abs((bearing - heading + 180.0) % 360.0 - 180.0)

    # This is deliberately a weak ranking: car heading is not view heading.
    score = (1.0 / (1.0 + distance / 25.0)) * max(0.0, math.cos(math.radians(angular_error)))
    tags = element.get("tags", {})
    return {
        "osm_type": element.get("type"),
        "osm_id": element.get("id"),
        "tags": tags,
        "center": {"lat": centroid_latitude, "lon": centroid_longitude},
        "distance_meters": round(distance, 2),
        "bearing_degrees": round(bearing, 2),
        "heading_error_degrees": round(angular_error, 2),
        "ranking_score": round(score, 5),
        "geometry": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("match_config.json"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N panoramas.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    metadata_path = (config_path.parent / config["input"]["metadata_file"]).resolve()
    metadata = load_json(metadata_path)
    if args.limit is not None:
        metadata = metadata[: args.limit]

    output_dir = config_path.parent / config["output"]["directory"] / config["output"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, pano in enumerate(metadata, start=1):
        latitude = float(pano["lat"])
        longitude = float(pano["lng"])
        heading = float(pano.get("car_heading", 0.0))
        try:
            elements = request_overpass(
                config["osm"]["overpass_url"],
                latitude,
                longitude,
                float(config["osm"]["search_radius_meters"]),
            )
            candidates = [
                candidate
                for element in elements
                if (candidate := candidate_record(element, latitude, longitude, heading)) is not None
            ]
            candidates.sort(key=lambda candidate: candidate["ranking_score"], reverse=True)
            candidates = candidates[: int(config["osm"]["top_candidates"])]
            status = "matched" if candidates else "no_buildings_found"
            error = None
        except Exception as exception:
            candidates = []
            status = "query_failed"
            error = str(exception)

        results.append(
            {
                "image_name": pano.get("image_name"),
                "pano_id": pano.get("pano_id"),
                "camera": {"lat": latitude, "lng": longitude, "car_heading": heading},
                "status": status,
                "warning": "car_heading is not a perspective crop viewing heading",
                "candidates": candidates,
                "error": error,
            }
        )
        print(f"[{index}/{len(metadata)}] {pano.get('image_name')}: {status} ({len(candidates)} candidates)")
        time.sleep(float(config["osm"]["request_pause_seconds"]))

    output_path = output_dir / config["output"]["matches_file"]
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
