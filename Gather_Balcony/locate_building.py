#!/usr/bin/env python3
"""
locate_buildings.py
-------------------
For each crop in crop_manifest.json, cast a ray from the panorama position
along the crop's bearing and find which OSM building it hits.

pip install osmnx geopandas shapely pyproj
"""
import json, math
from pathlib import Path
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

BASE_DIR   = Path(__file__).resolve().parent
MANIFEST   = BASE_DIR / "crop_manifest.json"
OUT_FILE   = BASE_DIR / "building_hits.json"

UTM        = "EPSG:32616"   # UTM zone 16N (Atlanta)
CAM_H      = 2.5            # assumed camera height above ground (m)
BUFFER_M   = 1.5            # tolerance for GPS / footprint offsets
MAX_RANGE  = 120.0          # max ray length (m)
FETCH_DIST = 250            # OSM fetch radius around the panos (m)
GRAZING    = 70.0           # flag hits steeper than this incidence angle (deg)

to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)


# ---------------------------------------------------------------------------
# Pixel -> compass bearing + elevation (mirrors the math in crop_perspective.py)
# Use this later with a balcony mask centroid (x, y) in a crop.
# ---------------------------------------------------------------------------
def pixel_to_bearing_elevation(x, y, crop):
    f  = (crop["out_w"] / 2.0) / math.tan(math.radians(crop["fov_deg"]) / 2.0)
    rx = (x - crop["out_w"] / 2.0) / f
    ry = -(y - crop["out_h"] / 2.0) / f
    rz = 1.0
    yaw, pitch = math.radians(crop["yaw_deg"]), math.radians(crop["pitch_deg"])
    ry2 =  ry * math.cos(pitch) + rz * math.sin(pitch)
    rz2 = -ry * math.sin(pitch) + rz * math.cos(pitch)
    rx3 =  rx * math.cos(yaw) + rz2 * math.sin(yaw)
    rz3 = -rx * math.sin(yaw) + rz2 * math.cos(yaw)
    lon = math.atan2(rx3, rz3)
    lat = math.atan2(ry2, math.hypot(rx3, rz3))
    bearing = (crop["car_heading"] + math.degrees(lon)) % 360.0
    return bearing, math.degrees(lat)


# ---------------------------------------------------------------------------
def parse_height(row):
    h = row.get("height")
    if h is not None and str(h) != "nan":
        try:
            return float(str(h).split()[0].replace("m", ""))
        except ValueError:
            pass
    lv = row.get("building:levels")
    if lv is not None and str(lv) != "nan":
        try:
            return float(lv) * 3.0
        except ValueError:
            pass
    return None  # unknown


def facade_normal_bearing(poly, hit, cam):
    geom = poly if poly.geom_type == "Polygon" else min(poly.geoms, key=lambda g: g.distance(hit))
    c = list(geom.exterior.coords)
    i = min(range(len(c) - 1), key=lambda k: LineString([c[k], c[k + 1]]).distance(hit))
    dx, dy = c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]
    n = np.array([dy, -dx], dtype=float)
    n /= np.linalg.norm(n)
    if n @ np.array([cam.x - hit.x, cam.y - hit.y]) < 0:
        n = -n  # make it point out toward the camera
    return (math.degrees(math.atan2(n[0], n[1])) + 360.0) % 360.0


def cast(origin, bearing, elev, gdf):
    b   = math.radians(bearing)
    ray = LineString([origin, Point(origin.x + MAX_RANGE * math.sin(b),
                                    origin.y + MAX_RANGE * math.cos(b))])
    cands = []
    for idx, row in gdf.iterrows():
        inter = ray.intersection(row.geometry.buffer(BUFFER_M))
        if not inter.is_empty:
            hit = nearest_points(row.geometry.boundary, inter)[0]
            cands.append((origin.distance(hit), idx, row, hit))
    cands.sort(key=lambda t: t[0])

    for d, idx, row, hit in cands:
        h = parse_height(row)
        if h is not None and CAM_H + d * math.tan(math.radians(elev)) > h:
            continue  # ray passes over this building
        nb  = facade_normal_bearing(row.geometry, hit, origin)
        inc = abs((bearing + 180.0 - nb + 180.0) % 360.0 - 180.0)
        return {
            "osm_type":        str(idx[0]) if isinstance(idx, tuple) else None,
            "osm_id":          str(idx[1] if isinstance(idx, tuple) else idx),
            "distance_m":      round(d, 2),
            "facade_normal_bearing": round(nb, 1),
            "incidence_deg":   round(inc, 1),
            "grazing_flag":    inc > GRAZING,
            "height_m":        h,
            "height_source":   "osm" if h is not None else "unknown",
        }
    return None


def main():
    crops = json.load(open(MANIFEST))
    lat0 = float(np.mean([c["lat"] for c in crops]))
    lng0 = float(np.mean([c["lng"] for c in crops]))

    print("Fetching OSM buildings...")
    gdf = ox.features_from_point((lat0, lng0), tags={"building": True}, dist=FETCH_DIST)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(UTM)
    print(f"  {len(gdf)} footprints")

    results = []
    for c in crops:
        x, y   = to_utm.transform(c["lng"], c["lat"])
        origin = Point(x, y)
        # center pixel of the crop; swap in a balcony centroid later
        bearing, elev = pixel_to_bearing_elevation(c["out_w"] / 2, c["out_h"] / 2, c)
        hit = cast(origin, bearing, elev, gdf)
        results.append({"crop_name": c["crop_name"], "pano_id": c["pano_id"],
                        "bearing": round(bearing, 1), "elevation": round(elev, 1), "hit": hit})
        print(f"{c['crop_name']}: bearing {bearing:.1f}° -> "
              f"{hit['osm_id'] if hit else 'no building'}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()