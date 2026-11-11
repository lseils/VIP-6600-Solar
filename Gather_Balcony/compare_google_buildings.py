#!/usr/bin/env python3
"""Compare local Overture and LiDAR footprints beside Google's hosted 3D tiles.

The script compares only geometry already present in the generated scene
metadata. Google tiles are streamed for visual context and are never queried
for geometry or included in the discrepancy report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any
import webbrowser

from dotenv import load_dotenv
from shapely.geometry import mapping, shape

from view_google_3d_tiles import CESIUM_VERSION, DEFAULT_SCENE_JSON, _finite_number


BASE_DIR = Path(__file__).resolve().parent


def _geometry(value: Any, label: str) -> Any | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in {
        "Polygon", "MultiPolygon",
    }:
        raise ValueError(f"{label} must be a Polygon, MultiPolygon, or null.")
    geometry = shape(value)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{label} must be a non-empty valid polygon.")
    return geometry


def compare_scene_buildings(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Scene metadata must be a JSON object: {path}")

    origin = metadata.get("origin")
    sources = metadata.get("sources")
    buildings = sources.get("buildings") if isinstance(sources, dict) else None
    features = (
        buildings.get("comparison_features")
        if isinstance(buildings, dict) else None
    )
    if not isinstance(origin, dict) or not isinstance(features, list):
        raise ValueError(
            "This scene does not contain comparison geometry. Rebuild it with "
            "the updated build_neighborhood_scene.py, then run this script "
            "again."
        )

    scene = {
        "latitude": _finite_number(origin.get("latitude"), "origin.latitude"),
        "longitude": _finite_number(origin.get("longitude"), "origin.longitude"),
        "elevation": _finite_number(
            origin.get("bare_earth_elevation_m", 0.0),
            "origin.bare_earth_elevation_m",
        ),
        "boundary_local": [],
    }
    if not -90.0 <= scene["latitude"] <= 90.0:
        raise ValueError("Scene latitude is outside valid WGS84 bounds.")
    if not -180.0 <= scene["longitude"] <= 180.0:
        raise ValueError("Scene longitude is outside valid WGS84 bounds.")

    terrain_mesh = metadata.get("terrain_mesh")
    boundary = (
        terrain_mesh.get("aoi_boundary_local")
        if isinstance(terrain_mesh, dict) else None
    )
    if isinstance(boundary, list):
        for index, point in enumerate(boundary):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"AOI boundary point {index} is invalid.")
            scene["boundary_local"].append([
                _finite_number(point[0], f"AOI boundary {index} east"),
                _finite_number(point[1], f"AOI boundary {index} north"),
            ])

    report_features: list[dict[str, Any]] = []
    for index, item in enumerate(features):
        if not isinstance(item, dict):
            raise ValueError(f"Building comparison feature {index} is invalid.")
        feature_id = str(item.get("id", f"feature-{index + 1}"))
        overture = _geometry(
            item.get("overture_footprint_local"),
            f"{feature_id} Overture footprint",
        )
        lidar = _geometry(
            item.get("lidar_footprint_local"),
            f"{feature_id} LiDAR footprint",
        )
        is_lidar_candidate = bool(item.get("lidar_only_candidate", False))
        ground = _finite_number(
            item.get("ground_elevation_m", scene["elevation"]),
            f"{feature_id} ground elevation",
        )
        overture_height = item.get("overture_height_m")
        if overture_height is not None:
            overture_height = _finite_number(
                overture_height, f"{feature_id} Overture height"
            )
        lidar_roofs = item.get("lidar_roof_elevations_m", [])
        if not isinstance(lidar_roofs, list):
            raise ValueError(f"{feature_id} LiDAR roof elevations must be a list.")
        lidar_roofs = [
            _finite_number(value, f"{feature_id} LiDAR roof elevation")
            for value in lidar_roofs
        ]
        point_count = item.get("lidar_point_count", 0)
        if isinstance(point_count, bool) or not isinstance(point_count, int):
            raise ValueError(f"{feature_id} LiDAR point count must be an integer.")

        overture_area = overture.area if overture is not None else 0.0
        lidar_area = lidar.area if lidar is not None else 0.0
        intersection_area = 0.0
        iou = None
        overture_only = None
        lidar_only = None
        if overture is not None and lidar is not None:
            intersection = overture.intersection(lidar)
            union = overture.union(lidar)
            intersection_area = intersection.area
            iou = intersection_area / union.area if union.area else 1.0
            overture_only = overture.difference(lidar)
            lidar_only = lidar.difference(overture)

        lidar_roof_elevation = median(lidar_roofs) if lidar_roofs else None
        lidar_height = (
            float(lidar_roof_elevation - ground)
            if lidar_roof_elevation is not None else None
        )
        height_difference = (
            lidar_height - overture_height
            if lidar_height is not None and overture_height is not None
            else None
        )
        overture_roof_local = (
            ground + overture_height - scene["elevation"]
            if overture_height is not None else None
        )
        lidar_roof_local = (
            lidar_roof_elevation - scene["elevation"]
            if lidar_roof_elevation is not None else None
        )
        centroid_geometry = overture if overture is not None else lidar
        centroid = (
            [
                float(centroid_geometry.centroid.x),
                float(centroid_geometry.centroid.y),
            ]
            if centroid_geometry is not None else [0.0, 0.0]
        )
        report_features.append({
            "id": feature_id,
            "lidar_only_candidate": is_lidar_candidate,
            "overture": mapping(overture) if overture is not None else None,
            "lidar": mapping(lidar) if lidar is not None else None,
            "overture_only": (
                mapping(overture_only)
                if overture_only is not None and not overture_only.is_empty
                else None
            ),
            "lidar_only": (
                mapping(lidar_only)
                if lidar_only is not None and not lidar_only.is_empty
                else None
            ),
            "centroid_local": centroid,
            "ground_elevation_m": ground,
            "overture_area_m2": overture_area,
            "lidar_area_m2": lidar_area,
            "intersection_area_m2": intersection_area,
            "overture_only_area_m2": (
                overture_only.area if overture_only is not None else None
            ),
            "lidar_only_area_m2": (
                lidar_only.area if lidar_only is not None else None
            ),
            "iou": iou,
            "overture_height_m": overture_height,
            "lidar_height_m": lidar_height,
            "overture_roof_local_m": overture_roof_local,
            "lidar_roof_local_m": lidar_roof_local,
            "height_difference_m": height_difference,
            "lidar_point_count": point_count,
            "status": (
                "LiDAR-only candidate"
                if is_lidar_candidate or overture is None
                else (
                    "No LiDAR roof coverage"
                    if lidar is None else "Footprints compared"
                )
            ),
        })

    compared = [item for item in report_features if item["iou"] is not None]
    candidates = [
        item for item in report_features
        if item["status"] == "LiDAR-only candidate"
    ]
    summary = {
        **scene,
        "feature_count": len(report_features),
        "compared_count": len(compared),
        "no_lidar_coverage_count": sum(
            item["status"] == "No LiDAR roof coverage"
            for item in report_features
        ),
        "lidar_only_candidate_count": len(candidates),
        "mean_iou": (
            sum(item["iou"] for item in compared) / len(compared)
            if compared else None
        ),
    }
    return summary, report_features


def write_report(path: Path, features: list[dict[str, Any]]) -> None:
    columns = (
        "id", "status", "overture_area_m2", "lidar_area_m2",
        "intersection_area_m2", "overture_only_area_m2",
        "lidar_only_area_m2", "iou", "overture_height_m",
        "lidar_height_m", "height_difference_m", "lidar_point_count",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for feature in features:
            writer.writerow({
                key: feature[key] for key in columns
            })


def _script_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def build_viewer_html(
    summary: dict[str, Any],
    features: list[dict[str, Any]],
    api_key: str,
) -> str:
    reviewed = [
        feature for feature in features
        if feature["iou"] is not None or feature["status"] == "LiDAR-only candidate"
    ]
    reviewed.sort(key=lambda feature: (
        feature["iou"] is not None,
        feature["iou"] if feature["iou"] is not None else 0.0,
        -float(feature["lidar_only_area_m2"] or feature["lidar_area_m2"]),
    ))
    config = _script_json({
        **summary,
        "apiKey": api_key,
        "features": features,
        "reviewed_ids": [feature["id"] for feature in reviewed[:100]],
    })
    average_iou = (
        f"{summary['mean_iou']:.2f}"
        if summary["mean_iou"] is not None else "n/a"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Building footprint comparison</title>
  <script src="https://ajax.googleapis.com/ajax/libs/cesiumjs/{CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
  <link href="https://ajax.googleapis.com/ajax/libs/cesiumjs/{CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; overflow:hidden; }}
    #panel {{ position:absolute; z-index:1; top:12px; left:12px; width:min(390px,calc(100vw - 48px));
      max-height:calc(100vh - 48px); overflow:auto; padding:12px; color:#f5f7f8;
      background:rgba(25,32,38,.92); border-radius:5px; font:13px/1.45 sans-serif; }}
    #summary {{ margin:7px 0 10px; color:#d5dde2; }}
    .legend {{ display:flex; gap:9px; flex-wrap:wrap; margin:9px 0; }}
    .legend span {{ white-space:nowrap; }} .swatch {{ display:inline-block; width:12px; height:12px;
      margin-right:4px; vertical-align:-2px; }}
    label {{ display:block; margin:5px 0; }}
    #reviewList {{ border-top:1px solid #59636b; margin-top:10px; padding-top:7px; }}
    .feature {{ display:block; width:100%; margin:4px 0; padding:6px; color:#fff; text-align:left;
      border:1px solid #64717a; border-radius:3px; background:#303b43; cursor:pointer; }}
    .feature:hover {{ background:#43515a; }}
    small {{ color:#c1cbd1; }}
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <section id="panel">
    <strong>Building footprint comparison</strong>
    <div id="summary"></div>
    <div id="tileStatus" role="status">Loading Google 3D Tiles…</div>
    <small>Google tiles are visual context only. Metrics compare local Overture
      footprints with detected class-6 LiDAR roof footprints; LiDAR coverage can
      be incomplete, so differences are review candidates, not errors.</small>
    <div class="legend">
      <span><i class="swatch" style="background:#ffb74d"></i>Overture</span>
      <span><i class="swatch" style="background:#4dd0e1"></i>LiDAR</span>
      <span><i class="swatch" style="background:#ef5350"></i>Overture-only</span>
      <span><i class="swatch" style="background:#ab47bc"></i>LiDAR-only</span>
    </div>
    <label><input id="showOverture" type="checkbox" checked> Overture outlines</label>
    <label><input id="showLidar" type="checkbox" checked> LiDAR outlines</label>
    <label><input id="showDiff" type="checkbox" checked> Local-source footprint differences</label>
    <div id="reviewList"></div>
  </section>
  <script>
    const config = {config};
    const summary = document.getElementById("summary");
    const reviewList = document.getElementById("reviewList");
    const viewer = new Cesium.Viewer("cesiumContainer", {{
      imageryProvider: false, baseLayerPicker: false, geocoder: false,
      homeButton: false, sceneModePicker: false, navigationHelpButton: false,
      animation: false, timeline: false, requestRenderMode: true
    }});
    viewer.scene.globe.show = false;
    viewer.scene.skyBox.show = false;
    viewer.scene.skyAtmosphere.show = false;
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#202a32");
    const tilesetUrl = new URL("https://tile.googleapis.com/v1/3dtiles/root.json");
    tilesetUrl.searchParams.set("key", config.apiKey);
    const tileStatus = document.getElementById("tileStatus");
    Cesium.Cesium3DTileset.fromUrl(tilesetUrl.toString(), {{
      showCreditsOnScreen: true
    }}).then(tileset => {{
      viewer.scene.primitives.add(tileset);
      tileStatus.textContent = "Google tiles loaded.";
    }}).catch(error => {{
      console.error("Could not load Google 3D Tiles.", error);
      tileStatus.textContent =
        "Google tiles could not load. Check Map Tiles API, billing, and key restrictions.";
    }});

    const enuCenter = Cesium.Cartesian3.fromDegrees(
      config.longitude, config.latitude, config.elevation
    );
    const enu = Cesium.Transforms.eastNorthUpToFixedFrame(enuCenter);
    const localToWorld = (point, elevation) => Cesium.Matrix4.multiplyByPoint(
      enu, new Cesium.Cartesian3(point[0], point[1], elevation),
      new Cesium.Cartesian3()
    );
    const colors = {{
      overture: Cesium.Color.fromCssColorString("#ffb74d"),
      lidar: Cesium.Color.fromCssColorString("#4dd0e1"),
      overtureOnly: Cesium.Color.fromCssColorString("#ef5350"),
      lidarOnly: Cesium.Color.fromCssColorString("#ab47bc")
    }};
    const groups = {{overture: [], lidar: [], diff: []}};
    const ringsFor = (geometry) => {{
      if (!geometry) return [];
      if (geometry.type === "Polygon") return geometry.coordinates;
      if (geometry.type === "MultiPolygon") return geometry.coordinates.flat();
      return [];
    }};
    const addOutline = (group, geometry, elevation, color, width) => {{
      for (const polygon of ringsFor(geometry)) {{
        for (const ring of polygon) {{
          if (ring.length < 2) continue;
          const positions = ring.map(point => localToWorld(point, elevation));
          const entity = viewer.entities.add({{
            polyline: {{
              positions, width, material: color,
              depthFailMaterial: color
            }}
          }});
          groups[group].push(entity);
        }}
      }}
    }};
    const flyToFeature = (feature) => {{
      const top = Math.max(
        feature.overture_roof_local_m ?? 0,
        feature.lidar_roof_local_m ?? 0
      );
      const position = localToWorld(feature.centroid_local, top + 50);
      viewer.camera.flyTo({{
        destination: position,
        orientation: {{
          heading: 0,
          pitch: Cesium.Math.toRadians(-65),
          roll: 0
        }},
        duration: 0.8
      }});
    }};
    const formatArea = value => value == null ? "n/a" : `${{value.toFixed(1)}} m²`;
    const formatHeight = value => value == null ? "n/a" : `${{value.toFixed(1)}} m`;
    const featuresById = new Map(config.features.map(feature => [feature.id, feature]));
    config.features.forEach(feature => {{
      const overtureZ = feature.overture_roof_local_m ?? 2;
      const lidarZ = feature.lidar_roof_local_m ?? overtureZ;
      addOutline("overture", feature.overture, overtureZ,
        colors.overture, 2);
      addOutline("lidar", feature.lidar, lidarZ,
        colors.lidar, 2);
      addOutline("diff", feature.overture_only, overtureZ,
        colors.overtureOnly, 3);
      addOutline("diff", feature.lidar_only, lidarZ,
        colors.lidarOnly, 3);
    }});
    config.reviewed_ids.forEach(id => {{
      const feature = featuresById.get(id);
      if (!feature) return;
      const button = document.createElement("button");
      button.className = "feature";
      const metrics = feature.iou == null
        ? `${{feature.status}} · ${{formatArea(feature.lidar_area_m2)}}`
        : `IoU ${{feature.iou.toFixed(2)}} · O-only ${{formatArea(feature.overture_only_area_m2)}} · L-only ${{formatArea(feature.lidar_only_area_m2)}}`;
      const heightText = feature.height_difference_m == null
        ? "" : ` · Δheight ${{formatHeight(feature.height_difference_m)}}`;
      button.textContent = `${{feature.id}} — ${{metrics}}${{heightText}}`;
      button.onclick = () => flyToFeature(feature);
      reviewList.append(button);
    }});
    summary.textContent =
      `${{config.compared_count}} compared · ${{config.no_lidar_coverage_count}} without LiDAR roof coverage · ` +
      `${{config.lidar_only_candidate_count}} LiDAR-only candidates · mean IoU ${{{average_iou}}}`;
    document.getElementById("showOverture").onchange = event =>
      groups.overture.forEach(entity => entity.show = event.target.checked);
    document.getElementById("showLidar").onchange = event =>
      groups.lidar.forEach(entity => entity.show = event.target.checked);
    document.getElementById("showDiff").onchange = event =>
      groups.diff.forEach(entity => entity.show = event.target.checked);

    const boundary = config.boundary_local;
    if (boundary.length >= 3) {{
      const localCenter = boundary.reduce(
        (sum, point) => [sum[0] + point[0] / boundary.length,
          sum[1] + point[1] / boundary.length], [0, 0]
      );
      const maxRadius = Math.max(...boundary.map(point =>
        Math.hypot(point[0] - localCenter[0], point[1] - localCenter[1])
      ));
      viewer.scene.screenSpaceCameraController.maximumZoomDistance = maxRadius * 3;
      viewer.camera.flyTo({{
        destination: localToWorld(localCenter, maxRadius * 1.8),
        orientation: {{heading:0, pitch:Cesium.Math.toRadians(-55), roll:0}},
        duration: 1.2
      }});
      const surfaceCenter = Cesium.Cartesian3.fromDegrees(
        config.longitude, config.latitude, 0
      );
      const inverse = Cesium.Matrix4.inverseTransformation(
        Cesium.Transforms.eastNorthUpToFixedFrame(surfaceCenter),
        new Cesium.Matrix4()
      );
      let correcting = false;
      const recenter = () => {{
        correcting = true;
        viewer.camera.flyTo({{
          destination: localToWorld(localCenter, maxRadius * 1.8),
          orientation: {{heading:0, pitch:Cesium.Math.toRadians(-55), roll:0}},
          duration: 0.5,
          complete: () => {{ correcting = false; }}
        }});
      }};
      viewer.camera.moveEnd.addEventListener(() => {{
        if (correcting) return;
        const screenCenter = new Cesium.Cartesian2(
          viewer.scene.canvas.clientWidth / 2,
          viewer.scene.canvas.clientHeight / 2
        );
        const worldPoint = viewer.camera.pickEllipsoid(
          screenCenter, Cesium.Ellipsoid.WGS84
        );
        if (!worldPoint) {{
          recenter();
          return;
        }}
        const local = Cesium.Matrix4.multiplyByPoint(
          inverse, worldPoint, new Cesium.Cartesian3()
        );
        let inside = false;
        for (let i=0, j=boundary.length-1; i<boundary.length; j=i++) {{
          const [xi, yi] = boundary[i], [xj, yj] = boundary[j];
          if (((yi > local.y) !== (yj > local.y)) &&
              local.x < (xj-xi)*(local.y-yi)/(yj-yi)+xi) inside = !inside;
        }}
        if (!inside) {{
          recenter();
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-json", type=Path, default=DEFAULT_SCENE_JSON,
        help="Neighborhood scene JSON; defaults to the pano_000_center scene.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--report", type=Path,
        help="CSV report path; defaults beside the scene JSON.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if not args.scene_json.is_file():
        raise FileNotFoundError(f"Scene metadata JSON not found: {args.scene_json}")

    summary, features = compare_scene_buildings(args.scene_json)
    report_path = args.report or args.scene_json.with_name(
        f"{args.scene_json.stem}_building_comparison.csv"
    )
    write_report(report_path, features)

    load_dotenv(BASE_DIR.parent / ".env")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. Add it to the repository .env "
            "file or export it in the shell."
        )
    page = build_viewer_html(summary, features, api_key)

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    url = f"http://127.0.0.1:{args.port}/"
    with ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler) as server:
        print(f"Serving comparison viewer at {url}")
        print(
            f"Local source comparison: {summary['compared_count']} Overture/LiDAR "
            f"footprints; mean IoU "
            f"{summary['mean_iou']:.3f}" if summary["mean_iou"] is not None
            else "Local source comparison: no matched footprints."
        )
        print(f"CSV report: {report_path}")
        print(
            "Google tiles provide visual context only; the CSV and metrics "
            "compare Overture with LiDAR."
        )
        print("Press Ctrl+C to stop the local viewer.")
        webbrowser.open(url)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
