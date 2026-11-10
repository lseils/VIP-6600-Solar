#!/usr/bin/env python3
"""View Google's hosted Photorealistic 3D Tiles over a generated scene area.

This serves a local CesiumJS viewer. Google tiles are requested by the browser
for visualization and are not saved or exported by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
import webbrowser

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_JSON = (
    BASE_DIR
    / "neighborhood_scenes"
    / "pano_000_center_neighborhood_scene.json"
)
CESIUM_VERSION = "1.105"


class SceneView(TypedDict):
    latitude: float
    longitude: float
    elevation: float
    view_radius: float
    boundary_local: list[list[float]]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Scene metadata {label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Scene metadata {label} must be a finite number.")
    return number


def load_scene_view(path: Path, radius_override: float | None) -> SceneView:
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Scene metadata must be a JSON object: {path}")

    origin = metadata.get("origin")
    if not isinstance(origin, dict):
        raise ValueError(f"Scene metadata has no valid origin: {path}")
    latitude = _finite_number(origin.get("latitude"), "origin.latitude")
    longitude = _finite_number(origin.get("longitude"), "origin.longitude")
    elevation = _finite_number(
        origin.get("bare_earth_elevation_m", 0.0),
        "origin.bare_earth_elevation_m",
    )
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("Scene metadata origin is outside valid WGS84 bounds.")

    sources = metadata.get("sources", {})
    terrain_source = (
        sources.get("terrain", {}) if isinstance(sources, dict) else {}
    )
    recorded_radius = (
        terrain_source.get("radius_meters")
        if isinstance(terrain_source, dict) else None
    )
    radius = (
        radius_override
        if radius_override is not None
        else (
            _finite_number(recorded_radius, "sources.terrain.radius_meters")
            if recorded_radius is not None
            else None
        )
    )
    if radius is not None and radius <= 0:
        raise ValueError("Scene radius must be greater than zero.")

    terrain_mesh = metadata.get("terrain_mesh", {})
    boundary = (
        terrain_mesh.get("aoi_boundary_local", [])
        if isinstance(terrain_mesh, dict) else []
    )
    boundary_radius = 0.0
    boundary_local: list[list[float]] = []
    if isinstance(boundary, list):
        for index, point in enumerate(boundary):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(
                    f"Scene metadata terrain boundary point {index} is invalid."
                )
            east = _finite_number(point[0], f"terrain boundary {index} east")
            north = _finite_number(point[1], f"terrain boundary {index} north")
            boundary_radius = max(boundary_radius, math.hypot(east, north))
            boundary_local.append([east, north])

    if radius_override is not None or not boundary_local:
        if radius is None:
            raise ValueError(
                "Scene metadata does not record an area radius or boundary; "
                "supply --radius."
            )
        boundary_local = [
            [-radius, -radius],
            [radius, -radius],
            [radius, radius],
            [-radius, radius],
        ]
        boundary_radius = radius * math.sqrt(2.0)

    # A recorded radius describes the half-width of a square; its corners are
    # farther from the center than that half-width.
    view_radius = max(
        boundary_radius,
        radius * math.sqrt(2.0) if radius is not None else 0.0,
    )
    return {
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "view_radius": view_radius,
        "boundary_local": boundary_local,
    }


def build_viewer_html(scene: SceneView, api_key: str) -> str:
    config = json.dumps({
        **scene,
        "apiKey": api_key,
    })
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Google Photorealistic 3D Tiles</title>
  <script src="https://ajax.googleapis.com/ajax/libs/cesiumjs/{CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
  <link href="https://ajax.googleapis.com/ajax/libs/cesiumjs/{CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html, body, #cesiumContainer {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    #status {{
      position: absolute; z-index: 1; top: 12px; left: 12px;
      max-width: min(620px, calc(100vw - 48px)); padding: 10px 12px;
      color: #fff; background: rgba(25, 32, 38, .88);
      border-radius: 4px; font: 14px/1.4 sans-serif;
    }}
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <div id="status">Loading Google Photorealistic 3D Tiles…</div>
  <script>
    const config = {config};
    const status = document.getElementById("status");
    const viewer = new Cesium.Viewer("cesiumContainer", {{
      imageryProvider: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: true,
      requestRenderMode: true
    }});
    viewer.scene.globe.show = false;
    viewer.scene.skyBox.show = false;
    viewer.scene.skyAtmosphere.show = false;
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#202a32");
    viewer.scene.screenSpaceCameraController.minimumZoomDistance = 5;
    viewer.scene.screenSpaceCameraController.maximumZoomDistance =
      config.view_radius * 3;

    const tilesetUrl = new URL("https://tile.googleapis.com/v1/3dtiles/root.json");
    tilesetUrl.searchParams.set("key", config.apiKey);
    const tileset = new Cesium.Cesium3DTileset({{
      url: tilesetUrl.toString(),
      showCreditsOnScreen: true
    }});
    viewer.scene.primitives.add(tileset);

    tileset.readyPromise.then(() => {{
      const center = Cesium.Cartesian3.fromDegrees(
        config.longitude, config.latitude, config.elevation
      );
      const surfaceCenter = Cesium.Cartesian3.fromDegrees(
        config.longitude, config.latitude, 0
      );
      const bounds = new Cesium.BoundingSphere(center, config.view_radius);
      const recenter = () => viewer.camera.flyToBoundingSphere(bounds, {{
          offset: new Cesium.HeadingPitchRange(
            0,
            Cesium.Math.toRadians(-38),
            config.view_radius * 2.4
          ),
          duration: 0.5
        }});
      let initialViewComplete = false;
      viewer.camera.moveEnd.addEventListener(() => {{
        if (!initialViewComplete || !config.boundary_local.length) return;
        const canvasCenter = new Cesium.Cartesian2(
          viewer.scene.canvas.clientWidth / 2,
          viewer.scene.canvas.clientHeight / 2
        );
        const worldPoint = viewer.camera.pickEllipsoid(
          canvasCenter,
          Cesium.Ellipsoid.WGS84
        );
        if (!worldPoint) {{
          recenter();
          return;
        }}
        const enu = Cesium.Transforms.eastNorthUpToFixedFrame(surfaceCenter);
        const inverseEnu = Cesium.Matrix4.inverseTransformation(
          enu,
          new Cesium.Matrix4()
        );
        const localPoint = Cesium.Matrix4.multiplyByPoint(
          inverseEnu,
          worldPoint,
          new Cesium.Cartesian3()
        );
        let inside = false;
        for (
          let i = 0, j = config.boundary_local.length - 1;
          i < config.boundary_local.length;
          j = i++
        ) {{
          const [xi, yi] = config.boundary_local[i];
          const [xj, yj] = config.boundary_local[j];
          const crosses = ((yi > localPoint.y) !== (yj > localPoint.y)) &&
            (localPoint.x < (xj - xi) * (localPoint.y - yi) / (yj - yi) + xi);
          if (crosses) inside = !inside;
        }}
        if (!inside) recenter();
      }});
      viewer.camera.flyToBoundingSphere(bounds, {{
        offset: new Cesium.HeadingPitchRange(
          0,
          Cesium.Math.toRadians(-38),
          config.view_radius * 2.4
        ),
        duration: 1.5,
        complete: () => {{ initialViewComplete = true; }}
      }});
      status.textContent =
        "Google Photorealistic 3D Tiles · view constrained to the reconstruction " +
        "area · use mouse wheel to zoom. Google and provider attribution is " +
        "shown in the Cesium credits.";
    }}).catch((error) => {{
      console.error(error);
      status.textContent =
        "Could not load Google 3D Tiles. Check that billing and the Map Tiles " +
        "API are enabled and this API key is authorized for localhost.";
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=DEFAULT_SCENE_JSON,
        help="Neighborhood scene JSON; defaults to the pano_000_center scene.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        help=(
            "Override the recorded scene radius in metres. The viewer frames "
            "the square area around that radius."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local viewer port (bound only to 127.0.0.1).",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.radius is not None and (
        not math.isfinite(args.radius) or args.radius <= 0
    ):
        parser.error("--radius must be a finite number greater than zero.")
    if not args.scene_json.is_file():
        raise FileNotFoundError(f"Scene metadata JSON not found: {args.scene_json}")

    load_dotenv(BASE_DIR.parent / ".env")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. Add it to the repository .env "
            "file or export it in the shell."
        )

    scene = load_scene_view(args.scene_json, args.radius)
    page = build_viewer_html(scene, api_key)

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
        print(f"Serving Google Photorealistic 3D Tiles at {url}")
        print(f"Scene center: {scene['latitude']:.6f}, {scene['longitude']:.6f}")
        print(f"Framed scene radius: {scene['view_radius']:.0f} m")
        print("Tiles are streamed by CesiumJS for display; this script does not save them.")
        print("Press Ctrl+C to stop the local viewer.")
        webbrowser.open(url)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
