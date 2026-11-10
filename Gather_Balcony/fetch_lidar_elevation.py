#!/usr/bin/env python3
"""Download a USGS 3DEP bare-earth DEM for a crop's surrounding area.

The output is a georeferenced GeoTIFF from the USGS 3DEP Bare Earth DEM
Dynamic service, plus a JSON sidecar recording the requested and returned
area, resolution, source service, and retrieval time.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "crop_manifest.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "lidar_elevation"
SERVICE_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer"
)
SERVICE_DESCRIPTION = (
    "USGS 3D Elevation Program (3DEP) Bare Earth DEM Dynamic service"
)
MAX_IMAGE_DIMENSION = 8000
USER_AGENT = "VIP-6600-Solar-3DEP-elevation-download/1.0"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"No crop records found in {path}")
    return manifest


def select_crop(
    manifest: list[dict[str, Any]], crop_name: str | None,
) -> dict[str, Any]:
    if crop_name:
        crop = next(
            (entry for entry in manifest if entry.get("crop_name") == crop_name),
            None,
        )
        if crop is None:
            choices = ", ".join(
                str(entry.get("crop_name")) for entry in manifest[:10]
            )
            raise ValueError(
                f"Crop {crop_name!r} is not in the manifest. Examples: {choices}"
            )
        return crop
    return next(
        (entry for entry in manifest if "_center." in str(entry.get("crop_name", ""))),
        manifest[0],
    )


def area_bbox(
    latitude: float, longitude: float, radius_meters: float,
) -> tuple[float, float, float, float]:
    """Approximate a radius-sized square in geographic coordinates."""
    latitude_delta = radius_meters / 111_320.0
    longitude_scale = 111_320.0 * math.cos(math.radians(latitude))
    if longitude_scale <= 0:
        raise ValueError("Longitude scale is invalid at this latitude.")
    longitude_delta = radius_meters / longitude_scale
    return (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )


def utm_epsg(latitude: float, longitude: float) -> int:
    """Return a WGS84 UTM CRS for the requested area."""
    if not -80 <= latitude <= 84:
        raise ValueError(
            "The 3DEP area export currently requires a location between "
            "80°S and 84°N so it can use a UTM output CRS."
        )
    zone = min(60, max(1, math.floor((longitude + 180) / 6) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def request_json(url: str, timeout: float = 60.0) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(
                    f"USGS 3DEP service error: {payload['error']}"
                )
            return payload
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(
                    f"USGS 3DEP request failed with HTTP {error.code}: "
                    f"{error.reason}"
                ) from error
        except URLError as error:
            last_error = error
            if attempt == 2:
                raise RuntimeError(
                    f"Could not reach the USGS 3DEP service: {error.reason}"
                ) from error
        time.sleep(2**attempt)
    raise RuntimeError(f"USGS 3DEP request failed: {last_error}")


def download_tiff(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    temporary_path = output_path.with_name(output_path.name + ".part")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120.0) as response:
                with temporary_path.open("wb") as file:
                    shutil.copyfileobj(response, file)
            with temporary_path.open("rb") as file:
                signature = file.read(4)
            if signature not in {
                b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+",
            }:
                raise RuntimeError(
                    "USGS 3DEP response was not a TIFF raster; check the "
                    "service status and try again."
                )
            temporary_path.replace(output_path)
            return
        except HTTPError as error:
            temporary_path.unlink(missing_ok=True)
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(
                    f"USGS 3DEP raster download failed with HTTP {error.code}: "
                    f"{error.reason}"
                ) from error
        except URLError as error:
            temporary_path.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(
                    f"Could not download the USGS 3DEP raster: {error.reason}"
                ) from error
        except (OSError, RuntimeError):
            temporary_path.unlink(missing_ok=True)
            raise
        time.sleep(2**attempt)


def export_dem(
    latitude: float,
    longitude: float,
    radius_meters: float,
    resolution_meters: float,
    output_path: Path,
) -> dict[str, Any]:
    west, south, east, north = area_bbox(
        latitude, longitude, radius_meters
    )
    dimension = math.ceil(2 * radius_meters / resolution_meters)
    if dimension > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Requested output is {dimension} pixels per side, exceeding the "
            f"USGS image service limit of {MAX_IMAGE_DIMENSION}. Increase "
            "--resolution or reduce --radius."
        )
    output_epsg = utm_epsg(latitude, longitude)
    params = urlencode({
        "bbox": f"{west:.10f},{south:.10f},{east:.10f},{north:.10f}",
        "bboxSR": "4326",
        "imageSR": str(output_epsg),
        "size": f"{dimension},{dimension}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_NearestNeighbor",
        "f": "json",
    })
    export_url = f"{SERVICE_URL}/exportImage?{params}"
    export_info = request_json(export_url)
    download_url = export_info.get("href")
    if not isinstance(download_url, str):
        raise RuntimeError(
            f"USGS 3DEP response did not contain a raster URL: {export_info}"
        )
    parsed_download_url = urlparse(download_url)
    if (
        parsed_download_url.scheme != "https"
        or parsed_download_url.hostname != "elevation.nationalmap.gov"
    ):
        raise RuntimeError(
            f"USGS 3DEP returned an unexpected raster host: "
            f"{parsed_download_url.hostname}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_tiff(download_url, output_path)

    returned_extent = export_info.get("extent", {})
    returned_spatial_reference = returned_extent.get("spatialReference", {})
    returned_width = export_info.get("width")
    returned_height = export_info.get("height")
    actual_pixel_size = None
    if (
        isinstance(returned_width, (int, float))
        and returned_width > 0
        and isinstance(returned_height, (int, float))
        and returned_height > 0
    ):
        actual_pixel_size = {
            "x_meters": abs(
                returned_extent["xmax"] - returned_extent["xmin"]
            ) / returned_width,
            "y_meters": abs(
                returned_extent["ymax"] - returned_extent["ymin"]
            ) / returned_height,
        }
    return {
        "source": SERVICE_DESCRIPTION,
        "service_url": SERVICE_URL,
        "service_description_url": f"{SERVICE_URL}?f=pjson",
        "surface": "bare-earth digital elevation model (DEM)",
        "input_center": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "radius_meters": radius_meters,
        "requested_bbox_wgs84": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "returned_dimensions_pixels": {
            "width": returned_width,
            "height": returned_height,
        },
        "returned_pixel_size_meters": actual_pixel_size,
        "returned_extent": {
            "xmin": returned_extent.get("xmin"),
            "ymin": returned_extent.get("ymin"),
            "xmax": returned_extent.get("xmax"),
            "ymax": returned_extent.get("ymax"),
            "epsg": returned_spatial_reference.get(
                "latestWkid", returned_spatial_reference.get("wkid")
            ),
        },
        "requested_resolution_meters_per_pixel": resolution_meters,
        "output_format": "32-bit floating-point GeoTIFF",
        "interpolation": "nearest neighbor",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": output_path.name,
        "notes": [
            "The service is a multi-resolution bare-earth DEM, not a surface "
            "model; buildings and tree canopy are intentionally not represented.",
            "The raster is a square bounding export enclosing the requested "
            "radius and therefore includes corner pixels beyond the radius.",
            "Requested pixel size controls the export grid; it does not "
            "guarantee the source DEM has that native resolution everywhere.",
            "The returned extent and pixel dimensions are recorded above; the "
            "service may adjust the requested extent to match the output grid.",
            "Source acquisition date and vertical datum can vary by source DEM. "
            "Check the service metadata and local source-product metadata before "
            "using elevations for survey-grade work.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Crop manifest produced by crop_perspective.py.",
    )
    parser.add_argument(
        "--crop",
        help="Crop name whose camera coordinates center the DEM request; "
        "defaults to the first center crop.",
    )
    parser.add_argument(
        "--radius", type=float, default=500.0,
        help="Half-width of the requested area around the crop coordinates, in metres.",
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Requested output pixel size in metres (default: 1 m).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for the GeoTIFF and metadata sidecar.",
    )
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("--radius must be greater than zero.")
    if args.resolution <= 0:
        parser.error("--resolution must be greater than zero.")

    crop = select_crop(load_manifest(args.manifest), args.crop)
    latitude = float(crop["lat"])
    longitude = float(crop["lng"])
    crop_stem = Path(str(crop["crop_name"])).stem
    output_path = args.output_dir / f"{crop_stem}_3dep_bare_earth.tif"
    metadata_path = output_path.with_suffix(".metadata.json")

    print(
        f"Requesting USGS 3DEP bare-earth DEM near "
        f"{latitude:.6f}, {longitude:.6f} "
        f"({args.radius:g} m radius, {args.resolution:g} m output pixels)..."
    )
    metadata = export_dem(
        latitude, longitude, args.radius, args.resolution, output_path
    )
    metadata["crop_name"] = crop.get("crop_name")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    dimensions = metadata["returned_dimensions_pixels"]
    extent = metadata["returned_extent"]
    print(f"Saved bare-earth GeoTIFF: {output_path}")
    print(f"Saved source/extent metadata: {metadata_path}")
    print(
        f"  Returned raster: {dimensions['width']} × {dimensions['height']} "
        f"pixels, EPSG:{extent['epsg']}; requested output resolution "
        f"{args.resolution:g} m/pixel."
    )
    pixel_size = metadata["returned_pixel_size_meters"]
    if pixel_size is not None:
        print(
            f"  Returned pixel size: {pixel_size['x_meters']:.3f} m east-west "
            f"× {pixel_size['y_meters']:.3f} m north-south."
        )
    print(
        "  This is bare-earth terrain elevation. It does not contain building "
        "roofs or tree canopy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
