#!/usr/bin/env python3
"""
fetch_streetview_tiles.py
--------------------------
Downloads Google Street View tiles along a dense 4-meter path and saves
FULL equirectangular panoramas (no cropping). Cropping is handled separately
by crop_perspective.py.

Outputs:
  street_images/pano_000.jpg   ← raw equirectangular panoramas
  panorama_metadata.json       ← GPS + heading for each pano
"""

import os
import io
import json
import math
import requests
from pathlib import Path
from dotenv import load_dotenv

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install pillow")
    exit(1)

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY not found in .env")

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_FOLDER  = "street_images"
METADATA_FILE  = "panorama_metadata.json"
ZOOM_LEVEL     = 3      # Zoom 3 → 8x4 tiles → 4096x2048 equirectangular

# Only the FIRST and LAST coordinate matter — the script interpolates every
# 4 meters between them automatically.
PATH_COORDINATES = [
    (33.764771, -84.382069),  # pano_005 — south end of brick building
    (33.765126, -84.382061),  # pano_009 — north end of brick building
]


# =============================================================================
# STEP 1: Session token
# =============================================================================
def get_session_token(api_key: str) -> str:
    print("Getting session token...")
    url = "https://tile.googleapis.com/v1/createSession"
    payload = {"mapType": "streetview", "language": "en-US", "region": "US"}
    r = requests.post(f"{url}?key={api_key}", json=payload)
    if not r.ok:
        print(r.status_code, r.text)
    r.raise_for_status()
    token = r.json()["session"]
    print(f"[OK] Session token obtained")
    return token


# =============================================================================
# STEP 2: Interpolate path → unique pano IDs
# =============================================================================
def get_dense_pano_ids(api_key: str, start_coord, end_coord,
                        step_meters: float = 4.0) -> list:
    print(f"\nCalculating dense path every {step_meters}m...")
    lat1, lon1 = start_coord
    lat2, lon2 = end_coord

    avg_lat        = math.radians((lat1 + lat2) / 2.0)
    lat_diff_m     = (lat2 - lat1) * 111139.0
    lon_diff_m     = (lon2 - lon1) * (111139.0 * math.cos(avg_lat))
    total_distance = math.sqrt(lat_diff_m**2 + lon_diff_m**2)
    num_steps      = max(int(total_distance / step_meters), 1)

    print(f"Total distance: ~{total_distance:.1f}m → sampling {num_steps + 1} points")

    unique_panos = []
    for i in range(num_steps + 1):
        frac        = i / float(num_steps)
        current_lat = lat1 + (lat2 - lat1) * frac
        current_lon = lon1 + (lon2 - lon1) * frac

        resp = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location": f"{current_lat},{current_lon}",
                    "key": api_key, "radius": 15}
        )
        data = resp.json()

        if data.get("status") == "OK":
            pano_id = data["pano_id"]
            if pano_id not in unique_panos:
                unique_panos.append(pano_id)
                print(f"  [+] New pano: {pano_id[:12]}... (step {i})")
        else:
            print(f"  [-] No pano at step {i} ({current_lat:.5f}, {current_lon:.5f})")

    print(f"[OK] {len(unique_panos)} unique panoramas found")
    return unique_panos


# =============================================================================
# STEP 3: Real GPS metadata for a pano ID
# =============================================================================
def get_pano_metadata(api_key: str, session: str, pano_id: str):
    """Returns (lat, lng, heading) or None."""
    url = (
        f"https://tile.googleapis.com/v1/streetview/metadata"
        f"?session={session}&key={api_key}&panoId={pano_id}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    meta = r.json()
    if "lat" not in meta or "lng" not in meta:
        return None
    return meta["lat"], meta["lng"], meta.get("heading", 0.0)


# =============================================================================
# STEP 4: Download all tiles and stitch into a raw equirectangular panorama
# =============================================================================
def download_and_stitch_panorama(api_key: str, session: str, pano_id: str,
                                  output_folder: str, index: int) -> str:
    """
    Downloads every tile at ZOOM_LEVEL and stitches them into a single
    equirectangular image. At zoom 3: 8 columns × 4 rows of 512×512 tiles
    → 4096×2048 output.

    Returns the saved filename, or None on failure.
    """
    zoom  = ZOOM_LEVEL
    num_x = 2 ** zoom        # 8 columns
    num_y = 2 ** (zoom - 1)  # 4 rows

    # Probe the first tile to get the actual tile size (256 or 512)
    probe_url = (
        f"https://tile.googleapis.com/v1/streetview/tiles"
        f"/{zoom}/0/0"
        f"?session={session}&key={api_key}&panoId={pano_id}"
    )
    probe = requests.get(probe_url)
    if probe.status_code != 200:
        print(f"    [ERROR] Could not fetch probe tile (HTTP {probe.status_code})")
        return None

    probe_img  = Image.open(io.BytesIO(probe.content))
    tile_w, tile_h = probe_img.size
    print(f"    Tile size: {tile_w}×{tile_h}px  →  "
          f"Panorama will be {num_x * tile_w}×{num_y * tile_h}px")

    # Build blank canvas and paste tiles
    pano_img = Image.new("RGB", (num_x * tile_w, num_y * tile_h))
    pano_img.paste(probe_img, (0, 0))   # reuse the probe tile

    failed = 0
    for y in range(num_y):
        for x in range(num_x):
            if x == 0 and y == 0:
                continue  # already pasted above
            tile_url = (
                f"https://tile.googleapis.com/v1/streetview/tiles"
                f"/{zoom}/{x}/{y}"
                f"?session={session}&key={api_key}&panoId={pano_id}"
            )
            r = requests.get(tile_url)
            if r.status_code == 200:
                tile_img = Image.open(io.BytesIO(r.content))
                pano_img.paste(tile_img, (x * tile_w, y * tile_h))
            else:
                print(f"    [WARN] Tile ({x},{y}) failed with HTTP {r.status_code}")
                failed += 1

    if failed > (num_x * num_y) // 4:
        print(f"    [ERROR] Too many failed tiles ({failed}), skipping pano")
        return None

    fname = f"pano_{index:03d}.jpg"
    fpath = os.path.join(output_folder, fname)
    pano_img.save(fpath, quality=95)
    print(f"    [OK] Saved raw panorama → {fname}")
    return fname


# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    session      = get_session_token(API_KEY)
    start_coord  = PATH_COORDINATES[0]
    end_coord    = PATH_COORDINATES[-1]
    pano_list    = get_dense_pano_ids(API_KEY, start_coord, end_coord, step_meters=4.0)

    all_metadata = []
    print(f"\nDownloading and stitching {len(pano_list)} panoramas...\n")

    for index, pano_id in enumerate(pano_list):
        print(f"[{index+1}/{len(pano_list)}] Pano {pano_id[:12]}...")

        meta = get_pano_metadata(API_KEY, session, pano_id)
        if meta is None:
            print(f"  [SKIP] No metadata for {pano_id[:12]}")
            continue

        real_lat, real_lng, car_heading = meta
        print(f"  GPS: ({real_lat:.6f}, {real_lng:.6f})  Car heading: {car_heading:.1f}°")

        fname = download_and_stitch_panorama(
            API_KEY, session, pano_id, OUTPUT_FOLDER, index
        )
        if fname is None:
            continue

        all_metadata.append({
            "image_name":  fname,
            "pano_index":  index,
            "pano_id":     pano_id,
            "lat":         real_lat,
            "lng":         real_lng,
            "car_heading": car_heading,
        })

    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Done! {len(all_metadata)} panoramas saved to '{OUTPUT_FOLDER}/'")
    print(f"  Metadata → {METADATA_FILE}")
    print(f"  Next step: python crop_perspective.py")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()