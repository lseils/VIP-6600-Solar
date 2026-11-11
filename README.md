# Balcony Plug-and-Play Solar PV Assessment

A framework for assessing balcony Plug-and-Play solar photovoltaic potential at
the neighborhood scale.

![Framework diagram](images/Framework.png)

- **Author:** Lydia Seils
- **Program:** M.S. Architecture, Computational Design, Georgia Tech
- **LinkedIn:**
  [linkedin.com/in/lydia-seils](https://www.linkedin.com/in/lydia-seils)
- **Email:** <Lydiamseils@gmail.com>
- **Portfolio:** <https://lseils.github.io/>

## Overview

This repository includes the following project components:

1. **[Gather_Balcony](Gather_Balcony/)**: Pipeline for extracting Google Street
   View (GSV) images and generating 3D geometry.
2. **[RANSAC](RANSAC/)**: Scripts for creating planar surfaces from
   reconstructed balcony geometry.
3. **To_Train_YOLO**: Material for training YOLOv8.
4. **[Yolo_Balcony_Detector](Yolo_Balcony_Detector/)**: Balcony detection model.
5. **[Ladybug](Ladybug/)**: Grasshopper script for solar irradiance analysis.
6. **[Documentation](Documentation/)**: Project notes and slide deck.

## About

Balconies on residential buildings provide an underutilized opportunity for
decentralized energy generation through Plug-and-Play (PnP) photovoltaic (PV)
systems. Realizing this potential at the neighborhood scale requires knowing
where viable balconies exist and how much solar energy they can generate.

Current urban-scale solar assessment frameworks have made significant progress
using open-source data for as-built geometry generation and solar analysis.
However, balconies have often been neglected and require street-view imagery as
a key data source. This project asks whether open-source 2D street imagery can
be used to reconstruct balcony geometry and assess PnP solar potential in
underserved urban neighborhoods.

The proposed open-source framework could serve as an accessible tool for
planners and researchers, supporting broader adoption of PnP systems for
low-income renters.

## Methods

1. **Image extraction**: Collect Google Street View imagery for the study area.
2. **Detection and segmentation**: Detect balcony elements in street-view
   imagery with a trained YOLOv8 model.
3. **Geometry reconstruction**: Convert image data into 3D mesh geometry, then
   use RANSAC to create planar balcony surfaces.
4. **PV placement**: Estimate the maximum number of PnP systems that can fit on
   each balcony and generate PV panel surfaces with the correct dimensions.
5. **Solar analysis**: Calculate solar irradiance with Ladybug inside
   Rhino/Grasshopper.
6. **Evaluation**: Review detection, reconstruction, and solar-analysis outputs.

## Usage

### Setup

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
git submodule update --init   # fetch Yolo_Balcony_Detector
uv sync                       # create .venv from pyproject.toml / uv.lock
cp .env.example .env          # then add your GOOGLE_MAPS_API_KEY
```

The depth scripts download Depth Anything V2 Large (~2.5 GB) from Hugging Face
on first use and cache it in `~/.cache/huggingface`. To fetch it ahead of time:

```bash
uv run hf download depth-anything/Depth-Anything-V2-Large-hf
```

Run scripts with `uv run python <script>.py` (or activate `.venv`). Linux
installs CUDA 12.6 PyTorch wheels; macOS uses the Apple GPU (MPS) or CPU.

> [!NOTE]
> `run_colmap.sh` uses a `colmap` binary from `PATH` if present, otherwise the
> PACE cluster Apptainer image. Its dense steps need an NVIDIA GPU (CUDA). On a
> machine with neither (e.g. macOS), `run_pipeline.py`'s `colmap` step runs the
> sparse reconstruction with the pip `pycolmap` package instead, which is what
> `depth_to_pointcloud.py` needs; use Docker or the cluster for COLMAP's dense
> `fused.ply`. `requirements.txt` is a legacy `pip freeze` from
> the cluster and is not used by uv.

### Docker (Linux with an NVIDIA GPU)

The image bundles CUDA COLMAP and the Python environment. It requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t vip-solar .
docker run --rm -it --gpus all --env-file .env -v "$PWD":/app \
  -v hf-cache:/root/.cache/huggingface vip-solar
```

The `hf-cache` volume keeps downloaded model weights between container runs.

The container starts in `Gather_Balcony`; run `python <script>.py` and
`./run_colmap.sh` directly.

### Run the pipeline for a site

Put a site's coordinates, facade bearing and camera settings in a YAML file
(see `Gather_Balcony/sites/midtown_example.yaml`), then from `Gather_Balcony`:

```bash
uv run python run_pipeline.py sites/midtown_example.yaml
```

Steps (`fetch`, `crop`, `colmap`, `depth`, `postprocess`) write to
`outputs/<name>/` and are skipped when their outputs already exist. Pick steps
with `--steps crop postprocess`; re-run finished ones with `--force`. The
runner calls the existing scripts unchanged; the manual commands below still
work.

### Gather balcony images and geometry

Run the following commands from the `Gather_Balcony` directory:

```bash
# Clear old data from previous runs.
rm -rf street_images colmap_workspace perspective_crops

# Scrape images.
uv run python fetch_streetview_tiles.py

# Prepare images for COLMAP.
uv run python crop_perspective.py

# Run COLMAP.
./run_colmap.sh

# Run post-processing.
uv run python open3d_postprocess.py
```

`crop_perspective.py` reads panoramas from `Gather_Balcony/street_images/`
when its metadata file is present; otherwise, it falls back to the repository
root's `street_images/`. It always writes crops to
`Gather_Balcony/perspective_crops/`, where the neighborhood scene builder
expects them.

### Download USGS 3DEP bare-earth elevation

After generating `crop_manifest.json`, request the 3DEP bare-earth DEM around
the camera location for the same default 300 m radius used by the building
visualization:

```bash
python fetch_lidar_elevation.py --crop pano_000_center.jpg
```

The script saves a clipped GeoTIFF and a JSON metadata sidecar in
`Gather_Balcony/lidar_elevation/`. Use `--radius` to change the radius,
`--resolution` to change the requested output pixel size, and `--output-dir` to
choose another location. The GeoTIFF can be inspected in QGIS or another GIS
tool. Its metadata sidecar records the actual returned extent, CRS, dimensions,
pixel size, source service, and retrieval time.

This is the USGS 3DEP **bare-earth DEM**, appropriate for terrain elevation;
it intentionally does not include building roofs or tree canopy. The service
returns a square bounding raster centered on the crop. The service is
multi-resolution: requesting 1 m output pixels does not guarantee 1 m native
source data at every location. Acquisition dates and vertical datums can vary
among source DEMs, so review the sidecar's source notes and local product
metadata before using elevations as survey-grade measurements. The output is
a separate input layer and does not change the building visualization yet.

### Build a combined neighborhood scene

The scene orchestrator runs the USGS 3DEP elevation step, fetches Overture
building features, samples bare-earth ground elevation beneath each footprint,
and creates both an interactive verification view and an OBJ mesh importable in
Rhino:

```bash
python -m pip install rasterio==1.5.2 laspy==2.6.1 lazrs==0.7.0 "shapely>=2.0"
python build_neighborhood_scene.py --crop pano_000_center.jpg
```

The scene builder defaults to a 500 m radius around the crop; override it with
`--radius` when needed.

By default, outputs are saved under
`Gather_Balcony/neighborhood_scenes/`: a terrain GeoTIFF and metadata sidecar,
an interactive HTML scene, a combined OBJ, and scene metadata JSON. The OBJ
uses local east/north metres centered on the selected crop; vertical values
are metres relative to bare-earth elevation at that origin. The terrain is
downsampled for a manageable mesh, and terrain triangles may be further
decimated in the HTML view only. Buildings are extruded from sampled ground
elevation. Overture heights or labeled estimates are retained unless nearby
class-6 USGS LiDAR points support a 3D DBSCAN cluster. Point clusters are
assigned to the nearest footprint within 8 m. The per-building DBSCAN epsilon
is twice the mean distance to the three nearest horizontal neighbors; DBSCAN
then uses east, north, and elevation so distinct roof levels can remain
separate. LiDAR outlines use a simplified concave hull, and roof height is the
95th-percentile aligned inlier elevation rather than the highest return.
Overlapping roof regions at similar heights are merged; higher regions take
precedence where they overlap, while uncovered Overture footprint areas keep
their original height. Roof caps use constrained triangulation where
available. Clusters need at least 20 points. This creates closed stepped
height regions, not sloped roof surfaces. The Overture
footprint lineage may include OpenStreetMap; this orchestrator does not query
OSM directly. Buildings are selected when their full footprint intersects the
requested radius and are rendered with the complete footprint, even where it
extends beyond the terrain mesh. LiDAR tile discovery includes the terrain
area and a buffered bounding area around all selected buildings; raw class-6
returns from that area can therefore support height regions across the full
building. The terrain, canopy surface, and class-separated point exports
remain clipped to the downloaded DEM area.

Class-6 returns are also checked for buildings missing from the footprint
dataset. Candidate roof clusters must be inside the scene boundary,
have a median height of at least 3 m above the DEM, include at least 20
clustered points, and cover at least 8 m². Points already assigned to a mapped
footprint within 8 m are excluded from unmatched-cluster construction to avoid
duplicate shells. Remaining clusters are built into the same building layer
as Overture footprints using the same concave-outline and robust roof-height
rules.

### View Google Photorealistic 3D Tiles

Google's Map Tiles API can stream its photorealistic 3D tiles into a local
CesiumJS viewer centered on a generated neighborhood scene. This is a
visualization only: the viewer does not download, save, or export Google's
tile data or derive geometry from it.
It uses a plain background instead of the Cesium globe, limits zoom-out, and
recenters after a pan if the view center leaves the scene boundary. Tiles may
appear outside the boundary near the camera-frustum edges, and may briefly
appear beyond it while dragging. Google still serves tiles through its global
streaming endpoint; the viewer limits navigation, it does not clip the data or
create a downloadable regional dataset.

Enable the **Map Tiles API** and billing for your Google Cloud project. The
browser-facing API key must be permitted for Map Tiles API and restricted to
the local viewer referrer (for example `http://127.0.0.1:8765/*`). Store the
key as `GOOGLE_MAPS_API_KEY` in the repository `.env` file; the script serves
the key to the local browser at runtime and does not write it into a file.

From the repository root, run:

```bash
python Gather_Balcony/view_google_3d_tiles.py \
  --scene-json Gather_Balcony/neighborhood_scenes/pano_000_center_neighborhood_scene.json
```

The script opens a viewer at `http://127.0.0.1:8765/`; press Ctrl+C in the
terminal to stop it. It reads the scene origin and boundary from the JSON.
Specify `--scene-json` for another crop, `--radius` to override the recorded
scene radius in metres, or `--port` to use another localhost port. Google
requires attribution to remain visible in the viewer; Cesium displays the
tile-provided credits on screen. Map Tiles API use requires billing and is
subject to Google's current pricing and terms. See the [API guide](https://developers.google.com/maps/documentation/tile/3d-tiles),
[policies](https://developers.google.com/maps/documentation/tile/policies),
and [pricing](https://developers.google.com/maps/billing-and-pricing/pricing#maps-3d-tiles-pricing).

### Compare Overture and LiDAR buildings with Google tiles

The comparison viewer overlays the local Overture footprints and detected
class-6 LiDAR roof footprints on Google's streamed 3D tiles. It computes
intersection-over-union, Overture-only/LiDAR-only footprint areas, and
Overture-versus-LiDAR height differences from those local datasets. It writes
a CSV review report beside the scene JSON. Google tiles are visual context
only; the script does not read their geometry or use it to calculate
disagreements.

The scene must contain the local comparison geometries. Rebuild an older scene
once with the current builder, then start the viewer from the repository root:

```bash
python Gather_Balcony/build_neighborhood_scene.py --crop pano_000_center.jpg
python Gather_Balcony/compare_google_buildings.py
```

By default, the viewer uses `http://127.0.0.1:8765/` and writes
`pano_000_center_neighborhood_scene_building_comparison.csv` beside the scene
JSON. Use `--scene-json`, `--report`, or `--port` to select other paths or a
port. LiDAR roof returns may cover only part of a building, so low overlap or
an Overture-only area is a review signal, not proof that the Overture footprint
is wrong. LiDAR-only candidates also require visual verification.

### Grounding DINO + SAM balcony overlays

The neighborhood scene builder can reuse the existing Grounding DINO and SAM
inference helpers in `Grounding_DINO/run_demo.py` and
`Grounding_DINO/run_sam.py`:

```bash
python Gather_Balcony/build_neighborhood_scene.py \
  --crop pano_000_center.jpg \
  --segment-balconies
```

The default `Grounding_DINO/grounding_dino_config.json` supplies the model
IDs, text prompt, and detection thresholds. The selected perspective crop must
already exist at `Gather_Balcony/perspective_crops/<crop-name>`; use
`--segmentation-image` to provide another path, or
`--grounding-dino-config` to select a different prompt/configuration. The
option is off by default so normal scene builds do not download or load the
models. PyTorch and Transformers must be installed in the active environment;
the model weights are downloaded on first use. A compatible CUDA-enabled
PyTorch install is needed for GPU inference.

SAM masks are projected through the crop's calibrated camera onto the nearest
visible building-mesh surfaces. The HTML scene displays colored surface
overlays, and the OBJ exports them as separately named mesh objects (without
embedded materials), while a 2D mask preview is saved as
`<crop-stem>_balcony_segmentation.png`. This is a visualization of image masks
on the existing building geometry, not a reconstruction of balcony depth or
structural geometry. If the scene mesh does not cover the detected image
region, that mask will have no 3D overlay; inspect the saved 2D preview and
the detection-level projection counts in the scene JSON.

The scene builder also checks the USGS TNM Access catalog for **3DEP LiDAR
point-cloud (LAZ) tiles** intersecting the expanded LiDAR search area. It selects the
newest-dated project among the intersecting projects, downloads its matching
tiles, and caches them under `neighborhood_scenes/point_cloud/`. Coverage
percentage in the scene metadata is estimated from catalog tile bounds; it is
not a guarantee that every location contains classified returns. The point
cloud is distinct from the bare-earth DEM above.

Vegetation meshes and the shade-focused point preview use LAS classes 3, 4,
and 5 (low, medium, and high vegetation), but exclude returns below 2 m above
the DEM by default so short grass and shrubs do not affect balcony shade.
Override that threshold with `--minimum-shade-relevant-height`. Class 5 is a
tree-candidate proxy, not a guarantee of individual tree species or crown-only
returns. Points up to 60 m above the DEM are clustered with DBSCAN; its `eps` is twice the mean distance
to the three nearest neighbors measured from streamed LAS chunks, and
`min_samples` is 5. To limit memory for 0.5 km scenes, clustering is partitioned
into 100 m spatial cells with a 25 m overlap; a deterministic reservoir sample
keeps at most 5,000 points per class per cell, and at most 3,000 points per
class are passed to each buffered DBSCAN fit. Clusters need at least 20 points.
Each cluster is represented by a porous ellipsoidal crown sized from its
5th-95th percentile LiDAR horizontal extents. The crown is a tapered layered
profile; when the observed vertical return thickness is too small to describe
a tree crown, its height is conservatively inferred from the horizontal crown
diameter. Crown surface patches are sampled deterministically with
class-specific foliage densities: 0.25 for low, 0.35 for medium, and 0.50 for
high vegetation. The gaps are a geometric transmission approximation, not a
calibrated leaf-area-index model. The disconnected class 3-5 crown meshes are
grouped into separate vegetation-class render meshes.

Use `--vegetation-point-preview` to make the sampled, shade-relevant class 3-5
returns visible in the HTML viewer at once instead of requiring legend toggles.
The preview remains sampled; the clipped class-separated LAZ files retain the
full classified returns.

Per-tile vertical offsets are estimated from class-2 ground returns against the
DEM before clustering. The crown dimensions remain an approximation: a merged
DBSCAN cluster can represent multiple trees, while a sparse cluster can
underestimate the crown. Review the rendered crowns and point preview before
using them for solar conclusions.

No solar predictions or seasonal tree-removal behavior are applied in this
Python geometry pipeline. The clustered meshes are geometry for visualization
and export, not survey-grade tree crowns. Review project date, classification,
point density, and coverage before using them for analysis.

The scene builder also writes AOI-clipped, class-separated LAZ files beneath
`neighborhood_scenes/lidar_points/`: `ground` (class 2), `low_vegetation`
(class 3), `medium_vegetation` (class 4), `high_vegetation` (class 5),
`building` (class 6), and `other` (all remaining classifications). Only points
that map to valid pixels in the terrain raster are included. These files retain
the source tile's coordinates, elevations, and LAS classification values; use
the scene metadata JSON to find their paths and point counts. The HTML
visualization displays up to 2,500 sampled points per class by default, with
each class independently toggleable in the legend. Use
`--max-lidar-visualization-points 0` to omit the HTML point traces while still
writing the LAZ files. Raw points are not added to the Rhino OBJ; that export
continues to contain the terrain, LiDAR-adjusted and LiDAR-detected building
meshes, constructed class 3-5 vegetation meshes, and supplied mesh layers. Class-6
returns are used to adjust the building geometry in Python; the raw points
remain available in the separate `building` LAZ exports.

The HTML view includes the estimated camera marker, crop field-of-view
frustum, and a target tracker from `building_hits.json`. The target OSM
reference is matched against Overture source metadata and the footprint is
highlighted when a match is available; otherwise the tracker remains visible
and reports that no Overture match was found. Use `--hits path/to/hits.json`
to select another hit file and `--camera-height` to change the estimated
camera height. The tracker is for visualization; it does not add camera or
target geometry to the Rhino OBJ.

The scene format includes `terrain`, `buildings`, `trees`, `balconies`, and
`other` layers. Future segmented balcony meshes can be supplied with
`--extra-meshes path/to/meshes.json`. That JSON is a list of mesh objects with
a `name`, `layer`, `vertices`, and `faces`; vertices are `[east, north,
elevation]` metre triples in the scene coordinate system, and faces are
zero-based triangle indices. Balcony geometry must first be aligned to the
scene origin and terrain-relative vertical datum. Tree meshes can use the same
interface for additional tree data or balconies after transforming them into
scene coordinates. No balcony data is fetched by the current pipeline.

### Visualize a camera and nearby buildings

From `Gather_Balcony`, after generating the crop manifest, identify the OSM
building hit and create an interactive 3D view:

```bash
python locate_building.py
python visualize_building_camera.py --crop pano_000_center.jpg
```

The HTML output is written to `3d_outputs/`. By default it uses Overture
building features and heights only—no OSM API or Overpass query is made—and
shows the estimated camera location and crop's field-of-view frustum. Overture
source data may itself include OpenStreetMap contributions. It embeds the
selected perspective crop when available, otherwise the source panorama. The
image is a reference panel only; it is not projected onto the model. An
internet connection is required. Install the Overture client in the active Python
environment with `python -m pip install -r ../requirements.txt` (or
`python -m pip install overturemaps==1.0.2` if installing only that dependency).
The script downloads only Overture building data within the selected camera
radius and filters the bounding-box result to the actual radius. The scene
hover text shows Overture height provenance and confidence when available.
`--overture-lidar-only` keeps only buildings with direct heights attributed to
USGS LiDAR. To compare with the earlier OSM-footprints plus Overture-heights
view, pass `--include-osm`; combine it with `--no-overture` to show OSM only.
Use `--overture-release RELEASE` to select a release. Image folders are
searched both in the current directory and beside the script; pass
`--crop-dir` or `--panorama-dir` if your images are elsewhere. When Overture
has no direct height, its floor/roof fields are used for labeled estimates;
features without height metadata use the labeled 9-meter fallback.

For a USGS LiDAR-height-only comparison, run:

```bash
python visualize_building_camera.py --crop pano_000_center.jpg --overture-lidar-only
```

This keeps only Overture features whose direct height has USGS LiDAR
provenance, and extrudes them using that reported height. Overture's building
polygon geometry is not guaranteed to be derived from LiDAR; its provenance is
identified in the scene. This is not a reconstruction directly from raw LiDAR
point clouds.

### Run balcony detection

1. Add input images to the `Yolo_Balcony_Detector/input/images` folder.
2. Confirm the correct paths in `Yolo_Balcony_Detector/detect.py`.
3. Run detection from the `Yolo_Balcony_Detector` directory:

```bash
python detect.py
```

The detection outputs are written to the `detections` folder.

The relevant `detect.py` configuration is:

```python
import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    model = YOLO("runs/detect/train3/weights/best.pt")
    model.predict(
        source="input/images",
        imgsz=640,
        project="detections",
        name="exp",
        show_labels=True,
        save=True,
    )
```

### Run Grounding DINO and SAM segmentation

The experimental Grounding DINO pipeline uses each DINO balcony box as a box
prompt for SAM. Prepare the COCO annotations first, then run from the
repository root:

```bash
python Grounding_DINO/prepare_dataset.py
python Grounding_DINO/run_demo.py
```

The SAM model is configured in
`Grounding_DINO/grounding_dino_config.json`. The first run downloads the DINO
and SAM checkpoints from Hugging Face. Results are written to
`Grounding_DINO/demo_outputs`, including binary masks in `masks/`, comparison
panels in `visualizations/`, and matched detection/mask IoU in `metrics.json`.

### Reconstruct geometry with RANSAC

1. Open the generated mesh in Rhinoceros 3D.
2. Run `RANSAC/PY-RANSAC.py` in the Rhino script editor.
3. Run `RANSAC/PV-Placement.py` in the Rhino script editor.

### Run solar analysis

Run `Ladybug/IncidentRadiation.gh` in Grasshopper inside Rhino.

## Results

- Automatic image extraction from the Google API was successful, efficient, and
  free to use.
- The detection model required training and resulted in 77.1% precision and
  63.7% recall.
- The 2D-to-3D conversion outputs a mesh but lacks even semi-accurate depth.
- RANSAC creates similarly dimensioned planes for balconies.
- The Python scripts can generate PV panels on balconies with ease.
- The Ladybug simulation produces possible results but has not been validated.

## Discussion

The research showed that this framework could help automate a manual task, but
several steps still have unresolved problems. Image extraction and detection
work reasonably well, but the extracted images affect the 2D-to-3D conversion
step. Angle, position, and field of view all need to be considered when scraping
imagery.

There were issues with photogrammetry calculations when scraping the Google
Street View Static API, so the workflow pivoted to scraping Google Street View
tiles. Tools such as COLMAP handle warping better when the input is full
panoramas with associated metadata rather than pre-cropped panorama images.
Getting panoramas into the correct format for COLMAP remains a work in progress.

The Python RANSAC and PV placement workflow is not perfectly automated and still
requires manual adjustment, but it faces fewer obstacles than the
image-to-geometry workflow. For the full framework, much is promising, but the
most important step, converting 2D street imagery to 3D geometry, is still in
progress.

## Acknowledgement of Artificial Intelligence Use

This project was developed with assistance from Claude (Anthropic), which was
used as an AI pair programmer throughout the development process. Claude
assisted with debugging, code explanation, understanding model architecture, and
general guidance on the YOLOv8 training pipeline. Research ideas, data
gathering, labeling, and conclusions were completed by the author.

The author assumes full responsibility for the validity, reproducibility, and
integrity of all work presented.

## References

- Chen, Y., Wang, T., Chen, G., Zhu, K., Tan, X., Wang, J., Guo, W., Wang, Q.,
  Luo, X., & Zhang, X. (2025). BFA-YOLO: A balanced multiscale object detection
  network for building façade elements detection. _Advanced Engineering
  Informatics_, 65, 103289. <https://doi.org/10.1016/j.aei.2025.103289>
- Claude, version 4.6 Sonnet, Anthropic.
- Kong, G., & Fan, H. (2021). Enhanced Facade Parsing for Street-Level Images
  Using Convolutional Neural Networks. _IEEE Transactions on Geoscience and
  Remote Sensing_, 59(12), 10519-10531.
  <https://doi.org/10.1109/TGRS.2020.3035878>
- Ultralytics. YOLOv8. Version 8.0, 2023.
  <https://github.com/ultralytics/ultralytics>
- X-AnyLabeling. Version 2.5.3, Viet-Anh Nguyen, 2026.
  <https://github.com/vietanhdev/anylabeling>
- Zhang, G., Pan, Y., & Zhang, L. (2022). Deep learning for detecting building
  façade elements from images considering prior knowledge. _Automation in
  Construction_, 133, 104016. <https://doi.org/10.1016/j.autcon.2021.104016>
