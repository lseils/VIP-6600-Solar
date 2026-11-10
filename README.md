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
