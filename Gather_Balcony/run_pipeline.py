#!/usr/bin/env python3
"""
Run the Gather_Balcony pipeline for one site config.

    uv run python run_pipeline.py sites/midtown_example.yaml
    uv run python run_pipeline.py sites/midtown_example.yaml --steps crop postprocess
    uv run python run_pipeline.py sites/midtown_example.yaml --force

Each step runs inside outputs/<name>/ and is skipped if its output exists.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEPS = ["fetch", "crop", "colmap", "depth", "postprocess"]


def run_main(module, argv=()):
    """Call a script's main() with its own argv, as if run from the shell."""
    sys.argv = [module.__file__, *argv]
    module.main()


def fetch(cfg):
    import fetch_streetview_tiles as m
    m.PATH_COORDINATES = [tuple(c) for c in cfg["path_coordinates"]]
    m.ZOOM_LEVEL = cfg.get("zoom_level", m.ZOOM_LEVEL)
    run_main(m)


def crop(cfg):
    import crop_perspective as m
    m.TARGET_BEARING = cfg["target_bearing"]
    m.PITCH_DEG = cfg.get("pitch_deg", m.PITCH_DEG)
    m.FOV_DEG = cfg.get("fov_deg", m.FOV_DEG)
    run_main(m)


def colmap(cfg):
    if shutil.which("colmap") or shutil.which("apptainer"):
        subprocess.run(["bash", str(HERE / "run_colmap.sh")], check=True)
    else:
        colmap_sparse_pycolmap()


def colmap_sparse_pycolmap():
    """Sparse-only COLMAP via pip pycolmap (CPU), mirroring run_colmap.sh steps 1-3.

    shortcut: no dense MVS here (patch_match_stereo needs CUDA), use Docker/cluster for fused.ply.
    """
    import pycolmap

    ws = Path("colmap_workspace")
    db, images, sparse = ws / "database.db", Path("perspective_crops"), ws / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)

    reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
    extraction = pycolmap.FeatureExtractionOptions(max_image_size=2048)
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              reader_options=reader, extraction_options=extraction)
    pycolmap.match_exhaustive(db)

    opts = pycolmap.IncrementalPipelineOptions(multiple_models=False)
    opts.mapper.init_min_num_inliers = 15
    opts.mapper.abs_pose_min_num_inliers = 12
    opts.mapper.abs_pose_min_inlier_ratio = 0.25
    recs = pycolmap.incremental_mapping(db, images, sparse, opts)
    if not recs:
        sys.exit("[ERROR] COLMAP could not register enough images for a sparse model")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    best.write_text(str(sparse / "0"))  # depth_to_pointcloud.py reads the .txt model
    print(f"[OK] Sparse model: {best.num_reg_images()} images, {best.num_points3D()} points → {sparse / '0'}")


def depth(cfg):
    import depth_to_pointcloud as m
    run_main(m)


def postprocess(cfg):
    import open3d_postprocess as m
    fused = Path("colmap_workspace/dense/fused.ply")
    src = fused if fused.exists() else Path("colmap_workspace/dense_depth/dense_from_depth.ply")
    run_main(m, ["--input", str(src), "--output_dir", "colmap_workspace/mesh"])


# Output whose existence means the step is already done.
DONE_MARKER = {
    "fetch": "panorama_metadata.json",
    "crop": "perspective_crops",
    "colmap": "colmap_workspace/sparse/0",
    "depth": "colmap_workspace/dense_depth/dense_from_depth.ply",
    "postprocess": "colmap_workspace/mesh",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--steps", nargs="+", choices=STEPS)
    p.add_argument("--force", action="store_true", help="re-run steps whose outputs exist")
    args = p.parse_args()
    args.config = args.config.resolve()  # before chdir

    cfg = yaml.safe_load(args.config.read_text())
    steps = args.steps or cfg.get("steps", STEPS)
    out = HERE / "outputs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)

    for step in steps:
        marker = Path(DONE_MARKER[step])
        if not args.force and marker.exists() and (marker.is_file() or any(marker.iterdir())):
            print(f"[skip] {step}: {marker} exists")
            continue
        if len(steps) == 1:
            print(f"\n===== {step} ({out}) =====", flush=True)
            globals()[step](cfg)
        else:
            # One process per step: frees model/GPU memory and isolates native crashes.
            subprocess.run([sys.executable, __file__, str(args.config),
                            "--steps", step, "--force"], check=True)
    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
