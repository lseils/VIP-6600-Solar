# Contributing to VIP-6600-Solar

Thanks for working on this project! A few guidelines to keep things running smoothly as a team.

## Cloning the repo

This repo uses a **git submodule** (`Yolo_Balcony_Detector`). Make sure to clone with `--recursive`, or the submodule folder will be empty:

```bash
git clone --recursive https://github.com/SustainableUrbanSystemsLab/VIP-6600-Solar.git
```

If you already cloned without `--recursive`, run this instead:

```bash
git submodule update --init --recursive
```

## Branch structure

This repo uses multiple branches:

```
main  ← stable, always working. Requires approval from Lydia or Patrick to merge.
 └── dev  ← shared integration/testing branch for the team
      ├── lydia/wip
      ├── "teammate"/wip
      ├── etc..
      └── etc..
```

- **`main`** — always stable and working. Only updated periodically from `dev`, once things have been tested together. Protected — only Lydia or Patrick can approve a merge into `main`.
- **`dev`** — the shared branch everyone works against. This is where your feature branches get merged first, so we can catch conflicts and integration issues before they ever touch `main`.
- **Feature branches** — where you actually do your work day-to-day.

**Never commit directly to `main` or `dev`.** All changes go through a branch and a pull request.

Create a new branch off `dev` for whatever you're working on:

```bash
git checkout dev
git pull
git checkout -b feature/short-description
```

Branch naming convention:
- `feature/short-description` — new functionality
- `fix/short-description` — bug fixes
- `yourname/whatever` — personal experiments, analysis, or anything that doesn't fit the above

## Making changes

Commit as you go with clear messages:

```bash
git add .
git commit -m "Add RANSAC parameter tuning for edge cases"
git push -u origin feature/short-description
```

If your branch has been open more than a day or two, pull the latest `dev` into it to avoid conflicts piling up:

```bash
git pull origin dev
```

## Opening a Pull Request

**Feature branch → `dev`:**
1. Push your branch, then open a Pull Request on GitHub into `dev`.
2. Write a short description of what changed and why.
3. Get it reviewed/merged — this is the lower-stakes integration step, so it moves faster than a `main` merge.

## Submodules

- `Yolo_Balcony_Detector` is a submodule with its own separate repo and history. If you need to update it, make and commit your changes inside that submodule's repo first, then come back to `VIP-6600-Solar` and commit the updated submodule pointer.
- `Gather_Balcony` is **not** a submodule — it's regular tracked files in this repo. Edit it like any other folder here.

## Questions

If git gets into a confusing state (merge conflicts, detached HEAD, etc.), stop and ask before trying to force your way out of it — it's much easier to fix early than after more commands are run on top of it.


# Setting up Gather_Balcony for own work - Using Computing Cluster

## Requirements

- **Python**: 3.11 (via conda — see note below on why not plain `venv`)
- **Cluster modules**:
  ```bash
  module load colmap/3.10
  ```
  (COLMAP runs via Apptainer container with CUDA support — no manual install needed. Run on a GPU-allocated node for CUDA-dependent steps.)

- **Cluster Settings**
  - VS Code version: 1.99.2
  - Environment setup: Default Modules
  - Quality of service: Inferno
  - NVIDIA GPU: V100 16GB
  - Number of cores: 4
  - Number of GPUs: 1
  - Memory: 16GB

⚠️ **Important — use scratch storage, not home.** Home directories on this cluster have a hard **20GB quota**. This project's dependencies (PyTorch, CUDA libraries, etc.) alone take up 10GB+, and the quota fills fast once you add the repo, caches, and other packages. Clone the repo and build the environment under your scratch path (e.g. `/storage/scratch1/<group>/<username>/`), **not** your home directory.

## Setup

1. **Clone the repo directly onto scratch, and `cd` into it:**
   ```bash
   cd /storage/scratch1/<group>/<your_username>
   git clone <repo_url>
   cd VIP-6600-Solar
   ```

2. **Load required modules:**
   ```bash
   module load colmap/3.10
   ```

3. **Install miniconda on scratch (skip if you already have conda pointed at scratch).**
   If your `conda info --base` shows a path under `/storage/home/...`, redirect it — home-based conda installs will hit the quota wall:
   ```bash
   cd /storage/scratch1/<group>/<your_username>
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p /storage/scratch1/<group>/<your_username>/miniconda3
   /storage/scratch1/<group>/<your_username>/miniconda3/bin/conda init bash
   source ~/.bashrc
   ```
   Also point pip's cache to scratch so it doesn't quietly refill your home quota over time:
   ```bash
   echo 'export PIP_CACHE_DIR=/storage/scratch1/<group>/<your_username>/pip_cache' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Create and activate the conda environment (Python 3.11):**
   ```bash
   conda create -n vip6600 python=3.11 -c conda-forge -y
   conda activate vip6600
   conda install pip -c conda-forge -y
   ```

5. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify GPU support.** PyTorch's default build may target a newer CUDA version than this node's driver supports:
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
   - If this prints `True`, you're done.
   - If it prints `False` with a *"CUDA initialization: driver too old"* warning, reinstall torch against CUDA 12.6, which is compatible with the V100 nodes on this cluster:
     ```bash
     pip uninstall torch torchvision -y
     pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126
     ```
     Re-run the check above to confirm `True` before proceeding.

7. **Confirm the environment is consistent:**
   ```bash
   pip check
   ```
   Should report "No broken requirements found."

## Notes for troubleshooting

- **Landed back on `(base)` after reconnecting?** Compute node allocations can end/rotate (new node hostname each time). You'll need to `conda activate vip6600` again each new session — it doesn't stay active across a fresh SSH connection.
- **"Disk quota exceeded" errors** mean something is installing to home instead of scratch — check `conda info --base` and `df -h ~` / `quota -s`.
- **requirements.txt conflicts:** if you ever need to regenerate this file, don't hand-edit strict `==` pins from a `pip freeze` created on a different machine/Python version — this caused a long chain of resolver conflicts (numpy vs scipy vs matplotlib vs torch/xformers) when the file was first set up. Prefer freezing directly from a cluster environment that's confirmed working.