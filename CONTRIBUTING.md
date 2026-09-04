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
main  ← stable, always working. Requires approval from Lydia to merge.
 └── dev  ← shared integration/testing branch for the team
      ├── lydia/wip
      ├── "teammate"/wip
      ├── etc..
      └── etc..
```

- **`main`** — always stable and working. Only updated periodically from `dev`, once things have been tested together. Protected — only Lydia can approve a merge into `main`.
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
