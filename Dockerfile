# Linux + NVIDIA GPU image: CUDA COLMAP plus the Python pipeline environment.
# The repository is bind-mounted at /app at run time (see README), so only the
# dependency files are baked in.
FROM colmap/colmap:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Runtime libraries needed by opencv-python and open3d
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the venv outside /app so the bind mount does not hide it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen

WORKDIR /app/Gather_Balcony
ENTRYPOINT []
CMD ["bash"]
