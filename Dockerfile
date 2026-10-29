# to stop the container, Ctrl+c in terminal

# if code changes are made, rebuild and run
# docker build -t outfit-builder .
# docker run --rm --env-file .env -e PORT=10000 -p 8000:10000 outfit-builder

# -------------------------------------------------------------
# 1️⃣  Base image – match the local Python version (3.12.10)
# -------------------------------------------------------------
FROM python:3.12-slim AS builder

# Set a deterministic locale (optional but helpful for reproducibility)
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# -------------------------------------------------------------
# 2️⃣  Install OS‑level build‑essentials needed by pip wheels
# -------------------------------------------------------------
# The wheel for Pillow already bundles libjpeg/libpng, so no extra libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# 3️⃣  Install the *exact* CPU‑only PyTorch
# -------------------------------------------------------------
# This must be installed before the rest of requirements so pip
# doesn’t pull the large CUDA wheel.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# -------------------------------------------------------------
# 4️⃣  Copy only the requirements file and install remaining deps
# -------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
# Install the rest of the requirements – the earlier torch install
# satisfies the “torch” entry, so pip will skip reinstalling.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake CLIP weights into the image. This sits above "COPY . ." so code
# changes don't trigger a re-download.
ENV HF_HOME=/app/.cache TORCH_HOME=/app/.cache
RUN python - <<'EOF'
import open_clip
open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
print("CLIP model cached")
EOF

# -------------------------------------------------------------
# 5️⃣  Copy the source code (layers after deps for cache benefit)
# -------------------------------------------------------------
COPY . .

# -------------------------------------------------------------
# 8️⃣  Runtime stage – copy only the built image (no build‑time deps)
# -------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
RUN chown appuser:appuser /app

# Copy the *built* virtual environment (site‑packages) and app code
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app /app

# Re‑apply the cache folder (weights have already been pulled)
ENV HF_HOME=/app/.cache \
    TORCH_HOME=/app/.cache \
    HF_HUB_OFFLINE=1

# -------------------------------------------------------------
# 9️⃣  Switch to non‑root user
# -------------------------------------------------------------
USER appuser

# -------------------------------------------------------------
# 🔟  Entrypoint – let Render provide $PORT, fall back to 10000
# -------------------------------------------------------------
EXPOSE 10000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
