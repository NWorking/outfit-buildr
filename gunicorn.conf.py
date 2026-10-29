import os

# Bind to the port Render injects via the PORT env var.
# If $PORT is not set (e.g. local testing) fall back to 10000.
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# One worker keeps memory usage low on the free tier (≈512 MiB RAM).
workers = 1

# A few threads let the worker handle concurrent requests (LLM calls can be slow).
threads = 4

# The LLM / image‑search calls can take a while, so give generous timeout.
timeout = 120

# Pre‑load the app (helps the first request be fast, and ensures the CLIP model is ready).
preload_app = True

# Disable access logs to keep the container quiet (optional – remove if you want logs).
accesslog = "-"
errorlog = "-"
loglevel = "info"
