"""Image retrieval module using CLIP embeddings from the ashraq/fashion-product-images-small dataset.

This module provides ``get_top_images`` which returns
four image file paths that best match a garment description. The implementation
downloads a configurable subset (default 2000 images) of the *ashraq/fashion-product-images-small* dataset,
computes CLIP embeddings for each image using ``open-clip-torch``, caches the embeddings and
the relative static-folder paths, and on subsequent calls performs a fast cosine-similarity
lookup.

The heavy work (downloading, preprocessing, embedding) runs only once - the first
time ``get_top_images`` is called - because the results are stored in ``.image_cache``
under the project root. Subsequent requests simply load the cached tensors,
embed the input text, and retrieve the nearest images.
"""

import json
from datetime import datetime, timezone
import random
from pathlib import Path
from typing import List

import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

# New imports for live image search
import os
import requests
import io
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_SUBDIR = "fashion"  # Images will be served from ``static/fashion/``
STATIC_DIR = PROJECT_ROOT / "static"
CACHE_DIR = PROJECT_ROOT / ".image_cache"
CACHE_VERSION = "v2"
EMB_PATH = CACHE_DIR / f"embeddings_{CACHE_VERSION}.pt"
PATHS_PATH = CACHE_DIR / f"paths_{CACHE_VERSION}.json"
METADATA_PATH = CACHE_DIR / f"metadata_{CACHE_VERSION}.json"

# Path for live‑search cache (description -> image URL)
LIVE_CACHE_PATH = CACHE_DIR / f"live_image_cache_{CACHE_VERSION}.json"

# Directory for low‑similarity logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directories exist.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model setup – a lightweight ViT‑B‑32 CLIP model.
# ---------------------------------------------------------------------------
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = torch.device("cpu")  # Use CPU for maximum portability.

_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
)
_model = _model.to(DEVICE).eval()
_preprocess = _preprocess
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_and_cache(num_images: int = 2000) -> tuple[torch.Tensor, List[str]]:
    """Download a subset of ashraq/fashion-product-images-small, compute CLIP embeddings, and cache.

    The subset is stratified across ``articleType`` to ensure representation of rare categories.
    """
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    # Group indices by articleType
    article_to_indices = {}
    for idx, sample in enumerate(ds):
        article = sample.get("articleType")
        if article is None:
            continue
        article_to_indices.setdefault(article, []).append(idx)
    categories = list(article_to_indices.keys())
    # Determine target per category (even distribution, at least one per category)
    target_per_category = max(1, num_images // len(categories))
    selected_indices = []
    random.seed(42)
    for article, indices in article_to_indices.items():
        k = min(target_per_category, len(indices))
        selected_indices.extend(random.sample(indices, k))
    # Fill up remaining needed images with random selection from the rest
    remaining_needed = num_images - len(selected_indices)
    if remaining_needed > 0:
        all_indices = set(range(len(ds)))
        remaining_pool = list(all_indices - set(selected_indices))
        if remaining_needed <= len(remaining_pool):
            selected_indices.extend(random.sample(remaining_pool, remaining_needed))
        else:
            selected_indices.extend(remaining_pool)
    subset = ds.select(selected_indices)

    embeddings: List[torch.Tensor] = []
    rel_paths: List[str] = []
    metadata: List[dict] = []

    for idx, sample in tqdm(enumerate(subset), total=len(subset), desc="Downloading & embedding images"):
        pil_img = sample["image"].convert("RGB")
        filename = f"img_{idx:04d}.jpg"
        img_path = STATIC_DIR / filename
        pil_img.save(img_path)
        rel_path = f"{STATIC_SUBDIR}/{filename}"
        rel_paths.append(rel_path)
        # Store selected metadata fields (cheap to keep)
        metadata.append({
            "path": rel_path,
            "productDisplayName": sample.get("productDisplayName"),
            "articleType": sample.get("articleType"),
            "baseColour": sample.get("baseColour"),
            "gender": sample.get("gender"),
            "season": sample.get("season"),
            "year": sample.get("year"),
        })
        # Compute CLIP embedding.
        img_tensor = _preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = _model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2‑normalise
        embeddings.append(emb.squeeze(0))

    img_embeddings = torch.stack(embeddings)  # Shape: [N, D]
    # Cache to disk.
    torch.save(img_embeddings, EMB_PATH)
    with open(PATHS_PATH, "w", encoding="utf-8") as f:
        json.dump(rel_paths, f)
    # Store metadata (optional, not used by current logic).
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return img_embeddings, rel_paths


def _load_cache() -> tuple[torch.Tensor, List[str]]:
    """Load cached embeddings and paths, building them on first use.

    Handles legacy pickle‑based ``paths.pt`` files by falling back to ``torch.load``
    and then rewriting the data as JSON for future runs. Respects the new cache version.
    """
    if EMB_PATH.is_file() and PATHS_PATH.is_file():
        embeddings = torch.load(EMB_PATH)
        try:
            with open(PATHS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # New format may be a list of dicts with a "path" key.
            if isinstance(data, list) and data and isinstance(data[0], dict) and "path" in data[0]:
                rel_paths = [item["path"] for item in data]
            else:
                rel_paths = data  # Assume plain list of strings.
        except Exception:
            # Legacy format – load the pickled list of paths.
            rel_paths = torch.load(PATHS_PATH)
            if not isinstance(rel_paths, list) or not all(isinstance(p, str) for p in rel_paths):
                return _download_and_cache()
            # Rewrite as JSON (plain list) for future loads.
            with open(PATHS_PATH, "w", encoding="utf-8") as f:
                json.dump(rel_paths, f)
        return embeddings, rel_paths
    # Cache miss – compute now.
    return _download_and_cache()


def _load_live_cache() -> dict:
    """Load the JSON cache that maps piece descriptions to a remote image URL.
    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    if LIVE_CACHE_PATH.is_file():
        try:
            with open(LIVE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_live_cache(cache: dict) -> None:
    """Write the live‑search cache back to disk.
    Overwrites the whole file atomically via a temporary write.
    """
    try:
        with open(LIVE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Live Image] Failed to write cache: {e}")


def _text_embedding(text: str) -> torch.Tensor:
    """Encode *text* with the same CLIP model and return a normalised embedding."""
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        txt_emb = _model.encode_text(tokens)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb  # Shape: [1, D]

def get_piece_image(description: str) -> str:
    """Return the best‑matching image URL for a garment *description*.

    The function follows these steps:
    1. Check a local JSON cache (``live_image_cache_…``) - if present, return it.
    2. Call SerpApi's Google Images endpoint (engine=google_images) using the ``SERPAPI_KEY`` environment variable. Extract the ``images_results`` list.
    3. Take the first few results (default 5) and re-rank them with the CLIP model against the *description* text. The highest-similarity image URL (the ``original`` field) is selected.
    4. If the API call fails, returns no results, or none of the candidates can be processed, fall back to the existing local ``get_top_images`` (top_k=1).
    5. Cache the successful mapping (description → image URL) for future calls.

    Returns a string URL - either a remote ``http(s)://`` URL or a relative static path (e.g. ``"fashion/img_0012.jpg"``) when the CLIP fallback is used.
    """
    # 1️⃣ Cache check
    cache = _load_live_cache()
    if description in cache:
        print(f"[Live Image] Cache hit for '{description}'")
        return cache[description]
    # 2️⃣ Prepare SerpApi request
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("[Live Image] SERPAPI_KEY not set - using CLIP fallback")
        fallback = get_top_images(description, top_k=1)
        return fallback[0] if fallback else ""
    try:
        params = {
            "engine": "google_images",
            "q": description,
            "api_key": api_key,
            "num": "10",
        }
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("images_results", [])
    except Exception as e:
        print(f"[Live Image] API request failed for '{description}': {e}")
        results = []
    # 3️⃣ No results → fallback
    if not results:
        print(f"[Live Image] No results from SerpApi for '{description}' - CLIP fallback")
        fallback = get_top_images(description, top_k=1)
        url = fallback[0] if fallback else ""
        return url
    # 4️⃣ Re‑rank using CLIP (limit to first few candidates)
    top_candidates = results[:5]
    txt_emb = _text_embedding(description)  # Shape: [1, D]
    best_score = -float("inf")
    best_url = None
    for cand in top_candidates:
        img_url = cand.get("original") or cand.get("thumbnail")
        if not img_url:
            continue
        try:
            img_resp = requests.get(img_url, timeout=10)
            img_resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            img_tensor = _preprocess(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_emb = _model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = (img_emb @ txt_emb.T).item()
            if sim > best_score:
                best_score = sim
                best_url = img_url
        except Exception as e:
            print(f"[Live Image] Failed to process candidate {img_url}: {e}")
            continue
    if best_url:
        print(f"[Live Image] Selected URL for '{description}' (score {best_score:.4f})")
        cache[description] = best_url
        _save_live_cache(cache)
        return best_url
    # 5️⃣ If all candidates failed, fallback to CLIP cache
    print(f"[Live Image] All candidates failed for '{description}' - CLIP fallback")
    fallback = get_top_images(description, top_k=1)
    url = fallback[0] if fallback else ""
    return url

def get_top_images(description: str, top_k: int = 4, threshold: float = 0.25) -> List[str]:
    """Return the *top_k* image paths whose CLIP embeddings are closest to *description*.

    Args:
        description: Free‑form garment description supplied by the user.
        top_k: Number of images to return (default 4 to match the UI).
    Returns:
        List of relative static‑folder paths (e.g. ``"fashion/img_0012.jpg"``).
    """
    embeddings, rel_paths = _load_cache()
    txt_emb = _text_embedding(description)  # Shape: [1, D]
    # Cosine similarity = dot product because both sides are L2‑normalised.
    sims = (embeddings @ txt_emb.T).squeeze(1)  # Shape: [N]

    k = min(top_k, sims.shape[0])
    top_indices = torch.topk(sims, k=k, largest=True).indices.tolist()
    return [rel_paths[i] for i in top_indices]
    

"""End of image_retrieval module."""
