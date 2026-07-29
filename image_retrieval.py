"""Image retrieval module using CLIP embeddings from the ashraq/fashion-product-images-small dataset.

This module is imported by ``generate.py`` and provides ``get_top_images`` which returns
four image file paths that best match a garment description. The implementation
downloads a configurable subset (default 2000 images) of the *ashraq/fashion-product-images-small* dataset,
computes CLIP embeddings for each image using ``open-clip-torch``, caches the embeddings and
the relative static‑folder paths, and on subsequent calls performs a fast cosine‑similarity
lookup.

The heavy work (downloading, preprocessing, embedding) runs only once – the first
time ``get_top_images`` is called – because the results are stored in ``.image_cache``
under the project root. Subsequent requests simply load the cached tensors,
embed the input text, and retrieve the nearest images.
"""

import json
import random
from pathlib import Path
from typing import List

import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

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


def _text_embedding(text: str) -> torch.Tensor:
    """Encode *text* with the same CLIP model and return a normalised embedding."""
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        txt_emb = _model.encode_text(tokens)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb  # Shape: [1, D]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_top_images(description: str, top_k: int = 4) -> List[str]:
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
