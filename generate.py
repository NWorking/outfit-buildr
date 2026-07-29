"""Image retrieval module using CLIP embeddings from the Fashion‑1K dataset.

The public function ``generate_images`` retains the original signature expected by the
Flask routes. It simply forwards the description to ``get_top_images`` which performs
the CLIP text‑image similarity lookup and returns the four most relevant image paths.
"""

from image_retrieval import get_top_images

def generate_images(description: str) -> list[str]:
    """Return four image paths that best match the garment description.

    The heavy lifting (dataset download, embedding caching, similarity search) lives in
    ``image_retrieval.get_top_images``.
    """
    return get_top_images(description, top_k=4)
