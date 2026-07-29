# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- **Setup the development environment**: `setup.bat`
  - Creates a virtual environment under `venv/` and installs all dependencies listed in `requirements.txt`.
  - After running, activate the environment with `call venv\Scripts\activate`.

- **Run the application**: `python app.py`
  - Starts the Flask development server on `http://127.0.0.1:5000/`.
  - The server reloads automatically when source files change (Flask `debug=True`).

- **Run a single test**: No project‑specific tests are bundled. If you add tests later (e.g., using `pytest`), you can run a specific test file with `pytest path/to/test_file.py`.

- **Lint/format code**: The repository does not include a linting configuration, but you can use standard tools such as `flake8` or `black` on the source files (e.g., `flake8 .` or `black .`).

## High‑Level Architecture

- **Entry point (`app.py`)**
  - Sets up a Flask app, defines routes, manages session state, and calls the LLM.
  - Routes:
    - `GET /` – renders the description entry page.
    - `POST /generate` – stores the user description, calls `generate_images`, and presents four image choices.
    - `POST /suggestions` – stores the selected image, calls the local Ollama vision model (qwen3-vl:4b) for outfit suggestions, and renders the results.
- Requires the Ollama server to be running locally and the model pulled (`ollama pull qwen3-vl:4b`).

- **Image retrieval (`image_retrieval.py`)**
  - On first use, downloads a 2000‑image subset of the `ashraq/fashion-product-images-small` dataset via the `datasets` library.
  - Computes CLIP embeddings (ViT‑B‑32, `open-clip-torch`) for each image and caches the tensor in `.image_cache/embeddings.pt` and the relative paths in `.image_cache/paths.pt`.
  - Provides `get_top_images(description, top_k=4)` which encodes the user description with the same CLIP model, performs a cosine‑similarity lookup, and returns the four most similar image paths (relative to the Flask `static/` folder).
  - Includes a legacy‑format fallback: if `paths.pt` is a pickled list (from an older version), it loads via `torch.load` and rewrites it as JSON for future runs.

- **Generate wrapper (`generate.py`)**
  - Exposes a thin wrapper `generate_images(description)` that forwards the call to `image_retrieval.get_top_images`. This keeps the Flask route signature unchanged while allowing the heavy retrieval logic to evolve independently.

- **Static assets**
  - `static/style.css` – minimal styling for the pages.
  - `static/fashion/` – holds the downloaded fashion images. These are served directly by Flask using `url_for('static', filename=…)`.

- **Templates** (`templates/`)
  - `base.html` – shared layout with a container and a stylesheet link.
  - `index.html` – form for garment description.
  - `choose_image.html` – displays the four candidate images as a radio group.
  - `results.html` – shows the selected image and a list of outfit suggestions returned by the LLM.

## Extensibility Notes
- To change the dataset size or source, modify the `num_images` argument in `image_retrieval._download_and_cache`.
- To switch the CLIP model, adjust `MODEL_NAME` and `PRETRAINED` constants in `image_retrieval.py`.
- The LLM call is isolated in `call_anthropic_llm`; swapping to a different provider only requires updating the payload and endpoint handling.

---

*All modifications to the repository should stay within the `outfit_buildr/` directory structure; no other project files are altered.*