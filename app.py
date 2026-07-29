import ollama
import markdown
from pathlib import Path
from image_retrieval import STATIC_DIR as IMAGE_STATIC_DIR


def _clean_suggestions(content: str) -> list[str]:
    """Normalize Ollama response into a list of suggestion strings."""
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        if line and line[0].isdigit() and len(line) > 1 and line[1] in ".: ":
            line = line.split(".", 1)[-1].strip()
            line = line.split(":", 1)[-1].strip()
        cleaned.append(line)
    return cleaned if cleaned else ["(No suggestions returned)"]


def _build_vision_prompt(description: str | None) -> str:
    """Prompt for the vision‑enabled Ollama model.

    If a description is provided we ask the model to combine it with the image.
    Otherwise we ask the model to infer the garment from the image alone.
    Additionally, we incorporate optional style and weather preferences from the session.
    """
    # Pull optional context from the user's session.
    from flask import session as _session
    style = _session.get("style_preference", "").strip()
    weather = _session.get("weather", "").strip()

    if description:
        prompt = (
            f"The user described their garment as: \"{description}\". "
            "Look closely at the actual garment shown in the image and suggest 3‑4 outfit pieces that pair well with THIS exact item."
        )
    else:
        prompt = (
            "Identify the garment shown in the image (type, color, style, notable details) "
            "and then suggest 3‑4 outfit pieces that would match it."
        )
    # Append optional style and weather contexts.
    if style:
        prompt += f" The user prefers a {style} style."
    if weather:
        prompt += f" The current weather is {weather}."
    return prompt


def _build_text_prompt(description: str | None) -> str:
    """Prompt for text‑only suggestions using the same Ollama model.

    This prompt does **not** reference an image, but includes optional style and weather.
    """
    from flask import session as _session
    style = _session.get("style_preference", "").strip()
    weather = _session.get("weather", "").strip()

    if description:
        prompt = (
            f"The user described their garment as: \"{description}\". "
            "Based on this description, suggest 3‑4 outfit pieces that pair well with THIS item."
        )
    else:
        prompt = "Suggest 3‑4 outfit pieces that would match a garment, without any image."
    if style:
        prompt += f" The user prefers a {style} style."
    if weather:
        prompt += f" The current weather is {weather}."
    return prompt


def generate_outfit_suggestions(image_path: Path, description: str | None = None) -> str:
    """Generate outfit suggestions using Ollama's qwen3-vl:4b vision model.

    Returns HTML markdown string containing the suggestions.
    """
    prompt_text = _build_vision_prompt(description)
    try:
        response = ollama.chat(
            model="qwen3-vl:4b",
            messages=[{"role": "user", "content": prompt_text, "images": [Path(image_path)]}],
        )
    except Exception as e:
        raise RuntimeError(f"Ollama vision model call failed: {e}")
    content = response.get("message", {}).get("content", "")
    cleaned = _clean_suggestions(content)
    # Convert list of suggestions into markdown bullet list and then HTML.
    markdown_text = "\n".join(f"- {s}" for s in cleaned)
    return markdown.markdown(markdown_text)


def generate_text_only_suggestions(description: str | None = None) -> str:
    """Generate outfit suggestions using a text‑only prompt.

    Calls the same Ollama model without providing an image.
    Returns HTML markdown string.
    """
    prompt_text = _build_text_prompt(description)
    try:
        response = ollama.chat(
            model="qwen3-vl:4b",
            messages=[{"role": "user", "content": prompt_text}],
        )
    except Exception as e:
        raise RuntimeError(f"Ollama text‑only model call failed: {e}")
    content = response.get("message", {}).get("content", "")
    cleaned = _clean_suggestions(content)
    markdown_text = "\n".join(f"- {s}" for s in cleaned)
    return markdown.markdown(markdown_text)

# Add a button on the select image page to add style/aesthetic description

import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
from generate import generate_images

app = Flask(__name__)
# Simple secret key – in production use a secure random value.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Ensure session works – set permanent session.
app.config["SESSION_PERMANENT"] = False

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    description = request.form.get("description", "").strip()
    style_preference = request.form.get("style_preference", "").strip()
    weather = request.form.get("weather", "").strip()
    if not description:
        flash("Please provide a garment description.")
        return redirect(url_for("index"))
    # Store the description and optional fields for later use.
    session["description"] = description
    session["style_preference"] = style_preference
    session["weather"] = weather
    # Generate images.
    images = generate_images(description)
    session["images"] = images
    return render_template("choose_image.html", images=images)

def call_anthropic_llm(description: str) -> list[str]:
    """Call the Anthropic-compatible API to get outfit suggestions.

    Returns a list of suggestion strings. If the API call fails, returns a
    single‑element list with an error message.
    """
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model = os.getenv("ANTHROPIC_MODEL")
    if not base_url or not model:
        return ["Anthropic API configuration is missing. Ensure ANTHROPIC_BASE_URL and ANTHROPIC_MODEL are set."]
    # Build the request payload.
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Give me 3-4 outfit suggestions for the following garment description: '{description}'. "
                    "Return each suggestion as a separate short sentence."
                ),
            }
        ],
    }
    try:
        response = requests.post(f"{base_url.rstrip('/')}/v1/messages", json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        # Anthropic response format includes a "content" list with a dict containing "text".
        content_items = data.get("content", [])
        suggestions_text = ""
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                suggestions_text += item.get("text", "")
        # Split suggestions by lines or numbered list.
        lines = [line.strip() for line in suggestions_text.strip().splitlines() if line.strip()]
        # Remove numbering if present (e.g., "1. Something")
        cleaned = []
        for line in lines:
            if line and line[0].isdigit() and (len(line) > 1 and line[1] in ".: "):
                # Strip leading number and punctuation.
                line = line.split(".", 1)[-1].strip()
                line = line.split(":", 1)[-1].strip()
            cleaned.append(line)
        return cleaned if cleaned else ["(No suggestions returned)"]
    except Exception as e:
        return [f"Error contacting Anthropic API: {e}"]

@app.route("/suggestions", methods=["POST"])
def suggestions():
    # Retrieve the chosen mode (vision or text_only). Default to vision.
    mode = request.form.get("mode", "vision")
    # DEBUG PRINT MODEL MODE
    mode = request.form.get("mode")
    print(f"MODE RECEIVED: {mode!r}", flush=True)
    # The garment description is stored in the session from the /generate step.
    description = session.get("description", "")
    # The selected image radio button is always present in the form.
    selected_image = request.form.get("selected_image")
    if mode == "text_only":
        # Text‑only mode does not need an image, but we keep selected_image for display.
        if not description:
            flash("Description is required for text‑only suggestions.")
            return redirect(url_for("index"))
        suggestions = generate_text_only_suggestions(description)
    else:
        # Vision mode (default).
        if not selected_image:
            flash("Please select an image.")
            return redirect(url_for("index"))
        session["selected_image"] = selected_image
        image_path = Path(IMAGE_STATIC_DIR) / selected_image
        suggestions = generate_outfit_suggestions(image_path, description if description else None)
    return render_template(
        "results.html",
        selected_image=selected_image,
        suggestions=suggestions,
    )

if __name__ == "__main__":
    # Run on default Flask port.
    app.run(debug=True)
