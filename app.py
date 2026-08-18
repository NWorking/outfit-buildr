import ollama
import os
import json
import re
import requests
import time
from pathlib import Path
import base64
import mimetypes
from flask import Flask, render_template, request, redirect, url_for, session, flash
from image_retrieval import STATIC_DIR as IMAGE_STATIC_DIR, get_top_images
from image_retrieval import get_top_images as generate_images
from groq import Groq

from pydantic import BaseModel, ValidationError

class Suggestion(BaseModel):
    title: str
    reasoning: str
    pieces: list[str]

# Helper to parse JSON responses from the LLM
def _parse_json_response(content: str) -> list[dict]:
    """Extract a JSON array from the LLM output.
    Strips optional markdown code fences and parses the JSON.
    Returns a list of dictionaries; on failure returns a fallback list with raw content.
    """
    # Remove possible markdown fences (```json ... ```)
    cleaned = re.sub(r'^\s*```json\s*', '', content, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "suggestions" in data:
            data = data["suggestions"]  # unwrap Groq's schema wrapper
        if not isinstance(data, list):
            print('Unexpected JSON structure:', data)
            return []
        validated = [Suggestion.model_validate(item).model_dump() for item in data]
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        print('JSON parse/validation error:', e)
        return [{"title": "Raw suggestion", "reasoning": "", "pieces": [cleaned]}]



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


def _encode_image(image_path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


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
            "Look closely at the actual garment shown in the image and suggest 4 different outfit combinations "
            "that pair well with THIS item, each with a different overall vibe."
        )
    else:
        prompt = (
            "Identify the garment shown in the image (type, color, style, notable details) "
            "and then suggest outfit pieces that would match it."
        )
    # Append optional style and weather contexts.
    if style:
        prompt += f" The user prefers a {style} style."
    if weather:
        prompt += f" The current weather is {weather}."
    prompt += (
        "\n\nDo NOT include the original garment itself in the \"pieces\" list — "
        "only list complementary items that would be worn with it.\n\n"
        "Respond with ONLY a JSON array, no other text before or after, "
        "no markdown code fences. Follow this exact format:\n"
        '[\n'
        '  {"title": "Golden Hour Glow", "reasoning": "Warm neutrals and soft '
        'textures complement the piece without competing.", "pieces": '
        '["tan suede ankle boots", "oversized cardigan", "gold hoop earrings"]},\n'
        '  {"title": "...", "reasoning": "...", "pieces": ["...", "..."]}\n'
        ']\n'
        "Return exactly 4 objects in the array. Each object's \"pieces\" list "
        "should have 2-8 short, concrete, searchable item descriptions."
    )
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
            "Based on this description, suggest 4 different outfit combinations "
            "that pair well with THIS item, each with a different overall vibe."
        )
    else:
        prompt = "Suggest 3‑4 outfit pieces that would match a garment, without any image."
    if style:
        prompt += f" The user prefers a {style} style."
    if weather:
        prompt += f" The current weather is {weather}."

    prompt += (
        "\n\nDo NOT include the original garment itself in the \"pieces\" list — "
        "only list other items in the outfit you come up with that would be worn with the original garment.\n\n"
        "Respond with ONLY the JSON array, no other text before or after, "
        "no markdown code fences. Follow the format defined by schema."
        
    )
    return prompt


def generate_outfit_suggestions(image_path: Path | None = None, description: str = "", schema:dict | None = None, provider: str = "ollama", response_format: dict | None = None, image_b64: str | None = None) -> tuple[list[dict], float]:
    """Generate outfit suggestions using Ollama's vision model.
    Returns a list of structured suggestion dictionaries.
    """
    prompt_text = _build_vision_prompt(description)

    if provider == "groq":
        # Use Groq's text‑only API. Build the same prompt as the text‑only
        # path (style and weather context are pulled inside _build_text_prompt)
        # and request a JSON‑formatted response.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        # Model choice – currently the only model from groq that supports image as input
        prompt_text = _build_vision_prompt(description)
        if image_path is not None:
            image_data_url = _encode_image(image_path)
        elif image_b64 is not None:
            image_data_url = image_b64
        else:
            raise RuntimeError("No image data provided for vision model")
        try:
            precall_time = time.time()
            client = Groq(api_key=api_key)
            # Step 1: Vision model — no JSON constraint, just describe what it sees
            vision_response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",   # currently the only model from groq that supports image as input
                temperature=0.9,
                messages=[
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                    }
                ],
            )
            postcall_time = time.time()
            thinking_time = round(postcall_time - precall_time, 2)
            # Extract the assistant message content
            vision_description = vision_response.choices[0].message.content

            # Step 2: Text model — strict schema enforcement, structures the description
            structuring_response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "Convert the following outfit descriptions into the required structured format."},
                    {"role": "user", "content": vision_description}
                ],
                response_format=response_format,  # your strict json_schema format from before
            )
            content = structuring_response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")
        # Parse the JSON output from the LLM (same as Ollama path)
        suggestions = _parse_json_response(content)
        return suggestions, thinking_time
    
    # Ollama
    else:
        try:
            precall_time = time.time()
            response = ollama.chat(
                model="qwen3-vl:2b",
                messages=[{"role": "user", "content": prompt_text, "images": [str(image_path)]}],
                think=False,
                format=schema,
                options={"num_ctx": 8192}
            )
            postcall_time = time.time()
            thinking_time = round(postcall_time - precall_time, 2)
        except Exception as e:
            raise RuntimeError(f"Ollama vision model call failed: {e}")
        content = response.get("message", {}).get("content", "")
        # Parse JSON output from LLM
        suggestions = _parse_json_response(content)
        return suggestions, thinking_time
        

def generate_text_only_suggestions(description: str, schema: dict | None = None, provider="ollama", response_format: dict | None = None) -> list[dict]:
    """Generate outfit suggestions using a text‑only prompt.
    Returns a list of structured suggestion dictionaries.
    """
    prompt_text = _build_text_prompt(description)

    if provider == "groq":
            # Use Groq's text‑only API. Build the same prompt as the text‑only
            # path (style and weather context are pulled inside _build_text_prompt)
            # and request a JSON‑formatted response.
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY environment variable not set")
            # Model choice – a fast, current chat model from Groq. Adjust if needed.
            groq_model = "openai/gpt-oss-20b"
            
            try:
                precall_time = time.time()
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    response_format=response_format,
                )
                postcall_time = time.time()
                thinking_time = round(postcall_time - precall_time, 2)
                # Extract the assistant message content
                content = response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"Groq API call failed: {e}")
            # Parse the JSON output from the LLM (same as Ollama path)
            suggestions = _parse_json_response(content)
            return suggestions, thinking_time
    # Ollama
    else:
        try:
            precall_time = time.time()
            response = ollama.chat(
                model="qwen3-vl:2b",
                messages=[{"role": "user", "content": prompt_text}],
                think="low",
                format=schema,
                options={"num_ctx": 8192}
            )
            postcall_time = time.time()
            thinking_time = round(postcall_time - precall_time, 2)
            print("RAW RESPONSE:", response, flush=True)
            print("TIME: ", thinking_time, flush = True)
        except Exception as e:
            raise RuntimeError(f"Ollama text‑only model call failed: {e}")
        content = response.get("message", {}).get("content", "")
        suggestions = _parse_json_response(content)
        return suggestions, thinking_time

# Add a button on the select image page to add style/aesthetic description

import re
from image_retrieval import STATIC_DIR as IMAGE_STATIC_DIR, get_top_images

app = Flask(__name__)
# Simple secret key – in production use a secure random value.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Ensure session works – set permanent session.
app.config["SESSION_PERMANENT"] = False
# Limit upload size to 5 MB to prevent excessive memory usage.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB


# Groq JSON output enforcer - only compatible with a few models (none support image input)
# compatible models: https://console.groq.com/docs/structured-outputs
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "outfit_suggestions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "pieces": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 8
                            }
                        },
                        "required": ["title", "reasoning", "pieces"],
                        "additionalProperties": False
                    },
                    "minItems": 4,
                    "maxItems": 4
                }
            },
            "required": ["suggestions"],
            "additionalProperties": False
        }
    }
    }

# Passes into Ollama's format parameter
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "reasoning": {"type": "string"},
            "pieces": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 8
            }
        },
        "required": ["title", "reasoning", "pieces"]
    },
    "minItems": 4,
    "maxItems": 4
}

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
        suggestions, thinking_time = generate_text_only_suggestions(description, schema, provider="ollama", response_format=response_format)
        # Store structured suggestions for later visualization
        session["outfit_suggestions"] = suggestions
        # Text‑only mode does not have a selected image
        selected_image = None
    else:
        # Vision mode (default).
        if not selected_image:
            flash("Please select an image.")
            return redirect(url_for("index"))
        session["selected_image"] = selected_image
        image_path = Path(IMAGE_STATIC_DIR) / selected_image
        suggestions, thinking_time = generate_outfit_suggestions(image_path, description if description else None, schema=schema, provider="groq", response_format=response_format)
        # Store structured suggestions for later visualization
        session["outfit_suggestions"] = suggestions
    # Render results page with suggestions (and optional selected image)
    return render_template(
        "results.html",
        selected_image=selected_image,
        suggestions=suggestions,
        thinking_time = thinking_time,
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file is too large (max 5 MB).")
    return redirect(url_for('index'))

@app.route("/upload_suggestions", methods=["POST"])
def upload_suggestions():
    # Handle direct image upload, bypassing CLIP candidate selection.
    uploaded_file = request.files.get("photo")
    if not uploaded_file:
        flash("No image file uploaded.")
        return redirect(url_for("index"))
    # Validate MIME type is an image.
    if not uploaded_file.mimetype or not uploaded_file.mimetype.startswith('image/'):
        flash("Uploaded file must be an image.")
        return redirect(url_for("index"))
    # Read file bytes into memory and base64‑encode.
    file_bytes = uploaded_file.read()
    if not file_bytes:
        flash("Uploaded image is empty.")
        return redirect(url_for("index"))
    b64_str = base64.b64encode(file_bytes).decode('utf-8')
    # Build data URI for LLM (Groq vision model expects a data URL).
    image_data_uri = f"data:{uploaded_file.mimetype};base64,{b64_str}"
    # Preserve optional style and weather fields.
    style_preference = request.form.get('style_preference', '').strip()
    weather = request.form.get('weather', '').strip()
    session['style_preference'] = style_preference
    session['weather'] = weather
    # No textual description needed for pure image upload.
    description = ''
    # Call vision model via Groq (provider='groq') – it supports image data URIs.
    suggestions, thinking_time = generate_outfit_suggestions(
        image_path=None,
        description=description,
        schema=schema,
        provider='groq',
        response_format=response_format,
        image_b64=image_data_uri,
    )
    # Store suggestions for later visualization.
    session['outfit_suggestions'] = suggestions
    # Render results directly, passing the base64 string for display.
    return render_template(
        'results.html',
        selected_image=None,
        uploaded_image_b64=b64_str,
        suggestions=suggestions,
        thinking_time=thinking_time,
    )


@app.route("/visualize", methods=["POST"])
def visualize():
    idx_str = request.form.get("suggestion_index")
    try:
        idx = int(idx_str)
    except Exception:
        flash("Invalid suggestion selection.")
        return redirect(url_for("index"))
    all_suggestions = session.get("outfit_suggestions", [])
    if idx < 0 or idx >= len(all_suggestions):
        flash("Suggestion index out of range.")
        return redirect(url_for("index"))
    suggestion = all_suggestions[idx]
    pieces = suggestion.get("pieces", [])
    items = []
    for piece in pieces:
        top = get_top_images(piece, top_k=1)
        image = top[0] if top else None
        items.append({"piece": piece, "image": image})
    uploaded_image_b64 = request.form.get('uploaded_image_b64')
    return render_template(
        "visualize.html",
        suggestion=suggestion,
        items=items,
        uploaded_image_b64=uploaded_image_b64,
    )

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == "__main__":
    # Run on default Flask port.
    app.run(debug=True)
