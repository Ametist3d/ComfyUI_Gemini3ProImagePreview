"""
ComfyUI Node for Google Gemini 3 Pro Image (Nano Banana Pro).
Single unified node with chat/refine mode support via metadata.
Uses direct HTTP requests to Google's Gemini API with file-based API key loading.
"""

import asyncio
import base64
import os
from io import BytesIO
from typing import Any

import aiohttp
import numpy as np
import torch
import math
from PIL import Image
from pydantic import BaseModel, Field

from comfy_api.latest import IO, ComfyExtension, Input
from typing_extensions import override

# -------------------------------------------------------------------------
# Pydantic Models for Gemini API
# -------------------------------------------------------------------------


class GeminiInlineData(BaseModel):
    data: str | None = Field(None, description="Base64 encoded data")
    mimeType: str | None = Field(None)


class GeminiPart(BaseModel):
    """Gemini API Part with support for thought signatures in multi-turn conversations."""
    model_config = {"extra": "allow"}  # Preserve unknown fields like thoughtSignature
    
    inlineData: GeminiInlineData | None = Field(None)
    text: str | None = Field(None)
    thought: bool | None = Field(None)  # Marks thought/reasoning parts
    thoughtSignature: str | None = Field(None)  # Required for images in multi-turn


class GeminiContent(BaseModel):
    """Gemini API Content with support for preserving all fields."""
    model_config = {"extra": "allow"}  # Preserve unknown fields
    
    parts: list[GeminiPart | dict] = Field(default_factory=list)  # Allow raw dicts too
    role: str = Field(default="user")


class GeminiTextPart(BaseModel):
    text: str | None = Field(None)


class GeminiSystemInstructionContent(BaseModel):
    parts: list[GeminiTextPart] = Field(default_factory=list)
    role: str | None = Field(None)


class GeminiImageConfig(BaseModel):
    aspectRatio: str | None = Field(None)
    imageSize: str | None = Field(None)

class GeminiGenerationConfig(BaseModel):
    """Generation config for text LLM."""
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    topP: float | None = Field(None, ge=0.0, le=1.0)
    topK: int | None = Field(None, ge=1)
    maxOutputTokens: int | None = Field(None, ge=1)
    seed: int | None = Field(None)


class GeminiGoogleSearch(BaseModel):
    """Google Search tool configuration."""
    pass  # Empty object enables Google Search


class GeminiTool(BaseModel):
    """Tool definition for Gemini API."""
    googleSearch: GeminiGoogleSearch | None = Field(None)


class GeminiLLMGenerateContentRequest(BaseModel):
    """Request for Gemini LLM API."""
    contents: list[GeminiContent | dict] = Field(default_factory=list)
    generationConfig: GeminiGenerationConfig | None = Field(None)
    systemInstruction: GeminiSystemInstructionContent | None = Field(None)
    tools: list[GeminiTool] | None = Field(None)


class GeminiThinkingConfig(BaseModel):
    thinkingLevel: str | None = Field(None)       # "minimal" or "High" (API is case-sensitive in examples)
    includeThoughts: bool | None = Field(None)


class GeminiImageGenerationConfig(BaseModel):
    responseModalities: list[str] | None = Field(None)
    imageConfig: GeminiImageConfig | None = Field(None)
    thinkingConfig: GeminiThinkingConfig | None = Field(None)  
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    seed: int | None = Field(None)


class GeminiImageGenerateContentRequest(BaseModel):
    contents: list[GeminiContent] = Field(default_factory=list)
    generationConfig: GeminiImageGenerationConfig | None = Field(None)
    systemInstruction: GeminiSystemInstructionContent | None = Field(None)



# -------------------------------------------------------------------------
# API Key Loading (same as gemini_3_pro_image_node.py)
# -------------------------------------------------------------------------

_cached_api_key: str | None = None
_cached_api_key_source: str = "missing"


def _clean_gemini_key(raw: str) -> str:
    """Clean up possible 'NAME=VALUE' formats or quotes."""
    if not raw:
        return ""
    s = raw.strip().strip('"').strip("'")
    if "=" in s:
        name, val = s.split("=", 1)
        if "KEY" in name.upper() or "API" in name.upper():
            s = val.strip().strip('"').strip("'")
    return s


def _load_api_key() -> tuple[str, str]:
    """
    Load API key with priority:
    1. ~/comfyui_google_api_key.env
    2. .env next to this node file
    
    Returns: (key, source) where source is "home_file", "local_env", or "missing"
    """
    global _cached_api_key, _cached_api_key_source
    
    if _cached_api_key:
        return _cached_api_key, _cached_api_key_source
    
    key = ""
    source = "missing"
    
    # 1) Home directory file
    home_path = os.path.expanduser("~")
    if not os.path.isdir(home_path):
        home_path = os.environ.get("HOME", "/root")
    
    home_key_path = os.path.join(home_path, "comfyui_google_api_key.env")
    
    try:
        if os.path.isfile(home_key_path):
            with open(home_key_path, "r", encoding="utf-8") as f:
                k = _clean_gemini_key(f.read())
                if k:
                    key = k
                    source = "home_file"
    except Exception:
        pass
    
    # 2) Local .env in node directory
    if not key:
        p = os.path.dirname(os.path.realpath(__file__))
        local_env = os.path.join(p, ".env")
        try:
            if os.path.isfile(local_env):
                with open(local_env, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            k_name, v = line.split("=", 1)
                            if k_name.strip() == "GEMINI_API_KEY":
                                k = _clean_gemini_key(v)
                                if k:
                                    key = k
                                    source = "local_env"
                                    break
        except Exception:
            pass
    
    _cached_api_key = key
    _cached_api_key_source = source
    print(f"[Gemini] API key loaded: {bool(key)} | source={source}")
    return key, source


def get_api_key() -> str:
    """Get the API key, raising an error if not found."""
    key, source = _load_api_key()
    if not key:
        raise ValueError(
            "Gemini API Key is missing.\n"
            "Expected in ~/comfyui_google_api_key.env or .env (GEMINI_API_KEY=...)."
        )
    return key


# -------------------------------------------------------------------------
# Image Conversion Helpers
# -------------------------------------------------------------------------


def tensor_to_base64(tensor: torch.Tensor, mime_type: str = "image/png") -> str:
    """Convert a ComfyUI image tensor to base64 string."""
    if tensor is None:
        raise ValueError("Input tensor is None")
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    arr = tensor.detach().cpu().numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    
    pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = BytesIO()
    fmt = "PNG" if "png" in mime_type.lower() else "JPEG"
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(data: str) -> torch.Tensor:
    """Convert base64 image data to ComfyUI tensor."""
    image_data = base64.b64decode(data)
    pil_img = Image.open(BytesIO(image_data)).convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def pil_to_base64(pil_img: Image.Image, mime_type: str = "image/png") -> str:
    """Convert PIL image to base64 string."""
    buf = BytesIO()
    fmt = "PNG" if "png" in mime_type.lower() else "JPEG"
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------------------------------------------------
# Response Parsing Helpers
# -------------------------------------------------------------------------


def extract_images_from_response(response_data: dict) -> list[torch.Tensor]:
    """Extract image tensors from Gemini API response."""
    images = []
    candidates = response_data.get("candidates", [])
    
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            inline_data = part.get("inlineData")
            if inline_data:
                mime_type = inline_data.get("mimeType", "")
                if mime_type.startswith("image/"):
                    # Skip thought images (they have thought=True or specific markers)
                    if part.get("thought"):
                        continue
                    data = inline_data.get("data", "")
                    if data:
                        try:
                            tensor = base64_to_tensor(data)
                            images.append(tensor)
                        except Exception as e:
                            print(f"[Gemini] Failed to decode image: {e}")
    
    return images


def extract_text_from_response(response_data: dict) -> str:
    """Extract text from Gemini API response."""
    texts = []
    candidates = response_data.get("candidates", [])
    
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)
    
    return "\n".join(texts)


def build_conversation_history(
    response_data: dict,
    user_content: GeminiContent,
    previous_history: list[dict] | None = None
) -> list[dict]:
    """Build conversation history for chat mode from response."""
    history = list(previous_history) if previous_history else []
    
    # Add user turn
    history.append(user_content.model_dump(exclude_none=True))
    
    # Add model response
    candidates = response_data.get("candidates", [])
    if candidates:
        model_content = candidates[0].get("content")
        if model_content:
            history.append(model_content)
    
    return history

# -------------------------------------------------------------------------
# Resolution Helpers
# -------------------------------------------------------------------------

def _aspect_to_float(aspect: str, ref_tensor: torch.Tensor | None) -> float:
    """Return W/H ratio."""
    if aspect and aspect != "auto" and ":" in aspect:
        a, b = aspect.split(":", 1)
        aw = int(a.strip()); ah = int(b.strip())
        if aw > 0 and ah > 0:
            return aw / ah

    # auto: use reference image aspect if available
    if ref_tensor is not None:
        t = ref_tensor[0] if ref_tensor.ndim == 4 else ref_tensor
        h, w = int(t.shape[0]), int(t.shape[1])
        if h > 0 and w > 0:
            return w / h

    # fallback 1:1
    return 1.0


def _snap(v: int, multiple: int = 8) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def _target_hw_from_area(image_size: str, aspect_ratio: str, ref_tensor: torch.Tensor | None) -> tuple[int, int]:
    """
    Interpret image_size as pixel AREA budget = (base_side^2),
    then pick H,W so that H*W ≈ area and W/H ≈ aspect.
    """
    base_side_map = {"0.5K": 512, "1K": 1024, "2K": 2048, "4K": 4096}
    base_side = base_side_map.get(image_size, 1024)
    area = base_side * base_side

    r = _aspect_to_float(aspect_ratio, ref_tensor)  # r = W/H

    # Solve: W = sqrt(area * r), H = sqrt(area / r)
    W = int(round(math.sqrt(area * r)))
    H = int(round(math.sqrt(area / r)))

    # Snap to multiples (keeps comfy/vae-friendly sizes)
    W = _snap(W, 8)
    H = _snap(H, 8)

    return H, W

def enforce_output_size(img_tensor, image_size, aspect_ratio, ref_tensor=None):
    if img_tensor is None:
        return img_tensor

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    target_h, target_w = _target_hw_from_area(image_size, aspect_ratio, ref_tensor)

    t = img_tensor[0]
    arr = (t.detach().cpu().numpy().clip(0, 1) * 255.0).astype("uint8")

    pil_img = Image.fromarray(arr, mode="RGB")
    pil_img = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)

    arr2 = (np.array(pil_img).astype("float32") / 255.0)
    return torch.from_numpy(arr2).unsqueeze(0)

# -------------------------------------------------------------------------
# API Request Helper
# -------------------------------------------------------------------------


GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# -------------------------------------------------------------------------
# Model lists & capability flags (keep aligned with Google docs)
# Source: https://ai.google.dev/gemini-api/docs/models
# -------------------------------------------------------------------------

# Image-capable (native image output) models supported by this node.
# NOTE: Some models may be preview/stable; adjust as needed.
GEMINI_IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",   # Nano Banana 2
    "gemini-3-pro-image-preview",       # Nano Banana Pro
    "gemini-2.5-flash-image",           # Nano Banana
]

# Text / multimodal models (text output) supported by the LLM node.
GEMINI_LLM_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

# Models that support Google Search grounding (aka googleSearch tool).
# See:
# - Models: https://ai.google.dev/gemini-api/docs/models
# - Google Search grounding: https://ai.google.dev/gemini-api/docs/google-search
GOOGLE_SEARCH_GROUNDING_SUPPORTED = {
    # Gemini 3
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",

    # Gemini 2.5 / 2.0
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",

    # Image models also list Search grounding as supported in docs, but our image node
    # currently doesn't attach tools; keep here for completeness.
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
}

# Image model capability flags
# gemini-3-pro-image-preview supports imageConfig.imageSize (1K/2K/4K).
# gemini-2.5-flash-image currently supports aspectRatio but not imageSize in imageConfig.
# See official docs: https://ai.google.dev/gemini-api/docs/image-generation
IMAGE_MODELS_SUPPORT_IMAGE_SIZE = {
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",   # supports 0.5K/1K/2K/4K
}

IMAGE_MODELS_SUPPORT_THINKING_LEVEL = {
    "gemini-3.1-flash-image-preview",   # thinkingLevel control only here
}

IMAGE_MODELS_SUPPORT_IMAGE_SEARCH_GROUNDING = {
    "gemini-3.1-flash-image-preview",   # image search grounding only here
}

GEMINI_MODEL = "gemini-3-pro-image-preview"  # default image model
GEMINI_LLM_MODEL = "gemini-3-pro-preview"  # default LLM model
GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)

# -------------------------------------------------------------------------
# ASPECTS & RESOLUTION OPTIONS (keep aligned with Gemini docs and node dropdowns)
# -------------------------------------------------------------------------

ASPECT_RATIOS_GEMINI3_1 = [
    "auto",

    # Square
    "1:1",

    # Portrait (common)
    "9:16", "10:16", "2:3", "3:4", "4:5", "5:7", "8:11", "9:19", "1:2", "3:5",

    # Landscape (common)
    "16:9", "16:10", "3:2", "4:3", "5:4", "7:5", "11:8", "19:9", "2:1", "5:3",

    # Ultrawide / banners
    "21:9", "32:9", "239:100",

    # Extreme banners (you already have some)
    "4:1", "1:4", "8:1", "1:8",
]

ASPECT_RATIOS_GEMINI3 = ["auto","1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"]

MODEL_ALLOWED_ASPECTS = {
    "gemini-3.1-flash-image-preview": set(ASPECT_RATIOS_GEMINI3_1),
    "gemini-3-pro-image-preview": set(ASPECT_RATIOS_GEMINI3),
    "gemini-2.5-flash-image": set(ASPECT_RATIOS_GEMINI3),
}

def _sanitize_aspect(model: str, aspect: str) -> str:
    allowed = MODEL_ALLOWED_ASPECTS.get(model)
    if not allowed:
        return aspect
    return aspect if aspect in allowed else "auto"

# -------------------------------------------------------------------------

async def call_gemini_api(
    api_key: str,
    contents: list[GeminiContent | dict],
    generation_config: GeminiImageGenerationConfig | None = None,
    system_instruction: GeminiSystemInstructionContent | None = None,
    model: str = GEMINI_MODEL,
    tools: list[dict] | None = None,
) -> dict:
    """Make a request to the Gemini API."""
    url = f"{GEMINI_API_BASE_URL}/{model}:generateContent?key={api_key}"
    
    # Build contents list - handle both Pydantic models and raw dicts
    serialized_contents = []
    for content in contents:
        if isinstance(content, dict):
            # Raw dict from conversation history - preserve as-is (includes thoughtSignature)
            serialized_contents.append(content)
        else:
            # Pydantic model - serialize it
            serialized_contents.append(content.model_dump(exclude_none=True))
    
    # Build payload manually to preserve raw dicts
    payload: dict = {"contents": serialized_contents}
    
    if tools:
        payload["tools"] = tools  # ✅ pass through as-is

    if generation_config:
        payload["generationConfig"] = generation_config.model_dump(exclude_none=True)

    if system_instruction:
        payload["systemInstruction"] = system_instruction.model_dump(exclude_none=True)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Gemini API error ({response.status}): {error_text}")
            
            return await response.json()


async def call_gemini_llm_api(
    api_key: str,
    contents: list[GeminiContent | dict],
    generation_config: GeminiGenerationConfig | None = None,
    system_instruction: GeminiSystemInstructionContent | None = None,
    tools: list[GeminiTool] | None = None,
    model: str = GEMINI_LLM_MODEL,
) -> dict:
    """Make a request to the Gemini LLM API."""
    url = f"{GEMINI_API_BASE_URL}/{model}:generateContent?key={api_key}"
    
    # Build contents list - handle both Pydantic models and raw dicts
    serialized_contents = []
    for content in contents:
        if isinstance(content, dict):
            serialized_contents.append(content)
        else:
            serialized_contents.append(content.model_dump(exclude_none=True))
    
    payload: dict = {"contents": serialized_contents}
    
    if generation_config:
        payload["generationConfig"] = generation_config.model_dump(exclude_none=True)
    
    if system_instruction:
        payload["systemInstruction"] = system_instruction.model_dump(exclude_none=True)
    
    if tools:
        payload["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Gemini LLM API error ({response.status}): {error_text}")
            
            return await response.json()


def extract_llm_text_from_response(response_data: dict) -> str:
    """Extract text from Gemini LLM API response."""
    texts = []
    candidates = response_data.get("candidates", [])
    
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)
    
    return "\n".join(texts)


def extract_grounding_metadata(response_data: dict) -> dict:
    """Extract grounding metadata (search results) from response."""
    metadata = {}
    candidates = response_data.get("candidates", [])
    
    if candidates:
        grounding = candidates[0].get("groundingMetadata")
        if grounding:
            metadata["groundingMetadata"] = grounding
            # Extract search queries if available
            search_queries = grounding.get("webSearchQueries", [])
            if search_queries:
                metadata["searchQueries"] = search_queries
            # Extract grounding chunks (sources)
            chunks = grounding.get("groundingChunks", [])
            if chunks:
                sources = []
                for chunk in chunks:
                    web = chunk.get("web", {})
                    if web:
                        sources.append({
                            "title": web.get("title", ""),
                            "uri": web.get("uri", ""),
                        })
                metadata["sources"] = sources
    
    return metadata


# -------------------------------------------------------------------------
# Main Node: Gemini 3 Pro Image with Chat Mode
# -------------------------------------------------------------------------


class GeminiImage3ProNode(IO.ComfyNode):
    """
    Unified node for Gemini 3 Pro Image generation with chat/refine mode support.
    
    - Generate images from text prompts
    - Edit images with optional input images
    - Continue conversations using metadata input/output for iterative refinement
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiImage3ProNode",
            display_name="Nano Banana Pro (Gemini 3 Pro Image)",
            category="api node/image/Gemini",
            description=(
                "Generate or edit images using Google Gemini 3 Pro Image model. "
                "Supports chat mode for iterative refinement via metadata input/output."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Text prompt describing the image to generate or the edits to apply. "
                        "Include any constraints, styles, or details the model should follow."
                    ),
                ),
                IO.Combo.Input(
                    "model",
                    options=GEMINI_IMAGE_MODELS,
                    default=GEMINI_MODEL,
                    tooltip="Gemini model to use for image generation/editing.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,  # INT32 max (2^31 - 1) required by Gemini API
                    control_after_generate=True,
                    tooltip=(
                        "When seed is fixed to a specific value, the model makes a best effort "
                        "to provide the same response for repeated requests. "
                        "Set to 0 for random seed. Max value: 2147483647."
                    ),
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=ASPECT_RATIOS_GEMINI3_1,  # Use the more extensive list for the dropdown; node will sanitize based on model
                    default="auto",
                    tooltip=(
                        "Output aspect ratio. 'auto' matches input image or defaults to 1:1. "
                        "Note: aspect_ratio is applied when supported by the selected model."
                    ),
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["0.5K", "1K", "2K", "4K"],
                    default="1K",
                    tooltip="Target output resolution. Note: 2K/4K (imageSize) is supported by gemini-3-pro-image-preview; for gemini-2.5-flash-image this setting is ignored.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    default="IMAGE+TEXT",
                    tooltip=(
                        "Choose 'IMAGE' for image-only output, or "
                        "'IMAGE+TEXT' to return both the generated image and a text response."
                    ),
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Controls randomness in generation. Higher values = more creative.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional reference image(s) for editing or style transfer. "
                        "To include multiple images, use the Batch Images node."
                    ),
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask for inpainting. White areas will be edited.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="System instructions that guide the model's behavior.",
                ),
                IO.Custom("GEMINI_CHAT_METADATA").Input(
                    "chat_metadata",
                    optional=True,
                    tooltip=(
                        "Optional conversation metadata from a previous generation. "
                        "Connect to enable chat/refine mode for iterative editing."
                    ),
                ),
                IO.Boolean.Input(
                    "enable_google_search",
                    default=False,
                    tooltip="Enable Google Search grounding (web).",
                ),
                IO.Combo.Input(
                    "search_mode",
                    options=["web", "web+image"],
                    default="web",
                    tooltip="For gemini-3.1-flash-image-preview you can also enable Image Search grounding.",
                ),
                IO.Combo.Input(
                    "thinking_level",
                    options=["default", "minimal", "High"],
                    default="default",
                    tooltip="Only applies to gemini-3.1-flash-image-preview. Default is minimal.",
                ),
                IO.Boolean.Input(
                    "include_thoughts",
                    default=False,
                    tooltip="Return thought parts in the response (we still filter thought images from outputs).",
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["default", "allow", "block"],  # optional; only send if not default
                    default="default",
                    tooltip="Optional safety control for generating people (only if supported by the backend).",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.String.Output(display_name="text"),
                IO.Custom("GEMINI_CHAT_METADATA").Output(display_name="chat_metadata"),
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        temperature: float,
        enable_google_search: bool = False,   
        search_mode: str = "web",             
        thinking_level: str = "default",     
        include_thoughts: bool = False,      
        person_generation: str = "default",
        images: Input.Image | None = None,
        mask: Input.Mask | None = None,
        system_prompt: str = "",
        chat_metadata: dict | None = None,
    ) -> IO.NodeOutput:
        
        # Get API key
        api_key = get_api_key()
        
        # Build parts for the user content
        parts: list[GeminiPart] = []
        
        # Add text prompt
        if prompt.strip():
            parts.append(GeminiPart(text=prompt))
        
        # Add input images
        if images is not None:
            num_images = images.shape[0] if images.ndim >= 4 else 1
            for i in range(min(num_images, 14)):  # Max 14 images
                if images.ndim >= 4:
                    img_tensor = images[i]
                else:
                    img_tensor = images
                
                img_b64 = tensor_to_base64(img_tensor, "image/png")
                parts.append(GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType="image/png",
                        data=img_b64,
                    )
                ))
        
        # Add mask if provided
        if mask is not None:
            # Convert mask to image format
            if mask.ndim == 2:
                mask_tensor = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
            elif mask.ndim == 3:
                mask_tensor = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
            else:
                mask_tensor = mask
            
            mask_b64 = tensor_to_base64(mask_tensor, "image/png")
            parts.append(GeminiPart(text="Mask for inpainting:"))
            parts.append(GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType="image/png",
                    data=mask_b64,
                )
            ))
        
        resolved_model = model if model in GEMINI_IMAGE_MODELS else GEMINI_MODEL
        # --- tools (grounding) ---
        tools = None
        if enable_google_search:
            if search_mode == "web+image" and resolved_model in IMAGE_MODELS_SUPPORT_IMAGE_SEARCH_GROUNDING:
                tools = [{
                    "google_search": {
                        "searchTypes": {
                            "webSearch": {},
                            "imageSearch": {},
                        }
                    }
                }]
            else:
                # web-only works broadly
                tools = [{"google_search": {}}]

        thinking_cfg = None
        if resolved_model in IMAGE_MODELS_SUPPORT_THINKING_LEVEL:
            if thinking_level != "default":
                thinking_cfg = GeminiThinkingConfig(
                    thinkingLevel=thinking_level,
                    includeThoughts=include_thoughts,
                )
            elif include_thoughts:
                # keep default level but request thoughts
                thinking_cfg = GeminiThinkingConfig(
                    includeThoughts=True
                )

        # Build user content
        user_content = GeminiContent(
            role="user",
            parts=parts,
        )
        
        # Build contents list (with conversation history if in chat mode)
        contents: list[GeminiContent | dict] = []
        previous_history: list[dict] | None = None
        
        if chat_metadata and isinstance(chat_metadata, dict):
            conversation_history = chat_metadata.get("conversation_history", [])
            if conversation_history:
                previous_history = conversation_history
                # Keep history as raw dicts to preserve thoughtSignature
                contents.extend(conversation_history)
        
        contents.append(user_content)
        
        # Build image config
        # NOTE: gemini-2.5-flash-image supports aspectRatio but does NOT accept imageSize.
        # Only gemini-3-pro-image-preview supports imageSize (1K/2K/4K) per docs.
        image_config = GeminiImageConfig()

        if aspect_ratio != "auto":
            image_config.aspectRatio = _sanitize_aspect(resolved_model, aspect_ratio)

        if resolved_model in IMAGE_MODELS_SUPPORT_IMAGE_SIZE:
            # Only 3.1 Flash supports 0.5K; Pro supports 1K/2K/4K.
            if resolution == "0.5K" and resolved_model != "gemini-3.1-flash-image-preview":
                print(f"[Gemini] Model '{resolved_model}' does not support 0.5K; using 1K.")
                image_config.imageSize = "1K"
            else:
                image_config.imageSize = resolution
        else:
            if resolution != "1K":
                print(f"[Gemini] Model '{resolved_model}' does not support imageSize; ignoring resolution={resolution}.")

        # Optional: personGeneration (send only if set)
        # NOTE: not shown on the Nano Banana page, but exists in some API schemas; guard to avoid INVALID_ARGUMENT
        if person_generation != "default":
            # Pydantic model_config extra isn't set on GeminiImageConfig; easiest is to inject via dict:
            image_config_dict = image_config.model_dump(exclude_none=True)
            image_config_dict["personGeneration"] = person_generation.capitalize()  # "Allow"/"Block" if required
        else:
            image_config_dict = image_config.model_dump(exclude_none=True)
        
        # Build generation config
        modalities = ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]
        # Clamp seed to INT32 max (Gemini API requirement)
        safe_seed = min(seed, 2147483647) if seed != 0 else None
        generation_config = GeminiImageGenerationConfig(
            responseModalities=modalities,
            imageConfig=GeminiImageConfig(**image_config_dict) if image_config_dict else None,
            thinkingConfig=thinking_cfg,
            temperature=temperature,
            seed=safe_seed,
        )
        # Build system instruction
        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(
                parts=[GeminiTextPart(text=system_prompt)],
                role=None,
            )
        
        # Make API request
        try:
            response_data = await call_gemini_api(
                api_key=api_key,
                contents=contents,
                generation_config=generation_config,
                system_instruction=gemini_system_prompt,
                model=resolved_model,
                tools=tools,  
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
        
        # Extract images (filter out thought images)
        image_tensors = extract_images_from_response(response_data)
        if image_tensors:
            output_image = torch.cat(image_tensors, dim=0) if len(image_tensors) > 1 else image_tensors[0]
            ref = images if images is not None else None
            output_image = enforce_output_size(output_image, resolution, aspect_ratio, ref_tensor=ref)
        else:
            # Return placeholder if no images generated
            output_image = torch.zeros((1, 1024, 1024, 3))
        
        # Extract text
        output_text = extract_text_from_response(response_data)
        
        # Build output metadata for chat mode
        conversation_history = build_conversation_history(
            response_data=response_data,
            user_content=user_content,
            previous_history=previous_history,
        )
        
        output_metadata = {
            "conversation_history": conversation_history,
            "model": (model if model in GEMINI_IMAGE_MODELS else GEMINI_MODEL),
            "seed": safe_seed if safe_seed else 0,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "last_prompt": prompt,
            "last_response_text": output_text,
        }

        grounding = response_data.get("candidates", [{}])[0].get("groundingMetadata")
        output_metadata["grounding_metadata"] = grounding
        output_metadata["enable_google_search"] = enable_google_search
        output_metadata["search_mode"] = search_mode
        output_metadata["thinking_level"] = thinking_level
        
        return IO.NodeOutput(output_image, output_text, output_metadata)


# -------------------------------------------------------------------------
# Node 2: Gemini 3 Pro LLM with Google Search
# -------------------------------------------------------------------------


class Gemini3ProLLMNode(IO.ComfyNode):
    """
    Gemini 3 Pro LLM node with optional Google Search grounding.
    
    - Generate text responses from prompts
    - Optional Google Search for up-to-date information
    - Multi-turn conversation support via metadata
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Gemini3ProLLMNode",
            display_name="Gemini 3 Pro LLM 🔍",
            category="api node/text/Gemini",
            description=(
                "Generate text responses using Gemini 3 Pro LLM. "
                "Supports Google Search grounding for up-to-date information. "
                "Chat mode available via metadata input/output."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the LLM.",
                ),
                IO.Combo.Input(
                    "model",
                    options=GEMINI_LLM_MODELS,
                    default=GEMINI_LLM_MODEL,
                    tooltip="Gemini model to use for text generation.",
                ),
                IO.Boolean.Input(
                    "enable_google_search",
                    default=False,
                    tooltip="Enable Google Search grounding for up-to-date information.",
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Controls randomness. Lower = more focused, higher = more creative.",
                ),
                IO.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Nucleus sampling threshold.",
                ),
                IO.Int.Input(
                    "top_k",
                    default=40,
                    min=1,
                    max=100,
                    optional=True,
                    tooltip="Top-k sampling.",
                ),
                IO.Int.Input(
                    "max_tokens",
                    default=8192,
                    min=1,
                    max=65536,
                    optional=True,
                    tooltip="Maximum output tokens.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for reproducibility. Set to 0 for random.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) for multimodal input.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="System instructions for the model.",
                ),
                IO.Custom("GEMINI_CHAT_METADATA").Input(
                    "chat_metadata",
                    optional=True,
                    tooltip="Conversation metadata for multi-turn chat.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="text"),
                IO.String.Output(display_name="operation_log"),
                IO.Custom("GEMINI_CHAT_METADATA").Output(display_name="chat_metadata"),
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        enable_google_search: bool,
        temperature: float,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 8192,
        seed: int = 0,
        images: Input.Image | None = None,
        system_prompt: str = "",
        chat_metadata: dict | None = None,
    ) -> IO.NodeOutput:
        
        # Get API key
        api_key = get_api_key()
        
        # Operation log
        log_lines: list[str] = []
        resolved_model = model if model in GEMINI_LLM_MODELS else GEMINI_LLM_MODEL
        log_lines.append(f"Model: {resolved_model}")
        log_lines.append(f"Google Search: {'enabled' if enable_google_search else 'disabled'}")
        log_lines.append(f"Search grounding supported: {'yes' if resolved_model in GOOGLE_SEARCH_GROUNDING_SUPPORTED else 'no'}")
        log_lines.append(f"Temperature: {temperature}")
        log_lines.append(f"Seed: {seed if seed != 0 else 'random'}")
        
        # Build parts for the user content
        parts: list[GeminiPart] = []
        
        # Add text prompt
        if prompt.strip():
            parts.append(GeminiPart(text=prompt))
            log_lines.append(f"Prompt length: {len(prompt)} chars")
        
        # Add input images if provided
        if images is not None:
            num_images = images.shape[0] if images.ndim >= 4 else 1
            images_added = min(num_images, 14)
            for i in range(images_added):
                if images.ndim >= 4:
                    img_tensor = images[i]
                else:
                    img_tensor = images
                
                img_b64 = tensor_to_base64(img_tensor, "image/png")
                parts.append(GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType="image/png",
                        data=img_b64,
                    )
                ))
            log_lines.append(f"Images: {images_added}")
        
        # Build user content
        user_content = GeminiContent(
            role="user",
            parts=parts,
        )
        
        # Build contents list (with conversation history if in chat mode)
        contents: list[GeminiContent | dict] = []
        previous_history: list[dict] | None = None
        
        if chat_metadata and isinstance(chat_metadata, dict):
            conversation_history = chat_metadata.get("conversation_history", [])
            if conversation_history:
                previous_history = conversation_history
                contents.extend(conversation_history)
                log_lines.append(f"Chat history: {len(conversation_history)} turns")
        
        contents.append(user_content)
        
        # Build generation config
        safe_seed = min(seed, 2147483647) if seed != 0 else None
        generation_config = GeminiGenerationConfig(
            temperature=temperature,
            topP=top_p,
            topK=top_k,
            maxOutputTokens=max_tokens,
            seed=safe_seed,
        )
        
        # Build system instruction
        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(
                parts=[GeminiTextPart(text=system_prompt)],
                role=None,
            )
            log_lines.append(f"System prompt: {len(system_prompt)} chars")
        
        # Build tools (Google Search grounding) only for models that support it
        tools = None
        if enable_google_search and resolved_model in GOOGLE_SEARCH_GROUNDING_SUPPORTED:
            tools = [GeminiTool(googleSearch=GeminiGoogleSearch())]
        elif enable_google_search and resolved_model not in GOOGLE_SEARCH_GROUNDING_SUPPORTED:
            log_lines.append("Google Search: requested but NOT supported by this model (ignored)")
        
        # Make API request
        try:
            response_data = await call_gemini_llm_api(
                api_key=api_key,
                contents=contents,
                generation_config=generation_config,
                system_instruction=gemini_system_prompt,
                tools=tools,
                model=resolved_model,
            )
            log_lines.append("API call: success")
        except Exception as e:
            log_lines.append(f"API call: failed - {e}")
            raise RuntimeError(f"Gemini LLM API call failed: {e}")
        
        # Extract text
        output_text = extract_llm_text_from_response(response_data)
        log_lines.append(f"Response length: {len(output_text)} chars")
        
        # Extract grounding metadata (search sources)
        grounding = extract_grounding_metadata(response_data)
        if grounding.get("sources"):
            log_lines.append(f"Search sources: {len(grounding['sources'])}")
            for src in grounding["sources"]:
                log_lines.append(f"  • {src.get('title', 'Unknown')}: {src.get('uri', '')}")
        if grounding.get("searchQueries"):
            log_lines.append(f"Search queries: {grounding['searchQueries']}")
        
        # Build conversation history
        conversation_history = build_conversation_history(
            response_data=response_data,
            user_content=user_content,
            previous_history=previous_history,
        )
        
        output_metadata = {
            "conversation_history": conversation_history,
            "model": resolved_model,
            "seed": safe_seed if safe_seed else 0,
            "google_search_enabled": enable_google_search,
            "last_prompt": prompt,
            "last_response_text": output_text,
            "grounding_metadata": grounding,
        }
        
        operation_log = "\n".join(log_lines)
        
        return IO.NodeOutput(output_text, operation_log, output_metadata)


# -------------------------------------------------------------------------
# Extension Registration
# -------------------------------------------------------------------------


class GeminiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GeminiImage3ProNode,
            Gemini3ProLLMNode,
        ]


async def comfy_entrypoint() -> GeminiExtension:
    return GeminiExtension()


# -------------------------------------------------------------------------
# Node Mappings (for traditional ComfyUI registration)
# -------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiImage3ProNode": GeminiImage3ProNode,
    "Gemini3ProLLMNode": Gemini3ProLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImage3ProNode": "Nano Banana Pro (Gemini 3 Pro Image) 🎨",
    "Gemini3ProLLMNode": "Gemini 3 Pro LLM 🔍",
}