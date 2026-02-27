# __init__.py

from .gemini_3_pro_image_preview import (
    GeminiImage3ProNode,
    Gemini3ProLLMNode,
)


NODE_CLASS_MAPPINGS = {
    "GeminiImage3ProNode": GeminiImage3ProNode,
    "Gemini3ProLLMNode": Gemini3ProLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImage3ProNode": "Gemini3-pro-image-preview",
    "Gemini3ProLLMNode": "Gemini-3-pro-llm",
}
