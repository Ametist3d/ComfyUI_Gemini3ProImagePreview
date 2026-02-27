
# 🍌 NanoBanana2 – Gemini Image Node for ComfyUI

NanoBanana2 is a custom ComfyUI node for Google Gemini image generation models.

Supported models:

- gemini-3.1-flash-image-preview (Nano Banana 2)
- gemini-3-pro-image-preview
- gemini-2.5-flash-image

Features:

- Text → Image
- Image → Image (edit / refine)
- Multi-turn refinement support
- Resolution control (0.5K / 1K / 2K / 4K)
- Extended aspect ratios (Nano Banana 2)
- Optional Google Search grounding
- Deterministic seed control
- Local resolution enforcement (pixel-budget based resizing)

------------------------------------------------------------

# Installation

1. Install into ComfyUI

Clone or copy this repository into:

ComfyUI/custom_nodes/NanoBanana2

Restart ComfyUI after installation.

------------------------------------------------------------

2. Install Requirements

Activate your ComfyUI environment and run:
```
pip install -r requirements.txt
```
Minimum dependencies:

aiohttp
pillow
torch
numpy

------------------------------------------------------------

# API Setup

NanoBanana2 requires a Google Gemini API key.

1. Create API Key

Visit:
https://ai.google.dev/

Create a Gemini API key.

2. in .env file change GEMINI_API_KEY=YOUR_API_KEY to your actual API-key.

OR 

3. Set Environment Variable

Windows (PowerShell):
```
setx GEMINI_API_KEY "YOUR_API_KEY"
```
Restart terminal and ComfyUI after setting it.

macOS / Linux:
```
export GEMINI_API_KEY="YOUR_API_KEY"
```
To verify:
```
echo $GEMINI_API_KEY
```
------------------------------------------------------------

# Supported Models

Model                               | 0.5K | Extended Aspects | Thinking Config | Image Search
-----------------------------------------------------------------------------------------------
gemini-3.1-flash-image-preview      | Yes  | Yes              | Yes             | Yes
gemini-3-pro-image-preview          | No   | Limited          | No              | Web only
gemini-2.5-flash-image              | No   | Limited          | No              | Web only

------------------------------------------------------------

# Node Settings

Model
Select Gemini image model. Nano Banana 2 exposes the most features.

Aspect Ratio
Includes auto, standard (1:1, 16:9, 9:16, 3:2, etc.), and extended ratios on Nano Banana 2.
Unsupported ratios are sanitized automatically.

Resolution
Resolution is interpreted as pixel area budget:

0.5K → ~512 × 512 area (~262K pixels)
1K   → ~1024 × 1024 area
2K   → ~2048 × 2048 area
4K   → ~4096 × 4096 area

For non-square ratios:
width × height ≈ base_size²

Example:
0.5K + 16:9 → approximately 688 × 384

The node enforces resolution locally to guarantee correct output size even if Gemini returns original input dimensions during edits.

Seed
-1 → random
Any positive integer → deterministic output

Temperature
0.2 – 0.6 → Stable
0.7 – 1.2 → More creative variation

Google Search
Optional grounding for improved realism or current references.
Nano Banana 2 additionally supports image search grounding.

Thinking Level (Nano Banana 2 only)
minimal
High

Higher thinking levels increase reasoning effort before generation.

Image Input (Optional)
If an image is connected:
- Performs edit/refinement
- Resolution is enforced locally
- Multi-turn refinement is preserved

------------------------------------------------------------

# Multi-Turn Refinement

The node preserves chat metadata to support:
- Iterative refinement
- Context-aware editing
- Image evolution workflows

------------------------------------------------------------

# Notes

- Unsupported settings are automatically sanitized.
- Extreme aspect ratios may reduce composition quality.
- Output resolution is always enforced locally to match selected pixel budget.

------------------------------------------------------------

# License

Add your preferred license here.
