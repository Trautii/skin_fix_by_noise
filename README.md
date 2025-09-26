# skin_fix_by_noise
Real Skin Post – ComfyUI Custom Node



Adds realistic microtexture/pores to skin as a post-process.



\## Nodes

\- \*\*Real Skin Post (Simple)\*\* – two knobs: `strength` (Soft Light) and `multiply`.

\- \*\*Real Skin Post (Advanced)\*\* – extra controls for pore scale/strength, highlight breakup, and low-freq color variance.



\## Install

Clone this repo into `ComfyUI/custom\_nodes/` and restart ComfyUI, no other requirements needed.



\## Usage

Use before saving your final image. Start with:

\- `strength = 0.10`

\- `multiply = 0.04`



Optionally feed a mask. Otherwise the node auto-detects skin (YCrCb+HSV) and feathers it.

