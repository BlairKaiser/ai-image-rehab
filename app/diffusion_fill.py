"""Draw a selected spot anew with a diffusion model: an SDXL anime checkpoint that is already on
this computer (Illustrious and its kin). The way the communities fix a small place: "inpaint only
masked", also called crop and stitch.

    PYTHON -I app/diffusion_fill.py IN.png OUT.png --brush "x,y x,y ..." [--size 12] [--zone "x,y ..."]
           [--strength 0.6] [--steps 22] [--prompt "..."] [--seed 0] [--checkpoint FILE] [--after-lama]

PYTHON must be an interpreter that has torch with CUDA, diffusers and transformers, for instance the
one of a Forge installation. Nothing is installed and nothing is written into that installation; its
checkpoint is only read. FILL_CHECKPOINT and FILL_SETUP name the model (see the README).

How a spot is drawn:

1. a piece of the picture around the selection is cut out, with room around it for the model to
   see what it is drawing in;
2. the piece is brought to the size the model draws at (its longer side to 1024 px), so a small
   spot is drawn large and comes back sharp;
3. the model draws the selected part anew, keeping `strength` of freedom (0.3 tidies, 0.6 redraws,
   1.0 forgets what was there). With --after-lama the mark is first taken out by LaMa and the
   model only draws that guess clean;
4. the piece is brought back to its size and only the selected pixels are put into the picture.
   A diffusion model shifts colours a little on the way through its coder; the shift is taken off
   so that the edge of the spot meets the picture exactly.

The rest of the picture never passes through the model: it stays pixel for pixel.
"""
import argparse
import os
import site
import sys

sys.path.append(site.getusersitepackages())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

# nothing here points at a place of its own: the person names the model (see the README)
CHECKPOINT = os.environ.get("FILL_CHECKPOINT", "")      # an SDXL anime checkpoint, one .safetensors file
SETUP = os.environ.get("FILL_SETUP", "")                # a folder with the SDXL layout files (model_index.json and the rest)
PROMPT = "masterpiece, best quality, amazing quality, very aesthetic, anime coloring, clean lineart, flat color"
AVOID = "worst quality, bad quality, lowres, sketch, jpeg artifacts, blurry, noise, text, signature, watermark, extra lines"
DRAWS_AT = 1024
# a guide trained for exactly this job on the Illustrious family: it is shown the piece with the spot blacked
# out and steers the model to draw the spot as a part of what is around it, even with nothing left of what was there
GUIDE = os.path.join(os.path.dirname(HERE), "models", "noobai-inpaint", "NoobAI_Inpainting_ControlNet.safetensors")
LOADED = {}


def available():
    return bool(CHECKPOINT) and os.path.exists(CHECKPOINT) and os.path.isdir(SETUP)


def pipeline(checkpoint=None):
    """The model, read once. Its parts go to the video card one at a time, so it fits beside other
    programs that hold a part of the card's memory."""
    checkpoint = checkpoint or CHECKPOINT
    if checkpoint not in LOADED:
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(checkpoint, config=SETUP, torch_dtype=torch.float16, local_files_only=True)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        # with the card mostly free the model's parts go to it whole, one at a time (fast); when other programs hold
        # a part of the card they go piece by piece (slower, but nothing spills into the main memory, which is far slower)
        if torch.cuda.mem_get_info()[0] > 7.0e9:
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        LOADED[checkpoint] = pipe
    return LOADED[checkpoint]


def guided_pipeline(checkpoint=None):
    """The same model with the inpainting guide beside it."""
    checkpoint = checkpoint or CHECKPOINT
    key = ("guided", checkpoint)
    if key not in LOADED:
        import torch
        from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetInpaintPipeline
        guide = ControlNetModel.from_single_file(GUIDE, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(checkpoint, controlnet=guide, config=SETUP, torch_dtype=torch.float16,
                                                                           local_files_only=True)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        # the model and its guide together do not fit a card of 8 GB: sent whole, they spill into the main memory and
        # one spot takes many minutes. Piece by piece it is a minute a spot and 4 GB of the card.
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        LOADED[key] = pipe
    return LOADED[key]


def inpaint(pixels, cover, start=None, strength=0.6, steps=22, prompt=None, seed=0, grow=2, checkpoint=None, guidance=5.0, guide=False,
            draws_at=None):
    """Draw anew the pixels where cover is at least a half.

    start    a picture to draw from instead of the picture itself (the same size): a first guess of
             another filler, which the model then only draws clean.
    Returns the new picture (float) and the number of pixels changed.
    """
    import torch
    import retouch
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    cover = np.asarray(cover, np.float32)
    if cover.max() > 1.0:
        cover = cover / 255.0
    hole = cover >= 0.5
    if not hole.any():
        return pixels, 0
    hole = ndimage.binary_dilation(hole, iterations=max(int(grow), 0)) if grow else hole
    height, width = hole.shape
    rows, cols = np.nonzero(hole)
    tall, wide = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    side = int(max(max(tall, wide) * 1.9, 192))                         # room around the spot, and a square piece
    middle_row, middle_col = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2
    top = int(np.clip(middle_row - side // 2, 0, max(height - side, 0)))
    left = int(np.clip(middle_col - side // 2, 0, max(width - side, 0)))
    bottom, right = min(top + side, height), min(left + side, width)
    part = pixels[top:bottom, left:right]
    first = (np.asarray(start)[..., :3].astype(np.float32) if start is not None else pixels)[top:bottom, left:right]
    gone = hole[top:bottom, left:right]
    by = (draws_at or DRAWS_AT) / float(max(part.shape[0], part.shape[1]))
    size = (max(int(round(part.shape[1] * by / 8)) * 8, 64), max(int(round(part.shape[0] * by / 8)) * 8, 64))
    way = Image.LANCZOS if by < 1 else Image.BICUBIC
    image = Image.fromarray(np.clip(first, 0, 255).astype(np.uint8)).resize(size, way)
    mask = Image.fromarray((gone * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    asked = dict(prompt=prompt or PROMPT, negative_prompt=AVOID, image=image, mask_image=mask, width=size[0], height=size[1],
                 strength=float(strength), num_inference_steps=int(steps), guidance_scale=float(guidance),
                 generator=torch.Generator("cpu").manual_seed(int(seed)))
    if guide:
        blacked = np.asarray(image).copy()
        blacked[np.asarray(mask) > 127] = 0
        drawn = guided_pipeline(checkpoint)(control_image=Image.fromarray(blacked), controlnet_conditioning_scale=1.0, **asked).images[0]
    else:
        drawn = pipeline(checkpoint)(**asked).images[0]
    answer = np.asarray(drawn.resize((part.shape[1], part.shape[0]), Image.LANCZOS if by > 1 else Image.BICUBIC), np.float32)
    # the colour shift of the model's coder: what the answer misses at the rim of the spot is carried smoothly across it
    shift = retouch.repaint(part - answer, gone, ~gone)
    answer = answer + np.where(gone[..., None], shift, 0.0)
    weight = np.clip(ndimage.gaussian_filter(gone.astype(np.float32), 1.0) * 1.6, 0.0, 1.0)
    weight[gone] = np.maximum(weight[gone], 0.85)
    weight = np.where(ndimage.binary_dilation(gone, iterations=2), ndimage.gaussian_filter(weight, 0.6), 0.0)[..., None]
    result = part + (answer - part) * weight
    out = pixels.copy()
    out[top:bottom, left:right] = result
    return out, int((np.abs(result - part).max(axis=2) > 0.5).sum())


def main():
    import retouch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--brush", action="append", default=[])
    parser.add_argument("--size", type=float, default=12.0)
    parser.add_argument("--zone", action="append", default=[])
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=22)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--after-lama", action="store_true")
    options = parser.parse_args()

    def points(text):
        return [tuple(float(number) for number in corner.split(",")) for corner in text.split()]

    picture = Image.open(options.source)
    alpha = picture.getchannel("A") if picture.mode in ("RGBA", "LA") else None
    pixels = np.asarray(picture.convert("RGB")).astype(np.float32)
    cover = np.zeros(pixels.shape[:2], bool)
    for zone in options.zone:
        cover |= retouch.polygon(picture.size, points(zone))
    if options.brush:
        cover |= retouch.brushed(picture.size, [points(stroke) for stroke in options.brush], options.size)
    if not cover.any():
        parser.error("nothing is selected: give --brush or --zone")
    start = None
    if options.after_lama:
        import neural_fill
        start = neural_fill.inpaint(pixels, cover, "cpu")[0]
    out, count = inpaint(pixels, cover, start, options.strength, options.steps, options.prompt, options.seed, checkpoint=options.checkpoint)
    result = Image.fromarray(retouch.to_bytes(out))
    if alpha is not None:
        result.putalpha(alpha)
    result.save(options.target)
    print("drawn anew: %d px" % count)


if __name__ == "__main__":
    main()
