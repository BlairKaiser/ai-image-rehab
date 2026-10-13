"""Redraw a selected spot of a picture with a neural network: what is selected is taken out and
drawn anew from what is around it.

    python -I app/neural_fill.py IN.png OUT.png --brush "x,y x,y ..." [--size 12] [--zone "x,y ..."] [--gpu]

The network is LaMa trained on anime and manga (the one used to wipe the lettering off manga pages
and put the drawing back under it): models/lama/anime-manga-big-lama.pt, 196 MB, from the
release "AnimeMangaInpainting" of github.com/Sanster/models. The file is TorchScript: it holds the
network's own code besides its weights, so it cannot be read as weights only. It is read with
torch.jit.load, which runs no Python from the file. Run this script with -I.

retouch.py paints a spot over with ONE fill by rule. This draws the spot anew: it carries lines and
fills on through the spot, each where it belongs. It does not know what was meant either: it shows
what is likely, so every result is looked at.

Only the selected pixels change, with a soft edge of two pixels. The network sees a square piece of
the picture around the selection, not the whole picture, and sees it enlarged: it fills better so.
Its answer is then drawn with the picture's own colours (retouch.snap): a network paints a plain
fill a little off and with grain, the picture's own fill is exact.
"""
import argparse
import os
import site
import sys

sys.path.append(site.getusersitepackages())      # with -I the user's own packages are not on the path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

MODEL = os.path.join(os.path.dirname(HERE), "models", "lama", "anime-manga-big-lama.pt")
AROUND = 160      # the network is shown at least this many px of the picture on every side of the selection
ENLARGE = 2.0     # and is shown the piece this many times larger than it is. Measured on the bench of 64 cases
                  # (mean miss in levels, with the picture's own colours put back): as it is 4.1, twice larger 3.6,
                  # three times 3.6, four times 3.7. Its lines come out sharper too.
LARGEST = 1536    # and no piece larger than this on a side: a larger one is shown reduced
LOADED = {}


def available():
    return os.path.exists(MODEL)


def network(device):
    import torch
    if device not in LOADED:
        LOADED[device] = torch.jit.load(MODEL, map_location=device).eval()
    return LOADED[device]


def inpaint(pixels, cover, device="cpu", grow=2, enlarge=None):
    """Draw anew the pixels where cover is at least a half.

    pixels   the picture, RGB;
    cover    the selection, 0 to 1;
    grow     the selection is widened by this many px before it is taken out: the soft edge of the
             thing selected lies there, and left in sight it is drawn back.
    Returns the new picture (float) and the number of pixels changed.
    """
    import torch
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
    around = max(AROUND, int(0.75 * max(rows.max() - rows.min(), cols.max() - cols.min())))
    top, bottom = max(rows.min() - around, 0), min(rows.max() + around + 1, height)
    left, right = max(cols.min() - around, 0), min(cols.max() + around + 1, width)
    part = pixels[top:bottom, left:right]
    gone = hole[top:bottom, left:right]
    tall, wide = gone.shape
    # a very large piece is shown to the network reduced, and only its answer for the hole is enlarged back
    enlarge = ENLARGE if enlarge is None else enlarge
    shrink = 1.0 / float(enlarge)                                             # show the piece larger than it is,
    shrink = max(shrink, max(tall, wide) / float(LARGEST))                    # but never a piece too large to compute
    if shrink != 1.0:
        size = (int(round(wide / shrink)), int(round(tall / shrink)))
        seen = np.asarray(Image.fromarray(np.clip(part, 0, 255).astype(np.uint8)).resize(size, Image.LANCZOS), np.float32)
        seen_gone = np.asarray(Image.fromarray((gone * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) > 40
    else:
        seen, seen_gone = part, gone
    # the network wants sides that divide by eight
    extra_rows, extra_cols = (-seen.shape[0]) % 8, (-seen.shape[1]) % 8
    image = np.pad(seen, ((0, extra_rows), (0, extra_cols), (0, 0)), mode="symmetric")
    mask = np.pad(seen_gone, ((0, extra_rows), (0, extra_cols)), mode="symmetric")
    with torch.no_grad():
        picture = torch.from_numpy(image / 255.0).permute(2, 0, 1)[None].to(device)
        holes = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
        answer = network(device)(picture, holes)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    answer = answer[:seen.shape[0], :seen.shape[1]]
    if shrink != 1.0:
        answer = np.asarray(Image.fromarray(np.clip(answer, 0, 255).astype(np.uint8)).resize((wide, tall), Image.BICUBIC if shrink > 1.0 else Image.LANCZOS), np.float32)
    weight = np.clip(ndimage.gaussian_filter(gone.astype(np.float32), 1.0) * 1.6, 0.0, 1.0)
    weight[gone] = np.maximum(weight[gone], 0.85)
    weight = ndimage.gaussian_filter(weight, 0.6)[..., None]
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
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--show", help="save the selection here, as a picture, to look at it")
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
    if options.show:
        shown = retouch.to_bytes(pixels)
        shown[cover] = shown[cover] // 2 + np.array([127, 0, 110], np.uint8)
        Image.fromarray(shown).save(options.show)
    scale = retouch.scale_of(pixels.shape)
    guess = inpaint(pixels, cover, "cuda" if options.gpu else "cpu", grow=max(2, int(round(2 * scale))), enlarge=max(1.0, ENLARGE / scale))[0]
    out, count = retouch.snap(pixels, cover, guess, scale)      # the plain fills of the answer, in the picture's own colours
    result = Image.fromarray(retouch.to_bytes(out))
    if alpha is not None:
        result.putalpha(alpha)
    result.save(options.target)
    print("drawn anew: %d px" % count)


if __name__ == "__main__":
    main()
