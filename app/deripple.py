"""Take the ripple (the pixel trace an image generator leaves) out of a picture without redrawing it.

    python app/deripple.py OUT_DIR PIC.png [PIC2.png ...] [--diff] [--painted]

What it removes (ripple_check.py says what the ripple is and measures it):

* the echo of every line: the strip one or two pixels wide beside it, lighter than the fill next to
  a dark line and darker next to a light shape, and the weaker strip of the opposite sign after it;
* the blotches inside a fill (the generator's tiles, about 8 px across) and the grain.

What it must not touch, and how that is kept:

* the lines themselves and their smooth edges. In the edge zone a pixel is pulled back to the fill
  only when it deviates the way an echo does: a little too light beside a dark line, a little too
  dark beside a light shape, by a small share of that line's contrast (FIRST, then fading out), and
  only right beside a strong line. A pixel that is darker than the fill beside a dark line is the
  soft edge of the line and stays;
* thin lines, lashes, freckles, pale small marks: they are edges or marks, not fills, so they keep
  their own pixels;
* shadows, highlights, gloss, a blush, a painted gradient. Inside a region the picture is compared
  with its own evened-out copy: what differs by less than DIRT levels is dirt and goes, what differs
  by more than twice that is drawing and stays as it was (only the grain is taken off it).

How it works: the picture is split into calm regions (no edge inside) and the edge zone between
them. The colour of a region is evened out from its own pixels only, counting mostly those that lie
away from the strong lines, so the strip beside a line takes the colour of the fill behind it.

What it does NOT do: it does not judge the drawing. A shadow that stands off its line with a strip
of light between (the eye reads a second contour), a stray tick, a shadow where none can be: these
are drawn, and retouch.py paints them out at a spot marked by hand. Closing such strips by rule was
tried twice and dropped: it either missed them or touched soft folds.

--painted  for a picture with soft shading everywhere: the colour is evened out over
           2 px instead of 5 and only the 3 px beside a line are distrusted instead of 5.5;
--reach, --guard  the same two numbers by hand;
--diff     also saves <name>-removed.png: what was taken away, eight times stronger, on grey. It
           must show only thin outlines along the lines and noise. A filled shape in it is damage.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

CALM = 5.0        # steeper than this (levels per px) is an edge
STRONG = 20.0     # an edge that throws an echo worth guarding against
MARK = 6.0        # a small mark this much darker or lighter than its fill is kept as drawn
FIRST = 0.14      # the first strip of an echo: at most this share of the edge's contrast
SECOND = 0.035    # the second, weaker strip
NEAR = 3          # an echo sits within this many px of the fill it belongs to
SMALL = 12        # a calm patch smaller than this (px) is not a fill
ALIKE = 7.0       # grain: neighbours within this many levels of a pixel are the same paint
DIRT = 5.0        # inside a fill a deviation below this is dirt, above twice this it is drawing


def regions(pixels, guard):
    soft = ndimage.gaussian_filter(pixels, (0.7, 0.7, 0))
    slope = np.zeros(pixels.shape[:2], np.float32)
    for channel in range(3):
        down, across = np.gradient(soft[..., channel])
        slope = np.maximum(slope, np.hypot(down, across))
    calm = ndimage.maximum_filter(slope, 3) < CALM
    distance = ndimage.distance_transform_edt(slope <= STRONG).astype(np.float32)
    # a faint small mark away from the lines (a pale freckle, a soft dot) is a drawing too: it is
    # weaker than an edge, so it would be smoothed away with the blotches. Echoes do not reach here.
    grey = soft @ np.array([0.299, 0.587, 0.114], np.float32)
    mark = np.maximum(ndimage.grey_closing(grey, size=(5, 5)) - grey, grey - ndimage.grey_opening(grey, size=(5, 5)))
    calm &= ~ndimage.binary_dilation((mark >= MARK) & (distance > guard))
    labels, count = ndimage.label(calm)
    if count:
        area = ndimage.sum(calm, labels, np.arange(1, count + 1))
        tiny = np.concatenate(([False], area < SMALL))
        calm &= ~tiny[labels]
        labels[~calm] = 0
    # beside a strong line a pixel carries the echo: it must take its colour, not give it
    trust = np.clip((distance - guard * 0.45) / (guard * 0.55), 0.004, 1.0).astype(np.float32)
    trust[~calm] = 0.0
    return calm, labels, trust, slope


def rebuild(pixels, labels, trust, reach):
    """The colour of every calm pixel from the pixels of its own region, the trusted ones first."""
    height, width = labels.shape
    edge = np.pad(labels, 1)
    moves = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    same = [((edge[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] == labels) & (labels > 0)).astype(np.float32)
            for dy, dx in moves]
    value = pixels.copy()
    passes = max(1, int(round(reach * reach * 1.5)))          # a 3x3 pass spreads by sqrt(2/3) px
    for _ in range(passes):
        weighted = value * trust[..., None]
        total = weighted.copy()
        weight = trust.copy()
        wide_value = np.pad(weighted, ((1, 1), (1, 1), (0, 0)))
        wide_weight = np.pad(trust, 1)
        for (dy, dx), mask in zip(moves, same):
            total += wide_value[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] * mask[..., None]
            weight += wide_weight[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] * mask
        np.divide(total, weight[..., None], out=value, where=weight[..., None] > 0)
    return value


def degrain(pixels, radius=3, passes=2):
    """Every pixel becomes the mean of the neighbours that are the same paint. Shapes keep their edges."""
    height, width = pixels.shape[:2]
    work = pixels.copy()
    for _ in range(passes):
        total = np.zeros_like(work)
        count = np.zeros((height, width), np.float32)
        wide = np.pad(work, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                near = wide[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
                alike = (np.abs(near - work).max(axis=2) < ALIKE).astype(np.float32)
                total += near * alike[..., None]
                count += alike
        work = total / count[..., None]
    return work


def settle_edges(pixels, values, calm, slope):
    """Beside a fill, pull back only what an echo looks like: a little too light next to a dark line,
    a little too dark next to a light shape. Everything else in the edge zone stays as drawn."""
    distance, place = ndimage.distance_transform_edt(~calm, return_indices=True)
    window = 2 * NEAR + 3
    beside = ~calm & (distance <= NEAR) & (ndimage.maximum_filter(slope, window) > STRONG * 1.5)
    own = values[place[0], place[1]]                     # the colour of the nearest fill
    away = values - own
    weight = np.ones(pixels.shape[:2], np.float32)
    for channel in range(3):
        plane = pixels[..., channel]
        darker = own[..., channel] - ndimage.minimum_filter(plane, window)     # a dark line right here
        lighter = ndimage.maximum_filter(plane, window) - own[..., channel]    # a light shape right here
        by_dark = darker > 2 * lighter
        by_light = lighter > 2 * darker
        size = np.maximum(darker, lighter)
        # next to a dark line the echo is lighter than the fill, and what is darker is the soft edge
        # of that line; next to a light shape it is the other way round
        up = np.where(by_dark, FIRST, SECOND) * size
        down = np.where(by_light, FIRST, SECOND) * size
        allowed = np.where(away[..., channel] > 0, up, down)
        step = np.abs(away[..., channel])
        fits = np.clip(2.0 - step / np.maximum(allowed, 1e-3), 0.0, 1.0)
        weight = np.minimum(weight, np.where(step < 0.5, 1.0, fits))
    # one decision for the three channels, so that a correction never changes the hue of a line
    out = values.copy()
    out[beside] = (values - away * weight[..., None])[beside]
    return out


def clean(pixels, reach=5.0, guard=5.5):
    pixels = pixels.astype(np.float32)
    calm, labels, trust, slope = regions(pixels, guard)
    even = rebuild(pixels, labels, trust, reach)          # every calm region evened out
    plain = degrain(pixels)                               # the grain gone, every shape as drawn
    out = np.where((slope > STRONG)[..., None], pixels, plain)      # a line keeps its own pixels
    # inside a region: what differs little from the evened fill is dirt, what differs much is drawing
    apart = plain - even
    size = ndimage.gaussian_filter(np.abs(apart).max(axis=2), 1.0)
    keep = np.clip((size - DIRT) / DIRT, 0.0, 1.0)
    keep = keep * keep * (3.0 - 2.0 * keep)
    inside = even + apart * keep[..., None]
    out[calm] = inside[calm]
    out = settle_edges(pixels, out, calm, slope)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("pictures", nargs="+")
    parser.add_argument("--painted", action="store_true")
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--guard", type=float, default=None)
    parser.add_argument("--diff", action="store_true")
    options = parser.parse_args()
    reach = options.reach if options.reach is not None else (2.0 if options.painted else 5.0)
    guard = options.guard if options.guard is not None else (3.0 if options.painted else 5.5)
    os.makedirs(options.out_dir, exist_ok=True)
    for path in options.pictures:
        picture = Image.open(path)
        alpha = picture.getchannel("A") if picture.mode in ("RGBA", "LA") else None
        before = np.asarray(picture.convert("RGB"))
        after = clean(before, reach, guard)
        name = os.path.splitext(os.path.basename(path))[0]
        result = Image.fromarray(after)
        if alpha is not None:
            result.putalpha(alpha)
        result.save(os.path.join(options.out_dir, name + ".png"))
        change = after.astype(np.int16) - before.astype(np.int16)
        print("%s: changed by more than 2 levels: %.1f%% of the picture, the largest change %d" % (
            name, float((np.abs(change).max(axis=2) > 2).mean()) * 100, int(np.abs(change).max())))
        if options.diff:
            shown = np.clip(128 + change * 8, 0, 255).astype(np.uint8)
            Image.fromarray(shown).save(os.path.join(options.out_dir, name + "-removed.png"))


if __name__ == "__main__":
    main()
