"""Measure the ripple of a generated picture: the echo of every line and the dirt in the fills.

    python app/ripple_check.py PIC.png [PIC2.png ...]

What the ripple is:

* next to every line the fill is not flat. Right at the line there is a strip one or two pixels wide
  that is LIGHTER than the fill on the light side of the edge (and darker on the dark side), then a
  weaker strip of the opposite sign. It is the ringing of the generator's decoder, an echo of the line.
  Where lines lie close together (knit ribs, eyelids, the rim of the glasses) the echoes fill the gaps
  and the place shimmers;
* the fill itself is slightly blotchy near detail and has a fine grain everywhere.

A drawing made by hand in flat colour has none of it: the fill keeps one value up to the line.

Numbers printed for each picture:

    echo        how much the first strip at a line overshoots the fill, in levels of 255
                (median and the worst tenth) and as a share of the contrast of that edge;
    blotches    unevenness 3-12 px across inside a flat fill, levels RMS: within 2-8 px of a strong
                line and far from the lines;
    grain       pixel noise of the flat fills, levels RMS.

The measure does not use the cleaner's model (deripple.py), so it can check the cleaner.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

STEP = 60        # an edge strong enough to measure: this many levels across four pixels
FLAT = 5         # the plateau beside it must stay within this range


def shifted(plane, offset, axis):
    """plane sampled at +offset along the axis, edges repeated."""
    if offset == 0:
        return plane
    places = np.clip(np.arange(plane.shape[axis]) + offset, 0, plane.shape[axis] - 1)
    return np.take(plane, places, axis=axis)


def echoes(plane):
    """Overshoot of the first strip beside strong edges, measured along rows and columns."""
    found_levels, found_share = [], []
    for axis in (0, 1):
        step = shifted(plane, 2, axis) - shifted(plane, -2, axis)
        strength = np.abs(step)
        # the centre of the edge: the steepest place along the axis
        centre = (strength > STEP) & (strength >= np.abs(shifted(step, 1, axis))) & (strength > np.abs(shifted(step, -1, axis)))
        for side in (1, -1):                      # the light side lies in this direction
            here = centre & (np.sign(step) == side)
            if not here.any():
                continue
            far_light = np.stack([shifted(plane, side * k, axis)[here] for k in range(7, 13)])
            far_dark = np.stack([shifted(plane, -side * k, axis)[here] for k in range(7, 13)])
            near_light = np.stack([shifted(plane, side * k, axis)[here] for k in range(1, 6)])
            light = np.median(far_light, axis=0)
            dark = np.median(far_dark, axis=0)
            good = (np.ptp(far_light, axis=0) < FLAT) & (light - dark > STEP * 0.8) & (np.ptp(far_dark, axis=0) < FLAT * 4)
            over = np.clip(near_light.max(axis=0) - light, 0, None)[good]
            found_levels.append(over)
            found_share.append(over / (light - dark)[good])
    if not found_levels:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(found_levels), np.concatenate(found_share)


def measure(pixels):
    """The ripple of a picture given as an RGB array, as numbers (see the top of the file)."""
    pixels = np.asarray(pixels).astype(np.float32)
    levels, share = [], []
    for channel in range(3):
        one, two = echoes(pixels[..., channel])
        levels.append(one)
        share.append(two)
    levels, share = np.concatenate(levels), np.concatenate(share)

    grey = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
    soft = ndimage.gaussian_filter(grey, 0.7)
    slope = np.hypot(*np.gradient(soft))
    distance = ndimage.distance_transform_edt(slope <= 20)        # to the nearest strong line
    calm = ndimage.maximum_filter(slope, 3) < 5                    # no edge of any kind right here
    # every calm patch is compared with its own colour, taken far from the lines; only patches that
    # are flat there count (a painted gradient is not dirt)
    patches, count = ndimage.label(calm)
    near_sum = near_count = far_sum = far_count = grain_sum = grain_count = 0.0
    if count:
        index = np.arange(1, count + 1)
        inner = calm & (distance > 10)
        size = ndimage.sum(inner, patches, index)
        # medians over the inner part of each patch
        order = np.argsort(patches[inner], kind="stable")
        ids = patches[inner][order]
        values = grey[inner][order]
        starts = np.searchsorted(ids, index, side="left")
        ends = np.searchsorted(ids, index, side="right")
        colour = np.full(count + 1, np.nan, np.float32)
        spread = np.full(count + 1, np.nan, np.float32)
        for one in np.nonzero(size >= 200)[0]:
            part = np.sort(values[starts[one]:ends[one]])
            colour[one + 1] = part[part.size // 2]
            spread[one + 1] = part[int(part.size * 0.9)] - part[int(part.size * 0.1)]
        flat = spread < 3.0
        usable = calm & flat[patches]
        rest = np.where(usable, grey - colour[patches], 0.0).astype(np.float32)
        cover = usable.astype(np.float32)

        def smooth(sigma):
            return ndimage.gaussian_filter(rest, sigma) / np.maximum(ndimage.gaussian_filter(cover, sigma), 1e-3)

        # blotches: unevenness 3-12 px across, the size of the generator's tiles and of the echo's tail;
        # a wider soft thing (a blush, a painted shadow) is not counted
        blotch = smooth(1.2) - smooth(5.0)
        grain = rest - smooth(1.2)
        near = usable & (distance >= 2) & (distance <= 8)
        far = usable & (distance > 16)
        near_sum, near_count = float((blotch[near] ** 2).sum()), float(near.sum())
        far_sum, far_count = float((blotch[far] ** 2).sum()), float(far.sum())
        grain_sum, grain_count = float((grain[usable] ** 2).sum()), float(usable.sum())

    def rms(total, number):
        return (total / number) ** 0.5 if number > 500 else float("nan")

    return {
        "echo": float(np.median(levels)) if levels.size else float("nan"),
        "echo_worst": float(np.percentile(levels, 90)) if levels.size else float("nan"),
        "echo_share": float(np.median(share)) if levels.size else float("nan"),
        "echo_share_worst": float(np.percentile(share, 90)) if levels.size else float("nan"),
        "places": int(levels.size),
        "blotches_near": rms(near_sum, near_count),
        "blotches_far": rms(far_sum, far_count),
        "grain": rms(grain_sum, grain_count),
    }


def check(path):
    found = measure(np.asarray(Image.open(path).convert("RGB")))
    if found["places"]:
        echo = "echo %4.1f median, %4.1f worst tenth (%.1f%% / %.1f%% of the edge), %d places" % (
            found["echo"], found["echo_worst"], found["echo_share"] * 100, found["echo_share_worst"] * 100, found["places"])
    else:
        echo = "echo: no strong edges"
    return "%s; blotches near lines %.2f, far from them %.2f; grain %.2f" % (
        echo, found["blotches_near"], found["blotches_far"], found["grain"])


def main():
    if not sys.argv[1:] or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    for path in sys.argv[1:]:
        print(path.replace("\\", "/").split("/")[-1])
        print("   ", check(path))


if __name__ == "__main__":
    main()
