"""Mend one spot of a picture that the generator drew wrongly: a coloured rim along a shape, a stray
stroke, a speck, a gap of light between a line and its shadow, a shadow where none can be.

    python app/retouch.py IN.png OUT.png SELECTION [--from x,y] [--keep shapes|lines|nothing] [--draw]

    SELECTION is one of
      --brush "x,y x,y ..." [--size 8]   a brush stroke through these points (repeat for several)
      --zone  "x,y x,y x,y ..."          the corners of an outline, in order around the spot
      --wand  x,y [--sense 10]           the thin mark under this point (repeat for several)

deripple.py removes the generator's trace in the pixels. What the generator DREW wrongly is a
drawing, and no rule can tell it from one that was meant. Such a spot is pointed at by hand, in two
steps: SELECT what is wrong, then PAINT IT OVER with the fill it lies on.

Which fill to paint with: the one under --from x,y; without it, the fill that borders the selection
most (dark lines do not count). The fill's own colours flow into the selection from where they
touch it, so shading goes on smoothly through the mended place.

What stays inside the selection (--keep):

* shapes (the default): dark lines stay, and so does every other shape that goes on outside the
  selection (a lash, the white of an eye, a lock of hair), with its thin tips and its soft edge.
  Select loosely. This takes away a coloured rim or a glow along a shape and a blotch;
* lines: only dark lines stay, every fill inside the selection becomes the chosen one. This takes
  away a shadow where none can be and closes a gap of light between a line and its shadow;
* nothing: what is selected is painted over, lines and all. Select just the thing itself. This
  takes away a stray stroke, a speck, an extra line. --wand implies it.

--draw is the other thing done by hand: it does not take away, it adds. The selection is painted
with one colour, that of the dark line nearest to it or the one under --from. For a piece of an
outline that the generator left out.

The measures of the tools (how narrow a line is, how far a tip reaches) were tuned on pictures about
1300 px across and grow with the picture: see scale_of().

A fill ("paint") here is a calm patch of the picture, skin or a black lash or cloth, together with
the thin tips that are joined to it and have its colour.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

GREY = np.array([0.299, 0.587, 0.114], np.float32)
LINE = 25.0       # a dark line stands out from the fills beside it by at least this many levels
LINE_SHARE = 0.35 # and by at least this share of their brightness (a shadow is a fifth darker, a line half)
LINE_WIDE = 13    # and is narrower than this many px
PAINT = 60        # a calm patch of at least this many px is a paint (a fill of the drawing)
ALIKE = 28.0      # a pixel within this many levels of a paint's colour, and joined to it, is that paint
TIP = 14          # that far (px) from its calm core a paint may reach with its thin tips
COLLAR = 3        # that far (px) beyond the selection the chosen fill is repainted too: there lies the
                  # soft fringe of the thing that goes, and left alone it stays as a ghost of it
MARK_WIDE = 11    # the wand's mark is narrower than this many px
REACH = 40        # and is followed no further than this many px from the click, unless an area is given
BASE = 1300.0     # these measures were tuned on pictures about this many px across; a larger picture has
                  # everything larger in it, lines and tips and fringes, so its measures grow with it


def scale_of(shape):
    """How much larger than the tuning pictures this one is: 1 up to BASE px on the short side."""
    return max(1.0, min(shape[0], shape[1]) / BASE)


def measures(scale=1.0):
    """The measures above, in pixels of a picture of the given scale."""
    def odd(value):
        return int(round(value)) | 1
    return {"line": odd(LINE_WIDE * scale), "mark": odd(MARK_WIDE * scale), "tip": int(round(TIP * scale)),
            "collar": int(round(COLLAR * scale)), "reach": int(round(REACH * scale)), "paint": int(round(PAINT * scale * scale)),
            "near": int(round(10 * scale))}


# ------------------------------------------------------------------ what the picture is made of

def dark_lines(pixels, fringe=True, wide=LINE_WIDE):
    """Dark lines: narrower than LINE_WIDE px and much darker than the fills beside them. A band of
    shadow is as narrow as an outline, but it is only a little darker than its fill: it is a fill,
    it has to be repainted and to give its colour. With fringe the soft pixel around the line is
    counted in, without it only the dark middle."""
    grey = ndimage.gaussian_filter(pixels @ GREY, 0.5)
    beside = ndimage.grey_closing(grey, size=(wide, wide))
    middle = (beside - grey) > np.maximum(LINE, LINE_SHARE * beside)
    return ndimage.binary_dilation(middle) if fringe else middle


def calm_patches(pixels):
    """Calm patches of the picture, numbered: places with no edge of any kind."""
    soft = ndimage.gaussian_filter(pixels, (0.7, 0.7, 0))
    slope = np.zeros(pixels.shape[:2], np.float32)
    for channel in range(3):
        down, across = np.gradient(soft[..., channel])
        slope = np.maximum(slope, np.hypot(down, across))
    return ndimage.label(ndimage.maximum_filter(slope, 3) < 5.0)[0]


def grown(pixels, core, blocked=None, tip=TIP):
    """A calm core together with the tips joined to it that have its colour."""
    distance, place = ndimage.distance_transform_edt(~core, return_indices=True)
    alike = (np.abs(pixels - pixels[place[0], place[1]]).max(axis=2) < ALIKE) & (distance <= tip)
    if blocked is not None:
        alike &= ~blocked
    return ndimage.binary_propagation(core, mask=alike | core)


def polygon(size, corners):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([tuple(int(round(number)) for number in corner) for corner in corners], fill=255)
    return np.asarray(mask) > 0


def brushed(size, strokes, width):
    """A selection made by brush strokes: every stroke is a list of points, width in px."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    half = width / 2.0
    for stroke in strokes:
        if len(stroke) > 1:
            draw.line([tuple(point) for point in stroke], fill=255, width=int(round(width)), joint="curve")
        for x, y in stroke:
            draw.ellipse([x - half, y - half, x + half, y + half], fill=255)
    return np.asarray(mask) > 0


# ------------------------------------------------------------------ letting colour flow in

def one_pass(part, todo, weight):
    value = part * weight[..., None]
    total = np.zeros_like(part)
    count = np.zeros(weight.shape, np.float32)
    total[1:] += value[:-1]
    count[1:] += weight[:-1]
    total[:-1] += value[1:]
    count[:-1] += weight[1:]
    total[:, 1:] += value[:, :-1]
    count[:, 1:] += weight[:, :-1]
    total[:, :-1] += value[:, 1:]
    count[:, :-1] += weight[:, 1:]
    fresh = total / np.maximum(count, 1e-6)[..., None]
    part[todo] = fresh[todo]


def even_out(part, todo, gives, passes, settle=False):
    """Let the colours of the todo pixels even out between the neighbours that give. With settle the
    passes go on, up to the given number, until nothing changes any more."""
    weight = gives.astype(np.float32)
    done = 0
    while done < passes:
        before = part[todo].copy() if settle else None
        for _ in range(min(200, passes - done)):
            one_pass(part, todo, weight)
        done += 200
        if settle and np.abs(part[todo] - before).max() < 0.02:
            break
    return part


def start_colours(part, todo, givers):
    """Every separate piece of todo starts from the mean colour of the givers that touch IT. The
    nearest giver by distance may lie behind a line and belong to another fill."""
    pieces, count = ndimage.label(todo)
    total = np.zeros((count + 1, 3), np.float64)
    number = np.zeros(count + 1, np.float64)
    for cut_todo, cut_giver in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                                ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                                ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                                ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        pair = (pieces[cut_todo] > 0) & givers[cut_giver]
        np.add.at(total, pieces[cut_todo][pair], part[cut_giver][pair])
        np.add.at(number, pieces[cut_todo][pair], 1)
    mean = (total / np.maximum(number, 1)[:, None]).astype(np.float32)
    fed = number > 0
    part[todo & fed[pieces]] = mean[pieces][todo & fed[pieces]]
    return part


def flow_in(part, todo, givers):
    """Fill the todo pixels of a patch with the colours of the givers that touch them.

    A small patch is solved directly. A large one is first solved at half size (and that one at a
    quarter, and so on), then only smoothed at full size: the same result in seconds, not a minute.
    At the coarse sizes a block that holds any kept pixel is a wall, so colours never leak through
    a line.
    """
    if not todo.any() or not (givers & ndimage.binary_dilation(todo)).any():
        return part
    gives = todo | givers
    start_colours(part, todo, givers)
    height, width = todo.shape
    if max(height, width) <= 96:
        return even_out(part, todo, gives, 40000, settle=True)
    tall, wide = height + height % 2, width + width % 2
    grow = ((0, tall - height), (0, wide - width))

    def blocks(mask):
        return np.pad(mask, grow).reshape(tall // 2, 2, wide // 2, 2)

    open_block = blocks(gives).all(axis=(1, 3))            # no kept pixel inside
    small_todo = open_block & blocks(todo).any(axis=(1, 3))
    small_givers = open_block & ~small_todo
    small = np.pad(part, grow + ((0, 0),)).reshape(tall // 2, 2, wide // 2, 2, 3).mean(axis=(1, 3))
    solved = flow_in(small.copy(), small_todo.copy(), small_givers)
    reached = small_todo & ndimage.binary_propagation(
        small_givers & ndimage.binary_dilation(small_todo), mask=small_todo | small_givers)
    big = np.repeat(np.repeat(solved, 2, axis=0), 2, axis=1)[:height, :width]
    take = todo & np.repeat(np.repeat(reached, 2, axis=0), 2, axis=1)[:height, :width]
    part[take] = big[take]
    return even_out(part, todo, gives, 400)


def repaint(pixels, todo, givers, flat=None):
    """Repaint the pixels of todo: the colour flows in from the givers that touch them. A pocket that
    no giver touches is left as it was: it lies behind a line, on the far side from the fill. Only
    when no giver touches todo anywhere (the fill was picked far away) all of it takes the flat
    colour, if one is given."""
    rows, cols = np.nonzero(todo)
    if not rows.size:
        return pixels
    top, bottom = max(rows.min() - 2, 0), min(rows.max() + 3, pixels.shape[0])
    left, right = max(cols.min() - 2, 0), min(cols.max() + 3, pixels.shape[1])
    part = pixels[top:bottom, left:right].copy()
    todo = todo[top:bottom, left:right].copy()
    givers = givers[top:bottom, left:right] & ~todo
    pockets, count = ndimage.label(todo)
    if count:
        fed = ndimage.binary_dilation(givers, structure=np.ones((3, 3)))
        reached = np.zeros(count + 1, bool)
        reached[np.unique(pockets[fed & todo])] = True
        if flat is not None and not reached.any():
            part[todo] = flat
        todo &= reached[pockets]
    out = pixels.copy()
    out[top:bottom, left:right] = flow_in(part, todo, givers)
    return out


# ------------------------------------------------------------------ selecting: the wand

def pick(pixels, points, sense=10.0, area=None, scale=1.0):
    """What the wand takes hold of: the thin mark under every point.

    A mark is a thin thing that stands out from the fill it lies on: darker (a stroke, a speck) or
    lighter (a glow, a pale rim along a dark shape, a gap of light) by more than sense levels. From
    the point the mark is followed as far as it goes on unbroken, but not further than REACH px: a
    long line is a part of the drawing. A wide shape, a lash or a lock, is no mark, so a stroke that
    grows out of a lash is picked without the lash. An area, if given, replaces the limit of REACH.
    A thin tip of a shape is a mark as well: the wand cannot know that it was meant. Look at what
    it picked before painting over, and take away what should stay.
    """
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    height, width = pixels.shape[:2]
    sized = measures(scale)
    wide, reach = sized["mark"], sized["reach"]
    down, across = np.mgrid[0:wide, 0:wide] - wide // 2
    round_window = down ** 2 + across ** 2 <= (wide / 2.0) ** 2      # a square one cuts a mark off squarely
    taken = np.zeros((height, width), bool)
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < width and 0 <= y < height):
            continue
        # only the surroundings of the click are looked at: a whole large picture takes seconds
        if area is not None and area.any():
            rows, cols = np.nonzero(area)
            box = [min(rows.min(), y), max(rows.max(), y), min(cols.min(), x), max(cols.max(), x)]
        else:
            box = [y - reach, y + reach, x - reach, x + reach]
        top, bottom = max(box[0] - wide, 0), min(box[1] + wide + 1, height)
        left, right = max(box[2] - wide, 0), min(box[3] + wide + 1, width)
        grey = ndimage.gaussian_filter(pixels[top:bottom, left:right] @ GREY, 0.5)
        darker = ndimage.grey_closing(grey, footprint=round_window) - grey
        lighter = grey - ndimage.grey_opening(grey, footprint=round_window)
        row, column = y - top, x - left
        # the click may land a pixel or two beside the mark: take the nearest pixel that is one
        up, back = max(row - 2, 0), max(column - 2, 0)
        strong = np.maximum(darker, lighter)[up:row + 3, back:column + 3] > sense
        if not strong.any():
            continue
        near_rows, near_columns = np.nonzero(strong)
        nearest = np.argmin((near_rows + up - row) ** 2 + (near_columns + back - column) ** 2)
        row, column = near_rows[nearest] + up, near_columns[nearest] + back
        marks = (darker if darker[row, column] >= lighter[row, column] else lighter) > sense
        if area is not None and area.any():
            marks &= area[top:bottom, left:right]
        else:
            rows, columns = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
            marks &= (rows - row) ** 2 + (columns - column) ** 2 <= reach ** 2
        pieces, _ = ndimage.label(marks, structure=np.ones((3, 3)))
        taken[top:bottom, left:right] |= pieces == pieces[row, column]
    return taken


# ------------------------------------------------------------------ painting over

def soften(result, part, edge, hard):
    """A repainted pixel right next to something that stayed keeps a mix of the two, as dark as it
    was: the soft edge of the thing that stayed."""
    if not hard.any() or not edge.any():
        return
    distance, place = ndimage.distance_transform_edt(~hard, return_indices=True)
    rows, cols = np.nonzero(edge & (distance <= 1.6))
    if rows.size:
        here = result[rows, cols]
        shape = part[place[0][rows, cols], place[1][rows, cols]]
        span = (shape - here) @ GREY
        was = (part[rows, cols] - here) @ GREY
        share = np.where(np.abs(span) > 8, np.clip(was / np.where(span == 0, 1, span), 0, 1), 0.0)
        result[rows, cols] = here + (shape - here) * share[:, None]


def paint_over(pixels, cover, source=None, keep="shapes", scale=1.0):
    """Paint the selected pixels over with one fill.

    cover    the selection: 1 where selected, 0 where not; in between at the soft edge of a brush;
    source   a point (x, y) on the fill to paint with. Without it the fill is the one that borders
             the selection most, dark lines not counted;
    keep     what stays inside the selection:
             "shapes"   dark lines and every other shape that goes on outside the selection, with
                        its tips and soft edge. For a rim, a glow, a blotch: select loosely;
             "lines"    dark lines only; every fill inside becomes the chosen one. For a shadow
                        where none can be, for a gap of light between a line and its shadow;
             "nothing"  nothing. For a stray stroke, a speck, an extra line: select the thing itself.

    Beyond the selection the chosen fill is repainted COLLAR px further: the soft fringe of a removed
    stroke lies there. Whatever stays keeps its soft edge. Where the edge of the selection cuts
    through something, the cut is as soft as the brush was.
    Returns the new picture, the number of pixels changed and the colour painted with (or None).
    """
    if keep not in ("shapes", "lines", "nothing"):
        raise ValueError("keep: shapes, lines or nothing")
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    cover = np.asarray(cover, np.float32)
    if cover.max() > 1.0:
        cover = cover / 255.0
    if not (cover >= 0.5).any():
        return pixels, 0, None
    height, width = cover.shape
    rows, cols = np.nonzero(cover > 0.02)
    reach = [rows.min(), rows.max(), cols.min(), cols.max()]
    if source is not None:
        x, y = int(round(source[0])), int(round(source[1]))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("the colour was picked outside the picture")
        reach = [min(reach[0], y), max(reach[1], y), min(reach[2], x), max(reach[3], x)]
    sized = measures(scale)
    tip, paint_size = sized["tip"], sized["paint"]
    pad = tip + int(round(30 * scale))
    top, bottom = max(reach[0] - pad, 0), min(reach[1] + pad + 1, height)
    left, right = max(reach[2] - pad, 0), min(reach[3] + pad + 1, width)
    part = pixels[top:bottom, left:right]
    alpha = np.clip(cover[top:bottom, left:right], 0.0, 1.0)
    touched = alpha > 0.02
    inside = alpha >= 0.5
    lines = dark_lines(part, wide=sized["line"])
    patches = calm_patches(part)
    sizes = np.bincount(patches.ravel())

    # the fill to paint with
    chosen, flat = None, None
    if source is not None:
        x, y = int(round(source[0])) - left, int(round(source[1])) - top
        spot = patches[y, x]
        if not spot:
            _, place = ndimage.distance_transform_edt(patches == 0, return_indices=True)
            spot = patches[place[0][y, x], place[1][y, x]]
        core = patches == spot
        chosen = grown(part, core, tip=tip)
        down, across = np.mgrid[0:part.shape[0], 0:part.shape[1]]
        close = core & ((down - y) ** 2 + (across - x) ** 2 <= 20 ** 2)
        flat = np.median(part[close if close.any() else core], axis=0)
    else:
        ring = ndimage.binary_dilation(inside, iterations=max(3, sized["collar"])) & ~inside & ~lines
        around = patches[ndimage.binary_dilation(inside, iterations=sized["near"]) & ~inside]
        names = [index for index in np.unique(around) if index and sizes[index] >= paint_size]
        names = sorted(names, key=lambda index: -int((around == index).sum()))[:16]
        most = 0
        for index in names:
            touch = int((grown(part, patches == index, blocked=inside, tip=tip) & ring).sum())
            if touch > most:
                chosen, most = grown(part, patches == index, tip=tip), touch

    # what stays inside the selection
    stay = np.zeros(inside.shape, bool)
    if keep != "nothing":
        # only the dark middle of a line: the soft pixel around it is repainted and mixed anew, or a
        # pale rim right at the line would stay
        stay |= dark_lines(part, fringe=False, wide=sized["line"])
    if keep == "shapes":
        # a shape may reach into the selection with its edge or a tip only, its calm middle outside
        for index in np.unique(patches[ndimage.binary_dilation(inside, iterations=tip)]):
            if index and sizes[index] >= paint_size:
                paint = grown(part, patches == index, tip=tip)
                if not (paint & inside).any() or (chosen is not None and (paint & chosen).any()):
                    continue
                if int((paint & ~inside).sum()) >= max(40, 0.15 * paint.sum()):
                    stay |= paint

    own = chosen if chosen is not None else ~lines
    todo = touched & ~stay
    collar = np.zeros(inside.shape, bool)
    if chosen is not None:
        collar = ndimage.binary_dilation(inside, iterations=sized["collar"]) & chosen & ~touched & ~lines & ~stay
        todo |= collar
    result = repaint(part.copy(), todo, own & ~todo, flat)
    soften(result, part, todo, stay)
    if chosen is not None:
        soften(result, part, collar, ~touched & ~own & ~todo)
    # the soft edge of the brush: what is not the fill itself goes as far as the brush covered it
    weight = np.where(own & (chosen is not None) | collar, 1.0, alpha)[..., None]
    result = np.where(todo[..., None], part + (result - part) * weight, part)
    changed = np.abs(result - part).max(axis=2) > 0.5
    colour = [int(round(value)) for value in np.median(result[changed], axis=0)] if changed.any() else None
    out = pixels.copy()
    out[top:bottom, left:right] = result
    return out, int(changed.sum()), colour


def probe(pixels, point, scale=1.0):
    """What lies under a point: its colour and, if that is a dark line, how wide the line is there.
    On a line the colour is that of its dark middle, not of the soft pixel that was clicked."""
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    height, width = pixels.shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("the point is outside the picture")
    wide = measures(scale)["line"]
    top, bottom = max(y - 2 * wide, 0), min(y + 2 * wide + 1, height)
    left, right = max(x - 2 * wide, 0), min(x + 2 * wide + 1, width)
    part = pixels[top:bottom, left:right]
    middle = dark_lines(part, fringe=False, wide=wide)
    rows, columns = np.mgrid[0:part.shape[0], 0:part.shape[1]]
    away = np.hypot(rows - (y - top), columns - (x - left))
    if (middle & (away <= 2.5)).any():
        deep = ndimage.distance_transform_edt(middle)
        here = middle & (away <= wide)
        colour = np.median(part[here & (deep >= min(2.0, deep[here].max()))], axis=0)
        return [int(round(value)) for value in colour], int(round(2 * deep[middle & (away <= wide / 2.0 + 2)].max()))
    colour = np.median(part[away <= 2.0], axis=0)
    return [int(round(value)) for value in colour], None


def draw(pixels, cover, source=None, scale=1.0):
    """Draw with one colour exactly where selected: a piece of an outline that is missing, or
    anything else that has to be painted in by hand.

    The colour is the one under the source point (on a line, the colour of its dark middle). Without
    a point it is the colour of the dark line nearest to the selection: the missing piece of an
    outline is drawn like the outline beside it. The edge of what is drawn is as soft as the brush.
    Returns the new picture, the number of pixels changed and the colour drawn with.
    """
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    cover = np.asarray(cover, np.float32)
    if cover.max() > 1.0:
        cover = cover / 255.0
    if not (cover >= 0.5).any():
        return pixels, 0, None
    sized = measures(scale)
    if source is not None:
        colour = np.array(probe(pixels, source, scale)[0], np.float32)
    else:
        height, width = cover.shape
        rows, cols = np.nonzero(cover > 0.02)
        pad = 3 * sized["reach"]
        top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad + 1, height)
        left, right = max(cols.min() - pad, 0), min(cols.max() + pad + 1, width)
        part = pixels[top:bottom, left:right]
        inside = cover[top:bottom, left:right] > 0.02
        middle = dark_lines(part, fringe=False, wide=sized["line"]) & ~inside
        away = ndimage.distance_transform_edt(~inside)
        colour = None
        for near in (sized["reach"] // 3, sized["reach"], pad):
            lines = middle & (away <= near)
            if int(lines.sum()) >= 12:
                colour = np.median(part[lines], axis=0)
                break
        if colour is None:
            raise ValueError("there is no line nearby to take the colour from: click a line or a fill with the eyedropper")
    weight = np.clip(cover, 0.0, 1.0)[..., None]
    out = pixels + (colour - pixels) * weight
    changed = np.abs(out - pixels).max(axis=2) > 0.5
    return out, int(changed.sum()), [int(round(value)) for value in colour]


def snap(pixels, cover, guess, scale=1.0, tolerance=24.0, border="guess"):
    """Draw a filler's guess again with the picture's own colours.

    A network that draws a spot anew knows where a line or the border of two fills should run, but
    it paints with colours of its own: a little off, a little blurred, with grain. Here every pixel
    of its guess is replaced by the nearest of the real things around the spot: one of the fills
    that touch the selection (each carried through the spot with its own shading) or the dark line
    beside it; between two of them a pixel keeps the share the guess gave it, made steeper, so a
    border comes out as sharp as the borders of the picture. Where the guess is like none of them
    (further than tolerance levels) it is left as the network drew it.

    pixels   the picture as it was; cover: the selection; guess: the filler's picture;
    border   what to do with a pixel that lies between two of the real things: "steep" draws the
             border sharper than the guess had it, "keep" leaves the share the guess gave, "guess"
             leaves such a pixel exactly as the filler drew it (only the plain fills are replaced).
    Returns the new picture and the number of pixels changed against the picture as it was.
    """
    pixels = np.asarray(pixels)[..., :3].astype(np.float32)
    guess = np.asarray(guess)[..., :3].astype(np.float32)
    cover = np.asarray(cover, np.float32)
    if cover.max() > 1.0:
        cover = cover / 255.0
    if not (cover >= 0.5).any():
        return pixels, 0
    sized = measures(scale)
    height, width = cover.shape
    rows, cols = np.nonzero(cover > 0.02)
    pad = sized["tip"] + int(round(30 * scale))
    top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad + 1, height)
    left, right = max(cols.min() - pad, 0), min(cols.max() + pad + 1, width)
    part, drawn = pixels[top:bottom, left:right], guess[top:bottom, left:right]
    alpha = np.clip(cover[top:bottom, left:right], 0.0, 1.0)
    inside = alpha >= 0.5
    hole = ndimage.binary_dilation(alpha > 0.02, iterations=max(2, sized["collar"]))
    middle = dark_lines(part, fringe=False, wide=sized["line"])
    lines = ndimage.binary_dilation(middle)
    patches = calm_patches(part)
    sizes = np.bincount(patches.ravel())
    ring = ndimage.binary_dilation(hole, iterations=3) & ~hole & ~lines
    around = patches[ndimage.binary_dilation(hole, iterations=sized["near"]) & ~hole]
    names = [index for index in np.unique(around) if index and sizes[index] >= sized["paint"]]
    names = sorted(names, key=lambda index: -int((around == index).sum()))[:8]
    fields = []
    pockets, count = ndimage.label(hole)
    for index in names:
        paint = grown(part, patches == index, blocked=hole, tip=sized["tip"])
        if int((paint & ring).sum()) < 4:
            continue
        givers = paint & ~hole
        field = repaint(part.copy(), hole, givers)[hole]
        # a piece of the selection this fill does not touch is none of its business
        reached = np.zeros(count + 1, bool)
        reached[np.unique(pockets[ndimage.binary_dilation(givers, structure=np.ones((3, 3))) & hole])] = True
        field[~reached[pockets[hole]]] = 1e4
        fields.append(field)
    away = ndimage.distance_transform_edt(~hole)
    beside = middle & ~hole & (away <= max(sized["reach"] // 2, 8))
    if int(beside.sum()) >= 12:
        fields.append(np.broadcast_to(np.median(part[beside], axis=0), (int(hole.sum()), 3)).astype(np.float32))
    if not fields:
        out = pixels.copy()
        out[top:bottom, left:right] = part + (drawn - part) * alpha[..., None]
        return out, int((np.abs(out - pixels).max(axis=2) > 0.5).sum())
    seen = drawn[hole]
    stack = np.stack(fields)                                         # thing, pixel, channel
    apart = np.linalg.norm(stack - seen[None], axis=2)
    order = np.argsort(apart, axis=0)
    spot = np.arange(seen.shape[0])
    first = stack[order[0], spot]
    if len(fields) > 1:
        second = stack[order[1], spot]
        span = second - first
        share = np.clip(((seen - first) * span).sum(axis=1) / np.maximum((span * span).sum(axis=1), 1e-3), 0.0, 1.0)
        if border == "steep":
            share = np.clip((share - 0.3) / 0.4, 0.0, 1.0)          # a blurred border is drawn sharp again
            share = share * share * (3.0 - 2.0 * share)
        mixed = first + span * share[:, None]
        if border == "guess":
            between = (share > 0.08) & (share < 0.92)
            mixed = np.where(between[:, None], seen, np.where((share >= 0.92)[:, None], second, first))
    else:
        mixed = first
    off = np.linalg.norm(seen - mixed, axis=1)
    own = np.clip((off - tolerance) / tolerance, 0.0, 1.0)[:, None]  # like nothing around: the guess stays
    result = part.copy()
    result[hole] = mixed + (seen - mixed) * own
    weight = np.where(hole, 1.0, 0.0).astype(np.float32)
    weight = np.minimum(weight, np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(inside, iterations=max(2, sized["collar"])).astype(np.float32), 0.7) * 1.4, 0, 1))
    result = part + (result - part) * weight[..., None]
    out = pixels.copy()
    out[top:bottom, left:right] = result
    return out, int((np.abs(result - part).max(axis=2) > 0.5).sum())


def to_bytes(picture):
    return np.clip(np.rint(picture), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--brush", action="append", default=[])
    parser.add_argument("--size", type=float, default=8.0)
    parser.add_argument("--zone")
    parser.add_argument("--wand", action="append", default=[])
    parser.add_argument("--sense", type=float, default=10.0)
    parser.add_argument("--from", dest="origin")
    parser.add_argument("--keep", choices=["shapes", "lines", "nothing"])
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--show", help="save the selection here, as a picture, to look at it")
    options = parser.parse_args()

    def point(text):
        return tuple(float(number) for number in text.split(","))

    picture = Image.open(options.source)
    alpha = picture.getchannel("A") if picture.mode in ("RGBA", "LA") else None
    pixels = np.asarray(picture.convert("RGB")).astype(np.float32)
    scale = scale_of(pixels.shape)
    cover = np.zeros(pixels.shape[:2], bool)
    if options.zone:
        cover |= polygon(picture.size, [point(corner) for corner in options.zone.split()])
    if options.brush:
        cover |= brushed(picture.size, [[point(corner) for corner in stroke.split()] for stroke in options.brush], options.size)
    if options.wand:
        cover = pick(pixels, [point(text) for text in options.wand], options.sense, cover if cover.any() else None, scale)
    if not cover.any():
        parser.error("nothing is selected: give --brush, --zone or --wand")
    if options.show:
        shown = to_bytes(pixels)
        shown[cover] = shown[cover] // 2 + np.array([127, 0, 110], np.uint8)
        Image.fromarray(shown).save(options.show)
    origin = point(options.origin) if options.origin else None
    if options.draw:
        out, count, colour = draw(pixels, cover, origin, scale)
        if alpha is not None:       # what is drawn outside the figure has to be seen
            alpha = Image.fromarray(np.maximum(np.asarray(alpha), (cover * 255).astype(np.uint8)))
    else:
        out, count, colour = paint_over(pixels, cover, origin, options.keep or ("nothing" if options.wand else "shapes"), scale)
    result = Image.fromarray(to_bytes(out))
    if alpha is not None:
        result.putalpha(alpha)
    result.save(options.target)
    print("painted over: %d px, colour %s" % (count, colour))


if __name__ == "__main__":
    main()
