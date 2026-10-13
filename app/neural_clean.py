"""Clean a picture with a neural network made for anime art.

    python -I app/neural_clean.py MODEL OUT_DIR PIC.png [PIC2.png ...] [--keep-big] [--cpu]

MODEL is one of:

    anime6b     RealESRGAN_x4plus_anime_6B.pth   17.9 MB, BSD-3-Clause, github.com/xinntao/Real-ESRGAN
    animevideo  realesr-animevideov3.pth          2.5 MB, the same project, a small fast network
    waifu1, waifu2, waifu3   cunet/art/noise1..3.pth, 5.1 MB each, MIT, github.com/nagadomi/nunif
                (waifu2x: takes noise off at the same size, weak to strong)

The files are expected in models/realesrgan/ and models/waifu2x/. They are read
with weights_only=True, so a model file cannot run code. The networks themselves are written out
below (RRDBNet and SRVGGNetCompact after Real-ESRGAN, CUNet after waifu2x), nothing else is installed.

The first two enlarge fourfold: these networks were trained to redraw a degraded anime frame
cleanly at four times the size. Reduced again to the size it came in, the picture keeps its drawing
but loses what the network took for damage. --keep-big also saves the fourfold picture.
"""
import argparse
import os
import site
import sys

sys.path.append(site.getusersitepackages())      # with -I the user's own packages are not on the path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class DenseBlock(nn.Module):
    def __init__(self, width=64, grow=32):
        super().__init__()
        self.conv1 = nn.Conv2d(width, grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(width + grow, grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(width + 2 * grow, grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(width + 3 * grow, grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(width + 4 * grow, width, 3, 1, 1)

    def forward(self, x):
        one = F.leaky_relu(self.conv1(x), 0.2)
        two = F.leaky_relu(self.conv2(torch.cat((x, one), 1)), 0.2)
        three = F.leaky_relu(self.conv3(torch.cat((x, one, two), 1)), 0.2)
        four = F.leaky_relu(self.conv4(torch.cat((x, one, two, three), 1)), 0.2)
        five = self.conv5(torch.cat((x, one, two, three, four), 1))
        return five * 0.2 + x


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = DenseBlock(), DenseBlock(), DenseBlock()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """The network of RealESRGAN_x4plus_anime_6B: six blocks, fourfold."""

    def __init__(self, blocks=6):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(blocks)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        first = self.conv_first(x)
        feat = first + self.conv_body(self.body(first))
        feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


class CompactNet(nn.Module):
    """The network of realesr-animevideov3: sixteen plain layers, fourfold."""

    def __init__(self, layers=16):
        super().__init__()
        body = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)]
        for _ in range(layers):
            body += [nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)]
        body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.body = nn.ModuleList(body)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return F.pixel_shuffle(out, 4) + F.interpolate(x, scale_factor=4, mode="nearest")


class SqueezeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width // 8, 1)
        self.conv2 = nn.Conv2d(width // 8, width, 1)

    def forward(self, x):
        weight = torch.sigmoid(self.conv2(F.relu(self.conv1(F.adaptive_avg_pool2d(x, 1)))))
        return x * weight


class TwoConvs(nn.Module):
    def __init__(self, enter, middle, leave, squeeze):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(enter, middle, 3), nn.LeakyReLU(0.1), nn.Conv2d(middle, leave, 3), nn.LeakyReLU(0.1))
        self.seblock = SqueezeBlock(leave) if squeeze else None

    def forward(self, x):
        out = self.conv(x)
        return self.seblock(out) if self.seblock is not None else out


class UNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoConvs(3, 32, 64, False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2)
        self.conv2 = TwoConvs(64, 128, 64, True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv_bottom = nn.Conv2d(64, 3, 3)

    def forward(self, x):
        one = self.conv1(x)
        two = F.leaky_relu(self.conv2_up(self.conv2(F.leaky_relu(self.conv1_down(one), 0.1))), 0.1)
        three = F.leaky_relu(self.conv3(F.pad(one, (-4, -4, -4, -4)) + two), 0.1)
        return self.conv_bottom(three)


class UNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoConvs(3, 32, 64, False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2)
        self.conv2 = TwoConvs(64, 64, 128, True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2)
        self.conv3 = TwoConvs(128, 256, 128, True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2)
        self.conv4 = TwoConvs(128, 64, 64, True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv_bottom = nn.Conv2d(64, 3, 3)

    def forward(self, x):
        one = self.conv1(x)
        two = self.conv2(F.leaky_relu(self.conv1_down(one), 0.1))
        three = F.leaky_relu(self.conv3_up(self.conv3(F.leaky_relu(self.conv2_down(two), 0.1))), 0.1)
        four = F.leaky_relu(self.conv4_up(self.conv4(F.pad(two, (-4, -4, -4, -4)) + three)), 0.1)
        five = F.leaky_relu(self.conv5(F.pad(one, (-16, -16, -16, -16)) + four), 0.1)
        return self.conv_bottom(five)


class CUNet(nn.Module):
    """The denoising network of waifu2x: two U-shaped nets one after the other. It has no padding,
    so what comes out is 28 px smaller on every side than what goes in."""

    def __init__(self):
        super().__init__()
        self.unet1, self.unet2 = UNet1(), UNet2()

    def forward(self, x):
        first = self.unet1(x)
        return F.pad(first, (-20, -20, -20, -20)) + self.unet2(first)


# name: file, key of the weights in the file, network, enlarges by
MODELS = {
    "anime6b": ("realesrgan/RealESRGAN_x4plus_anime_6B.pth", "params_ema", RRDBNet, 4),
    "animevideo": ("realesrgan/realesr-animevideov3.pth", "params", CompactNet, 4),
    "waifu1": ("waifu2x/pretrained_models/cunet/art/noise1.pth", "state_dict", CUNet, 1),
    "waifu2": ("waifu2x/pretrained_models/cunet/art/noise2.pth", "state_dict", CUNet, 1),
    "waifu3": ("waifu2x/pretrained_models/cunet/art/noise3.pth", "state_dict", CUNet, 1),
}
_loaded = {}


def load(name, device):
    if (name, device) not in _loaded:
        file, key, make, _ = MODELS[name]
        weights = torch.load(os.path.join(HOME, file), map_location="cpu", weights_only=True)[key]
        network = make()
        network.load_state_dict(weights, strict=True)
        network.eval().to(device)
        if device == "cuda":
            network.half()
        _loaded[(name, device)] = network
    return _loaded[(name, device)]


def denoise(pixels, name, device="cuda", tile=200, shrink=28):
    """The same picture with the noise taken off by waifu2x, worked through in tiles."""
    network = load(name, device)
    height, width = pixels.shape[:2]
    tall, wide = -(-height // tile) * tile, -(-width // tile) * tile
    padded = np.pad(np.array(pixels), ((shrink, shrink + tall - height), (shrink, shrink + wide - width), (0, 0)), mode="reflect")
    source = torch.from_numpy(padded).permute(2, 0, 1).float().div(255.0)[None]
    out = np.zeros((tall, wide, 3), np.float32)
    with torch.no_grad():
        for top in range(0, tall, tile):
            for left in range(0, wide, tile):
                part = source[:, :, top:top + tile + 2 * shrink, left:left + tile + 2 * shrink].to(device)
                if device == "cuda":
                    part = part.half()
                out[top:top + tile, left:left + tile] = network(part).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(np.rint(out[:height, :width] * 255.0), 0, 255).astype(np.uint8)


def enlarge(pixels, name, device="cuda", tile=256, margin=16):
    """The picture four times larger, worked through in tiles so that it fits in memory."""
    network = load(name, device)
    height, width = pixels.shape[:2]
    source = torch.from_numpy(np.array(pixels)).permute(2, 0, 1).float().div(255.0)[None]
    big = np.zeros((height * 4, width * 4, 3), np.float32)
    with torch.no_grad():
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                y0, x0 = max(top - margin, 0), max(left - margin, 0)
                y1, x1 = min(top + tile + margin, height), min(left + tile + margin, width)
                part = source[:, :, y0:y1, x0:x1].to(device)
                if device == "cuda":
                    part = part.half()
                out = network(part).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                bottom, right = min(top + tile, height), min(left + tile, width)
                big[top * 4:bottom * 4, left * 4:right * 4] = out[(top - y0) * 4:(bottom - y0) * 4, (left - x0) * 4:(right - x0) * 4]
    return np.clip(np.rint(big * 255.0), 0, 255).astype(np.uint8)


def clean(pixels, name, device="cuda"):
    """The picture cleaned by the named network, at the size it came in.

    A network that enlarges works fourfold and the result is brought back by the plain mean of
    every 4 x 4 block. A sharper way of reducing (Lanczos) overshoots at the crisp edges the
    network draws and so puts the echo back, twice as strong as it was.
    """
    if MODELS[name][3] == 1:
        return denoise(pixels, name, device)
    big = Image.fromarray(enlarge(pixels, name, device))
    return np.asarray(big.resize((pixels.shape[1], pixels.shape[0]), Image.BOX))


def double(pixels, device="cuda", painted=False):
    """The picture at twice its size, drawn sharp by the anime network.

    For a painted picture the network's answer is taken whole: it draws the soft small details
    anew, which is what such a picture wants.

    For a flat-colour picture the network gives only the SHARPNESS of lines and borders. Left to
    itself it wipes pale freckles off a cheek, draws thin red ticks where they were and turns a
    brown nose line bright red. So here the picture keeps its own colours everywhere (the colour
    is taken from the picture, enlarged plainly), and takes from the network only how light or dark
    a pixel is, and only near a line or a border of two fills. A calm fill, with its freckles and
    blush, stays the picture's own.

    The result still carries the network's grain; run the cleaner over it (gently) afterwards.
    """
    from scipy import ndimage
    size = (pixels.shape[1] * 2, pixels.shape[0] * 2)
    sharp = np.asarray(Image.fromarray(enlarge(pixels, "anime6b", device)).resize(size, Image.BOX), np.float32)
    if painted:
        return np.clip(np.rint(sharp), 0, 255).astype(np.uint8)
    grey = np.array([0.299, 0.587, 0.114], np.float32)
    soft = np.asarray(Image.fromarray(pixels).resize(size, Image.BICUBIC), np.float32)
    light = ndimage.gaussian_filter(soft @ grey, 1.0)
    slope = np.hypot(ndimage.sobel(light, axis=0), ndimage.sobel(light, axis=1)) / 8.0
    near = ndimage.binary_dilation(slope > 7.0, iterations=3)                       # a line or a border, and a little around it
    share = np.clip(ndimage.gaussian_filter(near.astype(np.float32), 1.5) * 1.3, 0.0, 1.0)
    tint = ndimage.gaussian_filter(soft, (1.0, 1.0, 0)) - light[..., None]          # the picture's colour apart from its lightness
    merged = (soft @ grey + (sharp @ grey - soft @ grey) * share)[..., None] + tint
    return np.clip(np.rint(merged), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("out_dir")
    parser.add_argument("pictures", nargs="+")
    parser.add_argument("--keep-big", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    device = "cpu" if options.cpu or not torch.cuda.is_available() else "cuda"
    os.makedirs(options.out_dir, exist_ok=True)
    for path in options.pictures:
        pixels = np.asarray(Image.open(path).convert("RGB"))
        name = os.path.splitext(os.path.basename(path))[0]
        if MODELS[options.model][3] == 1:
            back = Image.fromarray(denoise(pixels, options.model, device))
        else:
            big = enlarge(pixels, options.model, device)
            if options.keep_big:
                Image.fromarray(big).save(os.path.join(options.out_dir, name + "-x4.png"))
            back = Image.fromarray(big).resize((pixels.shape[1], pixels.shape[0]), Image.BOX)
        back.save(os.path.join(options.out_dir, name + ".png"))
        print("%s: done with %s on %s" % (name, options.model, device))


if __name__ == "__main__":
    main()
