# Model files

Nothing here is part of the repository. `python get_models.py` (in the folder above) downloads the
two files below and checks each by size and MD5.

| File | For | Size | From |
| --- | --- | --- | --- |
| `lama/anime-manga-big-lama.pt` | redrawing a selected spot | 196 MB | release "AnimeMangaInpainting" of [Sanster/models](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting) |
| `realesrgan/RealESRGAN_x4plus_anime_6B.pth` | doubling the size | 17 MB | release v0.2.2.4 of [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4), BSD-3-Clause |

The first is a TorchScript file (it holds the network's own code as well as its weights) and is read
with `torch.jit.load`. The second holds weights only and is read with `weights_only=True`.

Cleaning a picture ("Remove ripple") needs neither.
