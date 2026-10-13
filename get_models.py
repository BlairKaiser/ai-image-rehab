"""Download the two model files the neural tools of AI Image Rehab use, and check them.

    python get_models.py

They go into models/. Cleaning a picture needs neither; "Remove a mark" needs the first, "Double the
size" the second. A file that is already there and whole is left alone.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    # folder, file, where from, bytes, md5
    ("lama", "anime-manga-big-lama.pt",
     "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
     205717870, "29f284f36a0a510bcacf39ecf4c4d54f"),
    ("realesrgan", "RealESRGAN_x4plus_anime_6B.pth",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
     17938799, "d58ce384064ec1591c2ea7b79dbf47ba"),
]


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for piece in iter(lambda: file.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def main():
    failed = False
    for folder, name, address, size, check in FILES:
        target = os.path.join(HERE, "models", folder, name)
        if os.path.exists(target) and os.path.getsize(target) == size and md5(target) == check:
            print("%s: already here" % name)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print("%s: %d MB from %s" % (name, size // 1048576, address))
        part = target + ".part"
        with urllib.request.urlopen(address) as answer, open(part, "wb") as file:
            done = 0
            for piece in iter(lambda: answer.read(1 << 20), b""):
                file.write(piece)
                done += len(piece)
                print("\r  %3d %%" % (done * 100 // size), end="", flush=True)
        print()
        if os.path.getsize(part) != size or md5(part) != check:
            os.remove(part)
            print("  the file that came is not the one expected: removed. Try again.")
            failed = True
            continue
        os.replace(part, target)
        print("  ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
