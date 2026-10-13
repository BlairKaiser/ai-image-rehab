"""A local page for cleaning a picture by hand: open it, clean it, mend the spots, download.

    python -I app/ripple_studio.py [--port 4390] [--dir FOLDER] [--out FOLDER] [--open]

Then open http://127.0.0.1:4390 in a browser. The page is ripple_studio.html beside this file; its
fonts and the logo are in the folder studio/.

It listens on this computer only (127.0.0.1) and works on one picture at a time, kept in memory
with every step, so a step can be undone and redone. A picture with see-through parts keeps them.
The tools are read anew when their files change, so mending them needs no restart. What the buttons do is done by the same code
as the command-line tools: deripple.py (automatic cleaning), retouch.py (a spot selected by hand),
neural_clean.py (double size through Real-ESRGAN) and neural_fill.py (a spot drawn anew by LaMa),
each only if its model file is in models.

--dir   the folder whose pictures the page may list and open by name (its button "From folder").
        By default examples/ in the project's folder (your Pictures folder if it is not there). Nothing outside it is read by path; a picture from
        anywhere else is opened by the page's own file button, by dropping it or by pasting, which
        sends its bytes.
--out   where "Save to folder" writes. By default out/ in the project's folder.
        The work in hand is written into its subfolder .session after every step (a picture per
        step, up to forty of them) and taken up again when the page is started.
--open  open the page in the browser once it is up.
"""
import argparse
import importlib
import importlib.util
import io
import json
import os
import site
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

sys.path.append(site.getusersitepackages())      # with -I the user's own packages are not on the path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import deripple  # noqa: E402
import retouch  # noqa: E402
import ripple_check  # noqa: E402

ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
PICTURES_HOME = os.path.join(os.path.expanduser("~"), "Pictures")
PLACES = {"dir": EXAMPLES if os.path.isdir(EXAMPLES) else PICTURES_HOME if os.path.isdir(PICTURES_HOME) else os.path.expanduser("~"),
          "out": os.path.join(ROOT, "out")}
BIGGEST = 60 * 1024 * 1024           # an uploaded file may not be larger than this
PICTURES = (".png", ".jpg", ".jpeg", ".webp")
KEPT = 40                            # that many steps are kept in memory, so that many can be undone
ASSETS = {"logo.svg": "image/svg+xml", "RobotoCondensed.ttf": "font/ttf", "RobotoCondensed-ExtraBold.ttf": "font/ttf",
          "ex-ai.png": "image/png", "ex-shapes.png": "image/png", "ex-lines.png": "image/png", "ex-draw.png": "image/png", "ex-auto.png": "image/png"}
WORK = {"name": "", "steps": [], "undone": [], "next": 1}   # the picture being worked on
BUSY = threading.Lock()              # one step at a time
COUNTING = threading.Lock()          # one measuring at a time
THUMBS = {}
READ = {}                            # when each tool's file was read


def refresh():
    """A tool whose file has changed is read anew, so mending the tools does not need a restart of
    the page, and the picture being worked on stays where it is."""
    late = tuple(sys.modules[name] for name in ("neural_fill", "neural_clean") if name in sys.modules)
    for tool in (deripple, retouch, ripple_check) + late:
        stamp = os.path.getmtime(tool.__file__)
        if READ.setdefault(tool.__name__, stamp) != stamp:
            importlib.reload(tool)
            READ[tool.__name__] = stamp


def kept_in():
    return os.path.join(PLACES["out"], ".session")


def persist():
    """Write the work down: the list of steps and every step's picture. The page is restarted, the
    computer is switched off, the work is still there when the page comes back."""
    folder = kept_in()
    os.makedirs(folder, exist_ok=True)
    steps = WORK["steps"] + WORK["undone"]
    for step in steps:
        path = os.path.join(folder, "step-%d.png" % step["id"])
        if not os.path.exists(path):
            if step["png"] is None:
                step["png"] = encode(whole(step))
            with open(path, "wb") as file:
                file.write(step["png"])
    names = {"step-%d.png" % step["id"] for step in steps}
    for name in os.listdir(folder):
        if name.startswith("step-") and name not in names:
            os.remove(os.path.join(folder, name))
    listed = {"name": WORK["name"], "next": WORK["next"],
              "steps": [{"id": step["id"], "label": step["label"], "kind": step["kind"]} for step in WORK["steps"]],
              "undone": [{"id": step["id"], "label": step["label"], "kind": step["kind"]} for step in WORK["undone"]]}
    with open(os.path.join(folder, "session.json"), "w", encoding="utf-8") as file:
        json.dump(listed, file, ensure_ascii=False)


def revive():
    """Take up the work that was written down, if any."""
    path = os.path.join(kept_in(), "session.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as file:
            listed = json.load(file)
        for row, target in ((listed["steps"], "steps"), (listed.get("undone", []), "undone")):
            WORK[target] = []
            for step in row:
                pixels, solid = split(Image.open(os.path.join(kept_in(), "step-%d.png" % step["id"])))
                WORK[target].append({"id": step["id"], "label": step["label"], "kind": step["kind"], "pixels": pixels, "alpha": solid,
                                     "png": None, "numbers": None})
        WORK["name"], WORK["next"] = listed["name"], listed["next"]
    except (OSError, KeyError, ValueError) as problem:      # a broken record is no reason not to start
        WORK["steps"], WORK["undone"] = [], []
        print("the saved work could not be read: %s" % problem, flush=True)


def split(picture):
    """The colours of a picture and, if any of it is see-through, how solid each pixel is."""
    if picture.mode in ("RGBA", "LA", "PA") or "transparency" in picture.info:
        whole = picture.convert("RGBA")
        solid = np.asarray(whole.getchannel("A"))
        return np.asarray(whole.convert("RGB")), (solid.copy() if solid.min() < 255 else None)
    return np.asarray(picture.convert("RGB")), None


def whole(step):
    picture = Image.fromarray(step["pixels"])
    if step["alpha"] is not None:
        picture.putalpha(Image.fromarray(step["alpha"]))
    return picture


def encode(picture, level=1):
    held = io.BytesIO()
    picture.save(held, "PNG", compress_level=level)
    return held.getvalue()


def add_step(label, kind, pixels, alpha="as before"):
    """kind: "start", "clean", "paint", "enlarge". A new step throws the undone ones away. What is
    see-through stays so from step to step, unless the step says otherwise."""
    if isinstance(alpha, str):
        before = WORK["steps"][-1]["alpha"] if WORK["steps"] else None
        alpha = before if before is not None and before.shape == pixels.shape[:2] else None
    WORK["steps"].append({"id": WORK["next"], "label": label, "kind": kind, "pixels": pixels, "alpha": alpha, "png": None, "numbers": None})
    WORK["next"] += 1
    WORK["undone"] = []
    if len(WORK["steps"]) > KEPT:        # the oldest mending goes; the picture as it came and the cleanings stay
        for index, step in enumerate(WORK["steps"]):
            if step["kind"] == "paint":
                del WORK["steps"][index]
                break


def start(name, picture):
    WORK["name"] = os.path.splitext(os.path.basename(name))[0] or "picture"
    WORK["steps"] = []
    add_step("start", "start", *split(picture))


def scale():
    return retouch.scale_of(current().shape)


def has_torch():
    """PyTorch is installed. Asked without loading it: that takes seconds."""
    if importlib.util.find_spec("torch") is None:
        importlib.invalidate_caches()      # it may have been installed since the page was started
        return importlib.util.find_spec("torch") is not None
    return True


def lacking():
    """What this copy still needs before its neural tools work: the commands to run, in order.
    The page shows them in place of the tools that are switched off."""
    needs = [] if has_torch() else ["pip install torch"]
    if not (os.path.exists(model_file()) and os.path.exists(fill_file())):
        needs.append("python get_models.py")
    return needs


def state():
    size = scale() if WORK["steps"] else 1.0
    torch_here = has_torch()
    return {"name": WORK["name"], "redo": len(WORK["undone"]), "enlarge": torch_here and os.path.exists(model_file()),
            "ai": torch_here and os.path.exists(fill_file()), "setup": lacking(), "strong": strong_ready(), "guided": guide_ready(),
            "line": int(round(6 * size)), "seethrough": bool(WORK["steps"]) and WORK["steps"][-1]["alpha"] is not None,
            "steps": [{"id": step["id"], "label": step["label"], "kind": step["kind"],
                       "width": int(step["pixels"].shape[1]), "height": int(step["pixels"].shape[0])}
                      for step in WORK["steps"]]}


def step_by(identity):
    for step in WORK["steps"] + WORK["undone"]:
        if step["id"] == identity:
            return step
    raise ValueError("that step is gone")


def current():
    return WORK["steps"][-1]["pixels"]


def model_file():
    return os.path.join(ROOT, "models", "realesrgan", "RealESRGAN_x4plus_anime_6B.pth")


def fill_file():
    return os.path.join(ROOT, "models", "lama", "anime-manga-big-lama.pt")


# ------------------------------------------------------------------ the strong model, in a helper process
STRONG = {"python": os.environ.get("FILL_PYTHON", ""), "port": int(os.environ.get("FILL_PORT", "4392")), "process": None}


def strong_ready():
    """The diffusion model can be asked for: there is an interpreter that has its libraries, and the model's file."""
    import diffusion_fill
    return bool(STRONG["python"]) and os.path.exists(STRONG["python"]) and diffusion_fill.available()


def guide_ready():
    """The guided way is offered only when asked for by FILL_GUIDED=1: in the trials it took the big marks out but
    drew small marks of its own in place of the small ones, a minute a spot."""
    import diffusion_fill
    return os.environ.get("FILL_GUIDED") == "1" and os.path.exists(diffusion_fill.GUIDE) and os.path.getsize(diffusion_fill.GUIDE) > 4.9e9


def helper():
    """Start the helper process if it is not running and wait until it answers."""
    address = "http://127.0.0.1:%d/" % STRONG["port"]
    for attempt in range(90):
        try:
            urllib.request.urlopen(address, timeout=2).read()
            return
        except OSError:
            if attempt == 0:
                log = open(os.path.join(PLACES["out"], "fill-worker.log"), "ab")
                STRONG["process"] = subprocess.Popen(
                    [STRONG["python"], "-I", os.path.join(HERE, "fill_worker.py"), "--port", str(STRONG["port"])],
                    stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            time.sleep(1)
    raise ValueError("the big model did not start: see fill-worker.log in the output folder")


def strong(way, cover, start=None, seed=0):
    """Have the helper draw the selection: way "tidy" draws a first guess clean, "guided" draws the spot anew."""
    helper()
    parts = {"pixels": current(), "cover": np.rint(np.clip(cover, 0, 1) * 255).astype(np.uint8)}
    if start is not None:
        parts["start"] = retouch.to_bytes(start)
    held = io.BytesIO()
    np.savez(held, **parts)
    query = "way=%s&seed=%d&grow=%d" % (way, seed, max(2, int(round(2 * scale()))))
    request = urllib.request.Request("http://127.0.0.1:%d/fill?%s" % (STRONG["port"], query), data=held.getvalue(), method="POST")
    try:
        answer = urllib.request.urlopen(request, timeout=1200).read()
    except urllib.error.HTTPError as problem:
        raise ValueError("the big model: " + problem.read().decode("utf-8", "replace")[:300])
    return np.load(io.BytesIO(answer)).astype(np.float32)


def fill_device():
    """The fast network runs on the processor (two or three seconds a spot) and leaves the video card to the big
    model and to whatever else the person runs. FILL_GPU=1 sends it to the card when the card has room."""
    if os.environ.get("FILL_GPU") == "1":
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2.5e9:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def warm_up():
    """Read the redrawing network ahead of the first use: it takes some seconds."""
    try:
        import neural_fill
        if neural_fill.available():
            neural_fill.network(fill_device())
    except Exception as problem:      # the page works without it
        print("the redrawing network is not ready: %s" % problem, flush=True)


# ------------------------------------------------------------------ the folder button

def inside_folder(relative):
    base = os.path.realpath(PLACES["dir"])
    path = os.path.realpath(os.path.join(base, relative or ""))
    if path != base and not path.startswith(base + os.sep):
        raise ValueError("that place is outside the folder the page was started with")
    return path


def folder(relative):
    path = inside_folder(relative)
    if not os.path.isdir(path):
        raise ValueError("no such folder")
    base = os.path.realpath(PLACES["dir"])
    folders, files = [], []
    for name in sorted(os.listdir(path), key=str.lower):
        if name.startswith("."):
            continue
        whole = os.path.join(path, name)
        if os.path.isdir(whole):
            folders.append(name)
        elif name.lower().endswith(PICTURES):
            files.append(name)
    here = os.path.relpath(path, base).replace("\\", "/")
    return {"path": "" if here == "." else here, "top": os.path.basename(base), "folders": folders, "files": files}


def thumb(relative):
    path = inside_folder(relative)
    if not path.lower().endswith(PICTURES):
        raise ValueError("not a picture")
    key = (path, os.path.getmtime(path))
    if key not in THUMBS:
        if len(THUMBS) > 400:
            THUMBS.clear()
        picture = Image.open(path)
        picture.draft("RGB", (192, 192))
        picture = picture.convert("RGBA")
        picture.thumbnail((144, 144), Image.LANCZOS)
        paper = Image.new("RGBA", picture.size, (238, 230, 212, 255))
        paper.alpha_composite(picture)
        held = io.BytesIO()
        paper.convert("RGB").save(held, "JPEG", quality=82)
        THUMBS[key] = held.getvalue()
    return THUMBS[key]


# ------------------------------------------------------------------ the steps

def clean(painted):
    add_step("clean", "clean", deripple.clean(current(), 2.0 if painted else 5.0, 3.0 if painted else 5.5))
    return {}


def selection(raw):
    """The selection comes as a picture of the same size: what is selected is opaque."""
    picture = Image.open(io.BytesIO(raw))
    cover = np.asarray(picture.getchannel("A") if "A" in picture.getbands() else picture.convert("L"), np.float32) / 255.0
    if cover.shape != current().shape[:2]:
        raise ValueError("the selection is not of the picture's size: clear it and select again")
    return cover


LABELS = {name: name for name in ("shapes", "lines", "nothing", "draw", "ai", "tidy", "guided")}      # the page words a step by its key


def paint(raw, query):
    keep = query.get("keep", ["shapes"])[0]
    if keep not in LABELS:
        raise ValueError("no such way")
    cover = selection(raw)
    if not (cover >= 0.5).any():
        raise ValueError("nothing is selected")
    source = (float(query["sx"][0]), float(query["sy"][0])) if "sx" in query else None
    if keep in ("ai", "tidy", "guided"):
        import neural_fill
        seed = int(query.get("seed", ["0"])[0])
        before = current().astype(np.float32)
        if keep == "guided":
            if not (strong_ready() and guide_ready()):
                raise ValueError("the big model with its guide is not here: choose “Fast”")
            drawn = strong("guided", cover, None, seed)
        else:
            if not neural_fill.available():
                raise ValueError("the network's file is not in models/lama: choose “Plain fill”")
            # a larger picture has thicker lines already, so its piece is enlarged less for the network
            drawn = neural_fill.inpaint(before, cover, fill_device(), grow=max(2, int(round(2 * scale()))),
                                        enlarge=max(1.0, neural_fill.ENLARGE / scale()))[0]
            if keep == "tidy":
                if not strong_ready():
                    raise ValueError("the big model is not here: choose “Fast”")
                drawn = strong("tidy", cover, drawn, seed)
        # the plain fills of the answer are drawn with the picture's own colours: no shift, no grain
        out, changed = retouch.snap(before, cover, drawn, scale())
        colour = None
        if changed:
            add_step(LABELS[keep], "paint", retouch.to_bytes(out))
    elif keep == "draw":
        out, changed, colour = retouch.draw(current(), cover, source, scale())
        solid = WORK["steps"][-1]["alpha"]
        if changed and solid is not None:      # what is drawn outside the figure has to be seen
            solid = np.maximum(solid, np.rint(np.clip(cover, 0, 1) * 255).astype(np.uint8))
        if changed:
            add_step(LABELS[keep], "paint", retouch.to_bytes(out), solid)
    else:
        out, changed, colour = retouch.paint_over(current(), cover, source, keep, scale())
        if changed:
            add_step(LABELS[keep], "paint", retouch.to_bytes(out))
    return {"changed": changed, "colour": colour}


def wand(body):
    points = [(float(x), float(y)) for x, y in body["points"]]
    taken = retouch.pick(current(), points, float(body.get("sense", 10.0)), None, scale())
    mask = np.zeros(taken.shape + (4,), np.uint8)
    mask[taken] = 255
    return encode(Image.fromarray(mask)), int(taken.sum())


def probe(query):
    colour, width = retouch.probe(current(), (float(query["x"][0]), float(query["y"][0])), scale())
    return {"colour": colour, "width": width}


def restore(raw, query):
    """Put back a step saved earlier: the page uses it to carry a picture with its steps over a
    restart. The first one, of kind "start", begins the work anew."""
    kind = query.get("kind", ["paint"])[0]
    if kind not in ("start", "clean", "paint", "enlarge"):
        raise ValueError("no such kind of step")
    pixels, solid = split(Image.open(io.BytesIO(raw)))
    if kind == "start":
        WORK["name"] = query.get("name", ["picture"])[0]
        WORK["steps"] = []
    elif not WORK["steps"]:
        raise ValueError("the first step must be the picture as it came")
    label = query.get("label", ["edit"])[0]
    add_step(label, kind, pixels, solid)
    return {}


def enlarge(painted):
    if not os.path.exists(model_file()):
        raise ValueError("the model's file is not in models/realesrgan")
    import neural_clean
    import torch
    before = current()
    # on the video card when there is one that torch can use; on the processor it works too, several times slower
    device = "cuda" if torch.cuda.is_available() else "cpu"
    twice = neural_clean.double(before, device, painted)      # a flat picture keeps its colours and its freckles
    solid = WORK["steps"][-1]["alpha"]
    if solid is not None:
        solid = np.asarray(Image.fromarray(solid).resize((before.shape[1] * 2, before.shape[0] * 2), Image.BICUBIC))
    add_step("enlarge", "enlarge", deripple.clean(twice, 2.0, 3.0), solid)
    return {}


def numbers(identity):
    """Mending a spot does not change the numbers of the whole picture: such a step shows the numbers
    of the step it was made from."""
    steps = WORK["steps"] + WORK["undone"]
    index = next(index for index, step in enumerate(steps) if step["id"] == identity)
    while index and steps[index]["kind"] == "paint":
        index -= 1
    step = steps[index]
    with COUNTING:
        if step["numbers"] is None:
            found = ripple_check.measure(step["pixels"])
            step["numbers"] = {"echo": found["echo_worst"], "blotches": found["blotches_near"], "grain": found["grain"]}
    return step["numbers"]


def save():
    os.makedirs(PLACES["out"], exist_ok=True)
    path = os.path.join(PLACES["out"], WORK["name"] + "-clean.png")
    number = 2
    while os.path.exists(path):
        path = os.path.join(PLACES["out"], "%s-clean-%d.png" % (WORK["name"], number))
        number += 1
    whole(WORK["steps"][-1]).save(path)
    return {"path": path.replace("\\", "/")}


class Local(ThreadingHTTPServer):
    # Windows lets a second program take a port that is in use and then answers from either of them:
    # there a taken port must be refused. Elsewhere reuse is what lets the page be started again at once.
    allow_reuse_address = os.name != "nt"


class Page(BaseHTTPRequestHandler):
    server_version = "AIImageRehab"

    def log_message(self, *args):
        pass

    def send(self, code, body, kind="application/json; charset=utf-8", extra=()):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def local(self):
        # only this computer's own page may talk to the server
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self.local():
            return self.send(403, {"error": "not local"})
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                with open(os.path.join(HERE, "ripple_studio.html"), "rb") as file:
                    return self.send(200, file.read(), "text/html; charset=utf-8")
            if url.path.startswith("/assets/") and url.path[8:] in ASSETS:
                with open(os.path.join(HERE, "studio", url.path[8:]), "rb") as file:
                    return self.send(200, file.read(), ASSETS[url.path[8:]])
            if url.path == "/api/state":
                return self.send(200, state())
            if url.path == "/api/probe":
                refresh()
                return self.send(200, probe(query))
            if url.path == "/api/folder":
                return self.send(200, folder(query.get("path", [""])[0]))
            if url.path == "/api/thumb":
                return self.send(200, thumb(query["path"][0]), "image/jpeg")
            if url.path == "/api/image":
                step = step_by(int(query["id"][0]))
                if "download" in query:      # the file to keep is packed tighter than the one for the screen
                    name = WORK["name"] + "-clean.png"
                    extra = (("Content-Disposition", "attachment; filename=\"picture-clean.png\"; filename*=UTF-8''" + quote(name)),)
                    return self.send(200, encode(whole(step), 6), "image/png", extra)
                if step["png"] is None:
                    step["png"] = encode(whole(step))
                return self.send(200, step["png"], "image/png")
            if url.path == "/api/numbers":
                return self.send(200, numbers(int(query["id"][0])))
            return self.send(404, {"error": "no such page"})
        except (IndexError, KeyError, ValueError, StopIteration, OSError) as problem:
            return self.send(400, {"error": str(problem)})
        except Exception as problem:      # an unforeseen one: the page shows it instead of waiting for ever
            return self.send(500, {"error": "%s: %s" % (type(problem).__name__, problem)})

    def do_POST(self):
        if not self.local() or self.headers.get("X-Studio") != "1":
            return self.send(403, {"error": "not local"})
        url = urlparse(self.path)
        query = parse_qs(url.query)
        length = int(self.headers.get("Content-Length") or 0)
        if length > BIGGEST:
            return self.send(413, {"error": "the file is larger than 60 MB"})
        raw = self.rfile.read(length) if length else b""
        if not BUSY.acquire(blocking=False):
            return self.send(409, {"error": "the previous step is still being computed, wait a moment"})
        started = time.time()
        try:
            refresh()
            if url.path == "/api/upload":
                start(query.get("name", ["picture"])[0], Image.open(io.BytesIO(raw)))
                done = {}
            elif url.path == "/api/open":
                path = inside_folder(query["path"][0])
                if not path.lower().endswith(PICTURES):
                    raise ValueError("not a picture")
                start(path, Image.open(path))
                done = {}
            elif url.path == "/api/restore":
                done = restore(raw, query)
            elif not WORK["steps"]:
                raise ValueError("open a picture first")
            elif url.path == "/api/wand":
                mask, count = wand(json.loads(raw))
                return self.send(200, mask, "image/png", (("X-Picked", str(count)),))
            elif url.path == "/api/clean":
                done = clean(bool(json.loads(raw or b"{}").get("painted")))
            elif url.path == "/api/paint":
                done = paint(raw, query)
            elif url.path == "/api/enlarge":
                done = enlarge(bool(json.loads(raw or b"{}").get("painted")))
            elif url.path == "/api/undo":
                if len(WORK["steps"]) > 1:
                    WORK["undone"].append(WORK["steps"].pop())
                done = {}
            elif url.path == "/api/redo":
                if WORK["undone"]:
                    WORK["steps"].append(WORK["undone"].pop())
                done = {}
            elif url.path == "/api/save":
                done = save()
            else:
                return self.send(404, {"error": "no such action"})
            if url.path != "/api/save":
                persist()
            done.update(state(), seconds=round(time.time() - started, 1))
            return self.send(200, done)
        except (IndexError, KeyError, ValueError, OSError) as problem:
            return self.send(400, {"error": str(problem)})
        except Exception as problem:      # an unforeseen one (no memory on the video card, a library that is not installed)
            return self.send(500, {"error": "%s: %s" % (type(problem).__name__, problem)})
        finally:
            BUSY.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=4390)
    parser.add_argument("--dir", default=PLACES["dir"])
    parser.add_argument("--out", default=PLACES["out"])
    parser.add_argument("--open", action="store_true")
    options = parser.parse_args()
    PLACES["dir"], PLACES["out"] = os.path.abspath(options.dir), os.path.abspath(options.out)
    address = "http://127.0.0.1:%d" % options.port
    try:
        server = Local(("127.0.0.1", options.port), Page)
    except OSError:
        if options.open:
            webbrowser.open(address)
        sys.exit("port %d is taken: the page may be running already, open %s (or start another with --port N)" % (options.port, address))
    revive()
    threading.Thread(target=warm_up, daemon=True).start()
    print("AI Image Rehab: %s" % address, flush=True)
    if options.open:
        threading.Timer(0.5, webbrowser.open, [address]).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
