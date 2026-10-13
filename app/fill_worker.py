"""A helper process for the page of ripple_studio.py: it keeps a diffusion model loaded and draws
spots with it on request.

    PYTHON -I app/fill_worker.py [--port 4392] [--idle 600]

The page itself runs under the plain Python, which has no diffusers. The model needs an interpreter
that has torch with CUDA, diffusers and transformers (for instance the one of a Forge installation), so
the page starts this helper with that interpreter the first time the strong model is asked for.

It listens on this computer only. A request is POST /fill?way=tidy|guided&strength=&seed=&grow= with
a body made by numpy.savez: pixels (the picture, RGB), cover (the selection, 0..255) and, for the
way "tidy", start (a first guess to draw clean). The answer is the new picture, numpy.save.

After --idle seconds without a request the helper ends by itself and gives the video card back.
"""
import argparse
import io
import os
import site
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.append(site.getusersitepackages())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import diffusion_fill  # noqa: E402

LAST = {"time": time.time(), "idle": 600}
ONE = threading.Lock()


class Helper(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, code, body, kind="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        guided = os.path.exists(diffusion_fill.GUIDE)
        self.answer(200, ('{"ready": true, "guided": %s}' % ("true" if guided else "false")).encode(), "application/json")

    def do_POST(self):
        if (self.headers.get("Host") or "").split(":")[0] not in ("127.0.0.1", "localhost"):
            return self.answer(403, b"not local", "text/plain")
        query = parse_qs(urlparse(self.path).query)
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        LAST["time"] = time.time()
        try:
            given = np.load(io.BytesIO(raw))
            way = query.get("way", ["tidy"])[0]
            with ONE:
                out, count = diffusion_fill.inpaint(
                    given["pixels"], given["cover"], given["start"] if "start" in given.files and way == "tidy" else None,
                    strength=float(query.get("strength", ["1.0" if way == "guided" else "0.4"])[0]),
                    seed=int(query.get("seed", ["0"])[0]), grow=int(query.get("grow", ["2"])[0]), guide=way == "guided")
            held = io.BytesIO()
            np.save(held, np.clip(np.rint(out), 0, 255).astype(np.uint8))
            LAST["time"] = time.time()
            return self.answer(200, held.getvalue())
        except Exception as problem:      # the page shows the reason to the person
            return self.answer(500, ("%s: %s" % (type(problem).__name__, problem)).encode("utf-8"), "text/plain; charset=utf-8")


def watch(server):
    while True:
        time.sleep(15)
        if time.time() - LAST["time"] > LAST["idle"] and not ONE.locked():
            server.shutdown()
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=4392)
    parser.add_argument("--idle", type=int, default=600)
    options = parser.parse_args()
    LAST["idle"] = options.idle
    server = ThreadingHTTPServer(("127.0.0.1", options.port), Helper)
    threading.Thread(target=watch, args=(server,), daemon=True).start()
    print("fill worker: http://127.0.0.1:%d" % options.port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
