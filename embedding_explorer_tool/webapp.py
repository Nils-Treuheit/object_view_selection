"""HTTP server for the embedding explorer web app.

Serves the single-page frontend (``webapp_template.html``), the vendored
plotly bundle, the raw frame/mask images, the composited mask-overlay frames,
and the ``/api/run`` endpoint that recomputes k-means + constrained xNN on
demand.

Run from the repository root with::

    python -m embedding_explorer_tool.webapp [--output_dir ...] [--data_root ...]
"""

import argparse
import json
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    from . import algorithms, webapp_plotting as plotting
except ImportError:
    import algorithms
    import webapp_plotting as plotting

TOOL_DIR = Path(__file__).parent
TEMPLATE_PATH = TOOL_DIR / "webapp_template.html"
PLOTLY_PATH = TOOL_DIR / "static" / "plotly.min.js"

DEFAULT_OUTPUT_DIR = "outputs_embedding_explorer"
DEFAULT_DATA_ROOT = "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/09_triprong_old"


class ExplorerState:
    """Loads the snapshot once and serves computation + images on demand.

    If ``output_dir`` has no snapshot yet, it is generated first (pre-filter +
    quality scoring + embedding, same as ``run.py``) so the app can point at a
    fresh ``--data_root`` and still start.
    """

    def __init__(self, output_dir, data_root, embedding=algorithms.DEFAULT_EMBEDDING,
                 embedding_model=algorithms.DEFAULT_EMBEDDING_MODEL, auto_thresholds=False):
        self.output_dir = Path(output_dir)
        if not algorithms.snapshot_exists(self.output_dir):
            if not data_root:
                raise FileNotFoundError(
                    f"No snapshot in {self.output_dir} and no --data_root given to generate one. "
                    "Pass --data_root (dataset root with images/ and masks/) or point --output_dir "
                    "at an existing pipeline output."
                )
            print(f"No snapshot in {self.output_dir}; generating embeddings first ...")
            algorithms.generate_snapshot(
                self.output_dir,
                data_root,
                embedding=embedding,
                embedding_model=embedding_model,
                auto_thresholds=auto_thresholds,
            )
        snapshot = algorithms.load_snapshot(self.output_dir)
        self.data_root = Path(data_root) if data_root else None
        self._load_snapshot()

        self._mds_lock = threading.Lock()
        self._mds_coords = None
        self._composite_cache = {}

    def _load_snapshot(self):
        """Read the snapshot files + image/mask maps (data_root-resolving)."""
        snapshot = algorithms.load_snapshot(self.output_dir)
        self.embeddings = snapshot["embeddings"]
        self.pool_ids = snapshot["pool_ids"]
        self.quality = snapshot["quality"]
        self.selected_ids = snapshot.get("selected_ids")

        root = self.data_root or Path(snapshot.get("data_root") or "")
        if not (root / "images").is_dir():
            raise FileNotFoundError(
                f"Dataset root {root} has no images/ directory. Pass --data_root."
            )
        self.data_root = root
        self.images = {int(p.stem): p for p in (root / "images").glob("*.png")}
        self.masks = {int(p.stem): p for p in (root / "masks").glob("*.png")}

    def reload(self):
        """Re-read the snapshot + image maps after the tuner regenerated them.

        Invalidates the cached MDS projection and composited frames so the next
        ``/api/run`` reflects the new pool.
        """
        with self._mds_lock:
            self._load_snapshot()
            self._mds_coords = None
            self._composite_cache = {}
        return {"pool": len(self.pool_ids), "data_root": str(self.data_root)}

    def get_coords(self):
        with self._mds_lock:
            if self._mds_coords is None:
                self._mds_coords = algorithms.project_mds(self.embeddings)
            return self._mds_coords

    def api_run(self, k, init, x, dims="2d"):
        k = int(k)
        x = int(x)
        coords = self.get_coords()
        result = algorithms.run_kmeans_xnn(self.embeddings, self.quality, k, init, x)
        fig = plotting.build_figure(
            coords, self.quality, result["labels"], result, self.pool_ids, dims=dims
        )
        return {
            "figure": json.loads(fig.to_json()),
            "text": algorithms.build_text(result, self.pool_ids, self.quality),
            "k": result["k"],
            "init": result["init"],
            "x": result["x"],
            "dims": dims,
            "picks": [int(self.pool_ids[i]) for i in result["picks"]],
            "centroid_ids": [int(self.pool_ids[c["medoid"]]) for c in result["clusters"]],
            "xnn": {
                str(int(self.pool_ids[c["medoid"]])): [
                    int(self.pool_ids[p]) for p in c["candidates"]
                ]
                for c in result["clusters"]
            },
            "selected_ids": self.selected_ids,
        }

    def image_bytes(self, frame_id):
        path = self.images.get(frame_id)
        return path.read_bytes() if path else None

    def mask_bytes(self, frame_id):
        path = self.masks.get(frame_id)
        return path.read_bytes() if path else None

    def composite_bytes(self, frame_id):
        cached = self._composite_cache.get(frame_id)
        if cached is not None:
            return cached
        img_path = self.images.get(frame_id)
        mask_path = self.masks.get(frame_id)
        if img_path is None or mask_path is None:
            return None
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        out = algorithms.compose_mask_overlay(image, mask)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        blob = buf.tobytes()
        self._composite_cache[frame_id] = blob
        return blob


class Handler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                return self._serve_index()
            if path == "/static/plotly.min.js":
                return self._serve_file(PLOTLY_PATH, "application/javascript")
            if path == "/api/reload":
                payload = self.state.reload()
                body = json.dumps(payload).encode("utf-8")
                return self._send(200, body, "application/json")
            if path.startswith("/api/run"):
                return self._serve_api(parsed.query)
            if path.startswith("/composite/"):
                fid = int(path[len("/composite/") :].split(".")[0])
                return self._serve_bytes(self.state.composite_bytes(fid), "image/png")
            if path.startswith("/image/"):
                fid = int(path[len("/image/") :].split(".")[0])
                return self._serve_bytes(self.state.image_bytes(fid), "image/png")
            if path.startswith("/mask/"):
                fid = int(path[len("/mask/") :].split(".")[0])
                return self._serve_bytes(self.state.mask_bytes(fid), "image/png")
            return self._send(404, b"not found", "text/plain")
        except (ValueError, KeyError):
            return self._send(404, b"bad request", "text/plain")

    def _serve_index(self):
        html = TEMPLATE_PATH.read_text()
        return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_file(self, path, ctype):
        if not path.exists():
            return self._send(404, b"not found", "text/plain")
        return self._send(200, path.read_bytes(), ctype)

    def _serve_api(self, query):
        params = urllib.parse.parse_qs(query)
        k = params.get("k", ["8"])[0]
        init = params.get("init", ["best_quality"])[0]
        x = params.get("x", ["10"])[0]
        dims = params.get("dims", ["2d"])[0]
        if dims not in ("2d", "3d"):
            dims = "3d"
        payload = self.state.api_run(k, init, x, dims)
        body = json.dumps(payload).encode("utf-8")
        return self._send(200, body, "application/json")

    def _serve_bytes(self, blob, ctype):
        if blob is None:
            return self._send(404, b"no data for frame", "text/plain")
        return self._send(200, blob, ctype)

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Threading server that stays quiet about dropped client connections."""

    def handle_error(self, request, client_address):
        exc_type, exc, _tb = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def run_server(state, port, open_browser=True):
    Handler.state = state
    server = QuietThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Embedding Explorer: {url}")
    print(f"  snapshot : {state.output_dir}")
    print(f"  data root: {state.data_root}")
    print(f"  pool     : {len(state.pool_ids)} samples, selected "
          f"{state.selected_ids if state.selected_ids is not None else 'n/a'}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding explorer web app (kMeans + constrained xNN)"
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Pipeline output dir with embeddings.npy / quality.csv; "
                             "created and populated from --data_root when it has no snapshot")
    parser.add_argument("--data_root", type=str, default="",
                        help="Dataset root with images/ and masks/ "
                             "(default: from report.json)")
    parser.add_argument("--embedding", type=str, default=algorithms.DEFAULT_EMBEDDING,
                        choices=algorithms.EMBEDDING_CHOICES,
                        help="Embedding type used when generating a fresh snapshot "
                             "(auto=infer from --embedding_model); unused when a snapshot exists")
    parser.add_argument("--embedding_model", type=str, default=algorithms.DEFAULT_EMBEDDING_MODEL,
                        help="Model name or path for embedding generation "
                             "(auto=infer from --embedding_model); unused when a snapshot exists")
    parser.add_argument("--port", type=int, default=8510)
    parser.add_argument("--no-browser", action="store_true", dest="no_browser")
    args = parser.parse_args()

    state = ExplorerState(
        args.output_dir,
        args.data_root,
        embedding=args.embedding,
        embedding_model=args.embedding_model,
    )
    run_server(state, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
