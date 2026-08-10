"""Second webapp: interactive pre-filter threshold tuner (separate port).

The main embedding explorer (``webapp.py``) assumes a snapshot already
exists. This app is the step before that: it loads a dataset once, runs the
real pre-filter pipeline with manually-tuned thresholds and shows the
accept/reject outcome in a text panel, and its "Run Embedding" button
generates the snapshot the explorer page displays.

Layout (see ``prefilter_template.html``):

    | Garbage Thresholds   |  PRE-FILTER RUN ... text panel   |
    |  - knob              |                                   |
    |  - knob              |  [Run Embedding]                  |
    | Outlier Thresholds   |                                   |
    |  - knob              |                                   |

Threshold knobs are grouped into "garbage thresholds" (absolute floors /
ceilings that reject a sample regardless of the population) and "outlier
thresholds" (robust z-score cutoffs relative to the population).  Knob
values are written straight into the ``PipelineConfig`` filter configs, so
everything shown here is exactly what ``run.py`` would do.
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

from config import PipelineConfig
from data_io.dataset import Dataset

DEFAULT_DATA_ROOT = "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/elephant"
DEFAULT_OUTPUT_DIR = "outputs_embedding_explorer"
DEFAULT_PORT = 8520

# (config key, label, min, max, step) for the absolute garbage thresholds.
GARBAGE_KNOBS = [
    ("blur_laplacian.hard_min_variance", "blur_laplacian · boundary Laplacian floor", 0.0, 50000.0, 100.0),
    ("blur_tenengrad.hard_min_tenengrad", "blur_tenengrad · boundary Tenengrad floor", 0.0, 1000.0, 1.0),
    ("vincents_artefacts.hard_max_fraction", "vincents_artefacts · artifact fraction ceiling", 0.0, 1.0, 0.005),
    ("vincents_motion_blur.hard_min_variance", "vincents_motion_blur · boundary Laplacian floor", 0.0, 50000.0, 10.0),
    ("vincents_area.hard_min_area_fraction", "vincents_area · foreground fraction floor", 0.0, 1.0, 0.005),
]

# (config key, label, min, max, step) for the population outlier z-cutoffs.
OUTLIER_KNOBS = [
    ("blur_laplacian.outlier_z", "blur_laplacian · population outlier cutoff", 0.5, 10.0, 0.25),
    ("blur_tenengrad.outlier_z", "blur_tenengrad · population outlier cutoff", 0.5, 10.0, 0.25),
    ("vincents_artefacts.outlier_z", "vincents_artefacts · population outlier cutoff", 0.5, 10.0, 0.25),
    ("vincents_motion_blur.outlier_z", "vincents_motion_blur · population outlier cutoff", 0.5, 10.0, 0.25),
    ("vincents_area.outlier_z", "vincents_area · population outlier cutoff", 0.5, 10.0, 0.25),
]

# (metrics attr, label) shown as raw stats of the accepted set in the report.
REPORT_STATS = [
    ("laplacian", "blur_laplacian stat"),
    ("tenengrad", "blur_tenengrad stat"),
    ("vincent_artifact_fraction", "vincents_artefacts stat"),
    ("vincent_boundary_blur_variance", "vincents_motion_blur stat"),
    ("vincent_area_fraction", "vincents_area stat"),
]

TEMPLATE_NAME = "prefilter_template.html"


def _conf(cfg, key):
    conf_name, attr = key.split(".", 1)
    conf = getattr(cfg.filters, conf_name, None)
    if conf is None or not hasattr(conf, attr):
        return None, None
    return conf, attr


def apply_knobs(cfg: PipelineConfig, garbage=None, outlier=None):
    """Write the tuning knobs into ``cfg`` (in place) and return it.

    ``garbage`` maps config keys to plain float values; ``outlier`` maps
    config keys to ``{"enabled": bool, "value": float}`` — a disabled outlier
    knob sets the z-cutoff to ``None`` (population outlier rejection off for
    that filter).  Unknown keys are ignored.
    """
    for key, value in (garbage or {}).items():
        conf, attr = _conf(cfg, key)
        if conf is None:
            continue
        setattr(conf, attr, float(value))
    for key, meta in (outlier or {}).items():
        conf, attr = _conf(cfg, key)
        if conf is None:
            continue
        if meta.get("enabled"):
            setattr(conf, attr, float(meta.get("value", 3.0)))
        else:
            setattr(conf, attr, None)
    return cfg


def config_payload(cfg: PipelineConfig):
    """Serialize the current knob values (with bounds) for the frontend."""
    garbage = [
        {"key": key, "label": label, "value": getattr(*_conf(cfg, key)),
         "min": lo, "max": hi, "step": step}
        for key, label, lo, hi, step in GARBAGE_KNOBS
    ]
    outlier = []
    for key, label, lo, hi, step in OUTLIER_KNOBS:
        value = getattr(*_conf(cfg, key))
        outlier.append({
            "key": key, "label": label,
            "value": value if value is not None else 3.0,
            "enabled": value is not None,
            "min": lo, "max": hi, "step": step,
        })
    return {"garbage": garbage, "outlier": outlier}


def _stats_line(attr, observations):
    values = sorted(float(getattr(obs.metrics, attr, 0.0)) for obs in observations)
    if not values:
        return "n/a"
    lo, mid, hi = values[0], median(values), values[-1]
    return f"{lo:.4f} / {mid:.4f} / {hi:.4f}"


def build_report_text(data_root, accepted, rejected, cfg, garbage=None, outlier=None):
    """Render the pre-filter outcome as plain text for the output panel."""
    lines = [
        "PRE-FILTER RUN",
        "=" * 72,
        f"data_root  : {data_root}",
        f"observations: {len(accepted) + len(rejected)}   accepted: {len(accepted)}   rejected: {len(rejected)}",
        "",
        "Garbage thresholds applied:",
    ]
    for key, _label, _lo, _hi, _step in GARBAGE_KNOBS:
        value = getattr(*_conf(cfg, key))
        lines.append(f"  {key:<45} {value:.4f}")
    lines.append("")
    lines.append("Outlier thresholds applied:")
    for key, _label, _lo, _hi, _step in OUTLIER_KNOBS:
        value = getattr(*_conf(cfg, key))
        lines.append(f"  {key:<45} {value:.4f}" if value is not None
                     else f"  {key:<45} off")
    lines.append("")
    lines.append("Rejected by filter:")
    reasons = {}
    for obs in rejected:
        reasons[obs.rejection_reason or "unknown"] = reasons.get(obs.rejection_reason or "unknown", 0) + 1
    if reasons:
        for reason, count in sorted(reasons.items()):
            lines.append(f"  {reason:<45} {count}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Accepted raw stats (min / median / max):")
    for attr, label in REPORT_STATS:
        lines.append(f"  {label:<45} {_stats_line(attr, accepted)}")
    return "\n".join(lines)


def run_prefilter(data_root, garbage=None, outlier=None):
    """Build the dataset + pipeline, run the pre-filter and return a report.

    Returns ``(text, accepted_ids, rejected_ids, reasons)``.
    """
    dataset = Dataset(data_root)
    dataset.load_images()
    return _run_prefilter(data_root, dataset.observations, garbage, outlier)


def _run_prefilter(data_root, observations, garbage=None, outlier=None):
    """Run the pre-filter pipeline over a prepared observation list."""
    from run import apply_soft_filters, build_filters, build_soft_filters

    cfg = apply_knobs(PipelineConfig(data_root=data_root), garbage, outlier)

    filter_pipeline = build_filters(cfg)
    soft_filters = build_soft_filters(cfg)

    for obs in observations:
        obs.rejected = False
        obs.rejection_reason = None

    if filter_pipeline.need_fitting:
        filter_pipeline.fit_observations(observations)

    accepted = []
    rejected = []
    for obs in observations:
        if not filter_pipeline.run(obs):
            rejected.append(obs)
            continue
        accepted.append(obs)

    apply_soft_filters(soft_filters, accepted, rejected)

    text = build_report_text(data_root, accepted, rejected, cfg, garbage, outlier)
    reasons = {}
    for obs in rejected:
        reasons[obs.rejection_reason or "unknown"] = \
            reasons.get(obs.rejection_reason or "unknown", 0) + 1
    return text, [obs.id for obs in accepted], [obs.id for obs in rejected], reasons


def auto_knobs(data_root):
    """Compute the data-driven ("auto") threshold knobs for a dataset.

    Uses ``utils.threshold_tuner.tune_thresholds`` — the same percentile
    floors run.py applies with auto thresholds on.  The tuned Laplacian /
    Tenengrad floors map onto the modern blur garbage knobs; every other
    knob keeps its config default (the tuner emits no ceiling / z-cutoffs).
    """
    dataset = Dataset(data_root)
    dataset.load_images()
    return _auto_knobs(dataset.observations)


def _auto_knobs(observations):
    from utils.threshold_tuner import tune_thresholds

    tuned = tune_thresholds(observations)

    garbage = {
        "blur_laplacian.hard_min_variance": tuned["laplacian_threshold"],
        "blur_tenengrad.hard_min_tenengrad": tuned["tenengrad_threshold"],
    }
    return garbage, {}


def run_prefilter_auto(data_root):
    """Run the pre-filter with auto-tuned thresholds.

    Returns ``(text, accepted_ids, rejected_ids, reasons, garbage, outlier)``
    so the frontend can show the tuned values that were applied.
    """
    garbage, outlier = auto_knobs(data_root)
    text, accepted, rejected, reasons = run_prefilter(data_root, garbage, outlier)
    return text, accepted, rejected, reasons, garbage, outlier


def run_embedding(data_root, output_dir, garbage=None, outlier=None):
    """Run the full snapshot generation (pre-filter + quality + embedding).

    Writes the snapshot into ``output_dir`` — the same directory the
    embedding explorer reads — so the explorer picks up the newly-tuned
    selection pool after a reload.
    """
    from embedding_explorer_tool.algorithms import generate_snapshot

    def cfg_override(cfg):
        cfg.auto_thresholds = False
        return apply_knobs(cfg, garbage, outlier)

    generate_snapshot(output_dir, data_root, auto_thresholds=False, cfg_override=cfg_override)
    return (f"Embedding snapshot written to {output_dir} "
            f"(data_root={data_root}). Click 'Reload Snapshot' in the "
            f"embedding explorer (default port 8510) to see the new pool.")


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Threading server that stays quiet about dropped client connections."""

    def handle_error(self, request, client_address):
        exc_type, exc, _tb = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class PrefilterApp:
    def __init__(self, data_root=DEFAULT_DATA_ROOT, output_dir=DEFAULT_OUTPUT_DIR):
        self.data_root = data_root
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.cfg = PipelineConfig(data_root=data_root)
        self.dataset = Dataset(data_root)
        self.dataset.load_images()

    def observation_count(self):
        return len(self.dataset.observations)

    def run(self, garbage=None, outlier=None):
        return _run_prefilter(self.data_root, self.dataset.observations,
                              garbage, outlier)

    def run_auto(self):
        garbage, outlier = _auto_knobs(self.dataset.observations)
        text, accepted, rejected, reasons = \
            _run_prefilter(self.data_root, self.dataset.observations,
                           garbage, outlier)
        return text, accepted, rejected, reasons, garbage, outlier


def make_handler(app: PrefilterApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[prefilter] {self.address_string()} {fmt % args}")

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    return {}
                return json.loads(self.rfile.read(length).decode())
            except (BrokenPipeError, ConnectionResetError, ValueError):
                return {}

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                template = Path(__file__).parent / TEMPLATE_NAME
                body = template.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/api/config":
                self._send_json({
                    "data_root": app.data_root,
                    "output_dir": app.output_dir,
                    "n_observations": app.observation_count(),
                    "knobs": config_payload(app.cfg),
                })
                return
            if path == "/static/plotly.min.js":
                static = Path(__file__).parent / "static" / "plotly.min.js"
                if not static.exists():
                    self.send_error(404)
                    return
                body = static.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/run":
                payload = self._read_body()
                with app.lock:
                    text, accepted_ids, rejected_ids, reasons = \
                        app.run(payload.get("garbage"), payload.get("outlier"))
                self._send_json({
                    "text": text,
                    "accepted": len(accepted_ids),
                    "rejected": len(rejected_ids),
                    "reasons": reasons,
                })
                return
            if path == "/api/run_auto":
                with app.lock:
                    text, accepted_ids, rejected_ids, reasons, garbage, outlier = \
                        app.run_auto()
                self._send_json({
                    "text": text,
                    "accepted": len(accepted_ids),
                    "rejected": len(rejected_ids),
                    "reasons": reasons,
                    "garbage": garbage,
                    "outlier": outlier,
                })
                return
            if path == "/api/embed":
                payload = self._read_body()
                with app.lock:
                    text = run_embedding(app.data_root, app.output_dir,
                                         payload.get("garbage"), payload.get("outlier"))
                self._send_json({"text": text})
                return
            self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Pre-filter threshold tuner webapp")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help=f"dataset root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="snapshot dir the embedding explorer reads (default: "
                             f"{DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    app = PrefilterApp(data_root=args.data_root, output_dir=args.output_dir)
    handler = make_handler(app)
    server = QuietThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Pre-filter tuner on http://localhost:{args.port}/")
    print(f"  data_root: {app.data_root}")
    print(f"  snapshot output (embedding explorer): {app.output_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
