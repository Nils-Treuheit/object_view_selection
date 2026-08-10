"""Second webapp: interactive pre-filter threshold tuner (separate port).

The main embedding explorer (``webapp.py``) assumes a snapshot already
exists. This app is the step before that: it loads a dataset once, runs the
real pre-filter pipeline with manually-tuned thresholds and shows the
accept/reject outcome in a text panel, and its "Run Embedding" button
generates the snapshot the explorer page displays.

Layout (see ``prefilter_template.html``):

    | Filter Parameters     |  PRE-FILTER RUN ... text panel   |
    |  - kernel_size ...    |                                   |
    | Garbage Thresholds    |  [Run Embedding]                  |
    |  - knob               |                                   |
    | Outlier Thresholds    |                                   |
    |  - knob               |                                   |

By default the webapp runs exactly the same pre-filter set as the default
``run.py`` pipeline (``vincent_empty_mask, vincent_border_pixel,
blur_laplacian, blur_tenengrad, vincents_artefacts``); the ``--filter_order``
argument swaps in any other order (e.g. adding the population-adapted
``vincents_area`` / ``vincents_motion_blur`` soft filters). Knobs are only
shown for the active filters: threshold knobs (absolute floors / ceilings and
population z-cutoffs) plus a dedicated **Filter Parameters** block for tuning
parameters like ``kernel_size`` / ``stroke_width`` / ``softness`` depending on
which filters are included. Knob values are written straight into the
``PipelineConfig`` filter configs, so everything shown here is exactly what
``run.py`` would do.
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

# The webapp's default pre-filter set mirrors the default run.py pipeline:
# the two hard Vincent filters (empty mask, border pixel) + the border-blur
# pair + the artifact filter. The soft vincents_area / vincents_motion_blur
# filters are NOT active unless explicitly added via --filter_order.
DEFAULT_FILTER_ORDER = [
    "vincent_empty_mask", "vincent_border_pixel",
    "blur_laplacian", "blur_tenengrad", "vincents_artefacts",
]

# (config key, label, min, max, step) for the absolute garbage thresholds.
# The filter name is the config-key prefix (key.split(".")[0]).
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

# (config key, label, min, max, step) for tunable filter parameters.
# Integer parameters (kernel_size, stroke_width) use step 1; float
# parameters (softness) use a fractional step.
PARAM_KNOBS = [
    ("vincents_artefacts.kernel_size", "vincents_artefacts · artifact kernel size", 1, 40, 1),
    ("blur_laplacian.stroke_width", "blur_laplacian · boundary band stroke width", 1, 40, 1),
    ("blur_tenengrad.stroke_width", "blur_tenengrad · boundary band stroke width", 1, 40, 1),
    ("vincents_motion_blur.stroke_width", "vincents_motion_blur · boundary band stroke width", 1, 40, 1),
    ("vincents_motion_blur.softness", "vincents_motion_blur · weight falloff softness", 0.05, 1.0, 0.05),
    ("vincents_area.softness", "vincents_area · weight falloff softness", 0.05, 1.0, 0.05),
]

# (filter name, metrics attr, label) shown as raw stats of the accepted set.
REPORT_STATS = [
    ("blur_laplacian", "laplacian", "blur_laplacian stat"),
    ("blur_tenengrad", "tenengrad", "blur_tenengrad stat"),
    ("vincents_artefacts", "vincent_artifact_fraction", "vincents_artefacts stat"),
    ("vincents_motion_blur", "vincent_boundary_blur_variance", "vincents_motion_blur stat"),
    ("vincents_area", "vincent_area_fraction", "vincents_area stat"),
]

TEMPLATE_NAME = "prefilter_template.html"


def _conf(cfg, key):
    conf_name, attr = key.split(".", 1)
    conf = getattr(cfg.filters, conf_name, None)
    if conf is None or not hasattr(conf, attr):
        return None, None
    return conf, attr


def active_knobs(cfg, knobs):
    """Only the knobs whose filter is part of the configured filter order."""
    order = set(cfg.filters.filter_order)
    return [k for k in knobs if k[0].split(".", 1)[0] in order]


def active_stats(cfg):
    order = set(cfg.filters.filter_order)
    return [s for s in REPORT_STATS if s[0] in order]


def apply_knobs(cfg: PipelineConfig, garbage=None, outlier=None, params=None):
    """Write the tuning knobs into ``cfg`` (in place) and return it.

    ``garbage`` maps config keys to plain float values; ``outlier`` maps
    config keys to ``{"enabled": bool, "value": float}`` — a disabled outlier
    knob sets the z-cutoff to ``None`` (population outlier rejection off for
    that filter); ``params`` maps config keys to parameter values (ints stay
    ints).  Unknown keys are ignored.
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
    for key, value in (params or {}).items():
        conf, attr = _conf(cfg, key)
        if conf is None:
            continue
        current = getattr(conf, attr)
        if isinstance(current, int):
            setattr(conf, attr, int(round(float(value))))
        else:
            setattr(conf, attr, float(value))
    return cfg


def config_payload(cfg: PipelineConfig):
    """Serialize the active knob values (with bounds) for the frontend."""
    garbage = [
        {"key": key, "label": label, "value": getattr(*_conf(cfg, key)),
         "min": lo, "max": hi, "step": step}
        for key, label, lo, hi, step in active_knobs(cfg, GARBAGE_KNOBS)
    ]
    outlier = []
    for key, label, lo, hi, step in active_knobs(cfg, OUTLIER_KNOBS):
        value = getattr(*_conf(cfg, key))
        outlier.append({
            "key": key, "label": label,
            "value": value if value is not None else 3.0,
            "enabled": value is not None,
            "min": lo, "max": hi, "step": step,
        })
    params = [
        {"key": key, "label": label, "value": getattr(*_conf(cfg, key)),
         "min": lo, "max": hi, "step": step}
        for key, label, lo, hi, step in active_knobs(cfg, PARAM_KNOBS)
    ]
    return {"garbage": garbage, "outlier": outlier, "params": params}


def _stats_line(attr, observations):
    values = sorted(float(getattr(obs.metrics, attr, 0.0)) for obs in observations)
    if not values:
        return "n/a"
    lo, mid, hi = values[0], median(values), values[-1]
    return f"{lo:.4f} / {mid:.4f} / {hi:.4f}"


def build_report_text(data_root, accepted, rejected, cfg, garbage=None, outlier=None, params=None):
    """Render the pre-filter outcome as plain text for the output panel."""
    lines = [
        "PRE-FILTER RUN",
        "=" * 72,
        f"data_root  : {data_root}",
        f"observations: {len(accepted) + len(rejected)}   accepted: {len(accepted)}   rejected: {len(rejected)}",
        "",
        "Filter order:",
        "  " + ", ".join(cfg.filters.filter_order),
        "",
    ]
    active_garbage = active_knobs(cfg, GARBAGE_KNOBS)
    lines.append("Garbage thresholds applied:")
    if active_garbage:
        for key, _label, _lo, _hi, _step in active_garbage:
            value = getattr(*_conf(cfg, key))
            lines.append(f"  {key:<45} {value:.4f}")
    else:
        lines.append("  (none)")
    lines.append("")
    active_outlier = active_knobs(cfg, OUTLIER_KNOBS)
    lines.append("Outlier thresholds applied:")
    if active_outlier:
        for key, _label, _lo, _hi, _step in active_outlier:
            value = getattr(*_conf(cfg, key))
            lines.append(f"  {key:<45} {value:.4f}" if value is not None
                         else f"  {key:<45} off")
    else:
        lines.append("  (none)")
    lines.append("")
    active_params = active_knobs(cfg, PARAM_KNOBS)
    lines.append("Filter parameters applied:")
    if active_params:
        for key, _label, _lo, _hi, _step in active_params:
            value = getattr(*_conf(cfg, key))
            lines.append(f"  {key:<45} {value}")
    else:
        lines.append("  (none)")
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
    for _filter, attr, label in active_stats(cfg):
        lines.append(f"  {label:<45} {_stats_line(attr, accepted)}")
    return "\n".join(lines)


def run_prefilter(data_root, garbage=None, outlier=None, params=None, filter_order=None):
    """Build the dataset + pipeline, run the pre-filter and return a report.

    Returns ``(text, accepted_ids, rejected_ids, reasons)``.
    """
    dataset = Dataset(data_root)
    dataset.load_images()
    return _run_prefilter(data_root, dataset.observations, garbage, outlier, params, filter_order)


def _run_prefilter(data_root, observations, garbage=None, outlier=None, params=None, filter_order=None):
    """Run the pre-filter pipeline over a prepared observation list."""
    from run import apply_soft_filters, build_filters, build_soft_filters

    cfg = PipelineConfig(data_root=data_root)
    if filter_order:
        cfg.filters.filter_order = list(filter_order)
    cfg.filters.explicit_filter_order = True
    cfg = apply_knobs(cfg, garbage, outlier, params)

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

    text = build_report_text(data_root, accepted, rejected, cfg, garbage, outlier, params)
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


def run_prefilter_auto(data_root, filter_order=None):
    """Run the pre-filter with auto-tuned thresholds.

    Returns ``(text, accepted_ids, rejected_ids, reasons, garbage, outlier)``
    so the frontend can show the tuned values that were applied.
    """
    garbage, outlier = auto_knobs(data_root)
    text, accepted, rejected, reasons = run_prefilter(
        data_root, garbage, outlier, filter_order=filter_order)
    return text, accepted, rejected, reasons, garbage, outlier


def run_embedding(data_root, output_dir, garbage=None, outlier=None, params=None, filter_order=None):
    """Run the full snapshot generation (pre-filter + quality + embedding).

    Writes the snapshot into ``output_dir`` — the same directory the
    embedding explorer reads — so the explorer picks up the newly-tuned
    selection pool after a reload.
    """
    from embedding_explorer_tool.algorithms import generate_snapshot

    def cfg_override(cfg):
        cfg.auto_thresholds = False
        if filter_order:
            cfg.filters.filter_order = list(filter_order)
        cfg.filters.explicit_filter_order = True
        return apply_knobs(cfg, garbage, outlier, params)

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
    def __init__(self, data_root=DEFAULT_DATA_ROOT, output_dir=DEFAULT_OUTPUT_DIR,
                 filter_order=None):
        self.data_root = data_root
        self.output_dir = output_dir
        self.filter_order = list(filter_order) if filter_order else None
        self.lock = threading.Lock()
        self.cfg = PipelineConfig(data_root=data_root)
        if self.filter_order:
            self.cfg.filters.filter_order = list(self.filter_order)
        self.cfg.filters.explicit_filter_order = True
        self.dataset = Dataset(data_root)
        self.dataset.load_images()

    def observation_count(self):
        return len(self.dataset.observations)

    def run(self, garbage=None, outlier=None, params=None):
        return _run_prefilter(self.data_root, self.dataset.observations,
                              garbage, outlier, params, self.filter_order)

    def run_auto(self):
        garbage, outlier = _auto_knobs(self.dataset.observations)
        text, accepted, rejected, reasons = \
            _run_prefilter(self.data_root, self.dataset.observations,
                           garbage, outlier, None, self.filter_order)
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
                        app.run(payload.get("garbage"), payload.get("outlier"),
                                payload.get("params"))
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
                                         payload.get("garbage"), payload.get("outlier"),
                                         payload.get("params"), app.filter_order)
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
    parser.add_argument("--filter_order", type=str, default=None,
                        help="Comma-separated pre-filter order (default: same set as "
                             "the default run.py pipeline: "
                             "vincent_empty_mask,vincent_border_pixel,blur_laplacian,"
                             "blur_tenengrad,vincents_artefacts). Only the named "
                             "filters run; e.g. add vincents_area,vincents_motion_blur "
                             "to include the population-adapted soft filters.")
    args = parser.parse_args()

    filter_order = None
    if args.filter_order:
        filter_order = [name.strip() for name in args.filter_order.split(",") if name.strip()]

    app = PrefilterApp(data_root=args.data_root, output_dir=args.output_dir,
                       filter_order=filter_order)
    handler = make_handler(app)
    server = QuietThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Pre-filter tuner on http://localhost:{args.port}/")
    print(f"  data_root: {app.data_root}")
    print(f"  snapshot output (embedding explorer): {app.output_dir}")
    print(f"  filter order: {', '.join(app.cfg.filters.filter_order)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
