"""Start both web apps (embedding explorer + pre-filter tuner) together.

Usage:
    python run_webapps.py -i /path/to/dataset -o ./outputs_embedding_explorer

The wrapper first ensures a snapshot exists in ``--output`` by running the
pre-filter + quality-scoring + embedding stages (the same "init" the
explorer would otherwise do on startup). Only when that is finished are the
two servers started and the URLs printed, so both pages open with data
ready:

* tuner (port 8520) — tune the pre-filter thresholds (default filter set =
  the default run.py pipeline; `--filter-order` swaps in any other order),
  preview the accept/reject outcome, then "Run Embedding" to regenerate the
  snapshot with the tuned thresholds.
* explorer (port 8510) — visualise the snapshot; after a tuner
  "Run Embedding", click "Reload Snapshot" to see the new pool.

Ctrl+C stops both.
"""

import argparse
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPLORER_PORT = 8510
TUNER_PORT = 8520
READY_TIMEOUT = 120.0


def _stream(name, stream):
    for line in iter(stream.readline, ""):
        print(f"[{name}] {line}", end="", flush=True)
    stream.close()


def _ready(port, path, timeout=10.0):
    url = f"http://127.0.0.1:{port}{path}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            time.sleep(0.5)
    return False


def wait_ready(explorer_port, tuner_port):
    """Block until both servers answer, so URLs are printed once pages are up."""
    ok_e = _ready(explorer_port, "/api/reload", READY_TIMEOUT)
    ok_t = _ready(tuner_port, "/api/config", READY_TIMEOUT)
    if not ok_e:
        print(f"[wrapper] embedding explorer did not come up on port {explorer_port}.", flush=True)
    if not ok_t:
        print(f"[wrapper] pre-filter tuner did not come up on port {tuner_port}.", flush=True)
    return ok_e and ok_t


def init_snapshot(data_root, output_dir, force=False):
    """Run the pre-filter + embedding init once so both pages start with data.

    Uses static config thresholds (auto-tuning off, the pipeline default);
    skipped when ``output_dir`` already holds a snapshot (unless ``force``).
    """
    from embedding_explorer_tool.algorithms import snapshot_exists, generate_snapshot

    if not force and snapshot_exists(output_dir):
        print(f"Snapshot already present in {output_dir}; skipping initial run.",
              flush=True)
        return
    print(f"Initial run: pre-filter + quality + embedding from {data_root} ...",
          flush=True)
    generate_snapshot(output_dir, data_root, auto_thresholds=False)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter + embed a dataset, then start the "
                    "embedding explorer + pre-filter tuner webapps"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="dataset root with images/ and masks/")
    parser.add_argument("-o", "--output", required=True,
                        help="shared snapshot dir (explorer reads it, tuner writes it)")
    parser.add_argument("--explorer-port", type=int, default=EXPLORER_PORT)
    parser.add_argument("--tuner-port", type=int, default=TUNER_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open the explorer in a browser")
    parser.add_argument("--regen", action="store_true",
                        help="re-run the init pre-filter + embedding even if a "
                             "snapshot already exists")
    parser.add_argument("--filter-order", dest="filter_order", type=str, default=None,
                        help="comma-separated pre-filter order for the tuner (default: "
                             "same set as the default run.py pipeline; add "
                             "vincents_area,vincents_motion_blur to include the "
                             "population-adapted soft filters)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.default_int_handler)

    data_root = str(Path(args.input).resolve())
    output_dir = str(Path(args.output).resolve())

    print("=== Two-page webapp init ===")
    print(f"  data_root : {data_root}", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)

    from embedding_explorer_tool.algorithms import snapshot_exists

    if args.regen or not snapshot_exists(output_dir):
        init_snapshot(data_root, output_dir, force=args.regen)

    commands = [
        ("explorer", [sys.executable, "-m", "embedding_explorer_tool.webapp",
                      "--data_root", data_root, "--output_dir", output_dir,
                      "--port", str(args.explorer_port)]
         + (["--no-browser"] if args.no_browser else [])),
        ("tuner", [sys.executable, "-m", "embedding_explorer_tool.prefilter_app",
                   "--data_root", data_root, "--output_dir", output_dir,
                   "--port", str(args.tuner_port)]
         + (["--filter_order", args.filter_order] if args.filter_order else [])),
    ]

    print("  starting servers ...", flush=True)
    procs = []
    for name, cmd in commands:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append((name, proc))
        threading.Thread(target=_stream, args=(name, proc.stdout), daemon=True).start()

    if not wait_ready(args.explorer_port, args.tuner_port):
        for _name, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for _name, proc in procs:
            proc.wait()
        sys.exit(1)

    print("\n=== Webapps ready ===")
    print(f"  pre-filter tuner  : http://127.0.0.1:{args.tuner_port}/")
    print(f"  embedding explorer: http://127.0.0.1:{args.explorer_port}/")
    print("  Ctrl+C stops both.\n", flush=True)

    try:
        while True:
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"[wrapper] {name} exited with code {proc.returncode}; "
                          "stopping the other.")
                    for _name, _proc in procs:
                        if _proc.poll() is None:
                            _proc.terminate()
                    sys.exit(1)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[wrapper] stopping both servers ...")
        for _name, proc in procs:
            proc.terminate()
        for _name, proc in procs:
            proc.wait()
        print("[wrapper] done.")


if __name__ == "__main__":
    main()
