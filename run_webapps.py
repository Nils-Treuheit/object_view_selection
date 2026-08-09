"""Start both web apps (embedding explorer + pre-filter tuner) together.

Usage:
    python run_webapps.py -i /path/to/dataset -o ./outputs_embedding_explorer

Both servers share the same ``--data_root`` / ``--output_dir``: the tuner
(port 8520) writes the snapshot the explorer (port 8510) reads. After
"Run Embedding" in the tuner, click "Reload Snapshot" in the explorer.

Ctrl+C stops both.
"""

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPLORER_PORT = 8510
TUNER_PORT = 8520


def _stream(name, stream):
    for line in iter(stream.readline, ""):
        print(f"[{name}] {line}", end="", flush=True)
    stream.close()


def main():
    parser = argparse.ArgumentParser(
        description="Start the embedding explorer + pre-filter tuner webapps"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="dataset root with images/ and masks/")
    parser.add_argument("-o", "--output", required=True,
                        help="shared snapshot dir (explorer reads it, tuner writes it)")
    parser.add_argument("--explorer-port", type=int, default=EXPLORER_PORT)
    parser.add_argument("--tuner-port", type=int, default=TUNER_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open the explorer in a browser")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.default_int_handler)

    data_root = str(Path(args.input).resolve())
    output_dir = str(Path(args.output).resolve())

    commands = [
        ("explorer", [sys.executable, "-m", "embedding_explorer_tool.webapp",
                      "--data_root", data_root, "--output_dir", output_dir,
                      "--port", str(args.explorer_port)]
         + (["--no-browser"] if args.no_browser else [])),
        ("tuner", [sys.executable, "-m", "embedding_explorer_tool.prefilter_app",
                   "--data_root", data_root, "--output_dir", output_dir,
                   "--port", str(args.tuner_port)]),
    ]

    print("Starting both web apps ...")
    print(f"  data_root   : {data_root}", flush=True)
    print(f"  output_dir  : {output_dir}", flush=True)
    print(f"  explorer    : http://127.0.0.1:{args.explorer_port}/", flush=True)
    print(f"  tuner       : http://127.0.0.1:{args.tuner_port}/", flush=True)
    print("  Ctrl+C stops both.\n", flush=True)

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
