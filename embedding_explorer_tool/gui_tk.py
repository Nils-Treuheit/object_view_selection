"""tkinter + embedded matplotlib mirror of the embedding explorer.

Same semantics as the web app but fully offline in a desktop window with a
**2D** MDS scatter on the left (cluster colours, quality-linked alpha, stars
for centroids, black-outlined dots for xNN candidates), an image viewer with
the mask overlay and a scrollable text output on the right, and text fields
for k, x and a frame ID.

Run::

    python -m embedding_explorer_tool.gui_tk [--output_dir ...] [--data_root ...]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

import tkinter as tk
import tkinter.font as tkfont
from tkinter import scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from PIL import Image, ImageTk

try:
    from . import algorithms
except ImportError:
    import algorithms

UI_FONT_FAMILY = "DejaVu Sans"
MONO_FONT_FAMILY = "DejaVu Sans Mono"
RIGHT_PADDING = 10


def _cluster_palette(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def plot_result(ax, coords, quality, labels, result, pool_ids):
    """Draw the 2D MDS scatter for one kMeans + xNN run on ``ax``.

    Returns a list of ``(artist_id, frame_ids)`` used to resolve pick events.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ids = np.asarray(pool_ids, dtype=int)
    quality = np.asarray(quality, dtype=float)
    clusters = result["clusters"]
    k = result["k"]
    palette = _cluster_palette(k)

    x, y = coords[:, 0], coords[:, 1]
    ax.clear()
    mapping = []

    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        rgba = [(palette[c][0], palette[c][1], palette[c][2], float(a)) for a in quality[mask]]
        artist = ax.scatter(
            x[mask], y[mask],
            s=32, c=rgba,
            label="Pool samples" if c == 0 else None,
            picker=6,
        )
        mapping.append((id(artist), ids[mask]))

        cand = np.asarray(clusters[c]["candidates"], dtype=int)
        cand_colors = [palette[c][:3]] * len(cand)
        artist = ax.scatter(
            x[cand], y[cand],
            s=110, c=cand_colors,
            edgecolors="black", linewidths=1.8, picker=6,
        )
        mapping.append((id(artist), ids[cand]))

    medoid = np.asarray([clusters[c]["medoid"] for c in range(k)], dtype=int)
    medoid_colors = [palette[c][:3] for c in range(k)]
    artist = ax.scatter(
        x[medoid], y[medoid],
        s=360, c=medoid_colors, marker="*",
        edgecolors="black", linewidths=1.0, label="Centroid (medoid frame)",
        picker=8,
    )
    mapping.append((id(artist), ids[medoid]))

    picks = np.asarray(result["picks"], dtype=int)
    artist = ax.scatter(
        x[picks], y[picks],
        s=300, c="#FFD700", marker="*",
        edgecolors="black", linewidths=1.2, label="Final pick",
        picker=8,
    )
    mapping.append((id(artist), ids[picks]))

    ax.scatter([], [], s=32, c="grey", label="xNN candidates")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_title(f"2D MDS of kMeans clusters (k={k}, init={result['init']}, xNN={result['x']})")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax.margins(0.08)
    return mapping


class ExplorerApp:
    def __init__(self, root, output_dir, data_root):
        self.root = root
        self._configure_fonts()

        self.snapshot = algorithms.load_snapshot(output_dir)
        self.embeddings = self.snapshot["embeddings"]
        self.pool_ids = self.snapshot["pool_ids"]
        self.quality = self.snapshot["quality"]
        self.coords = algorithms.project_mds(self.embeddings, n_components=2)

        root_path = Path(data_root) if data_root else Path(self.snapshot.get("data_root") or "")
        self.data_root = root_path
        self._image_dir = root_path / "images"
        self._mask_dir = root_path / "masks"
        self._image_cache = {}

        self.result = None
        self._id_map = {}

        self._build_widgets()
        self.run(8, "farthest", 3)

    def _configure_fonts(self):
        default = tkfont.nametofont("TkDefaultFont")
        default.configure(family=UI_FONT_FAMILY, size=11)
        for name in ("TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=UI_FONT_FAMILY, size=11)
            except tk.TclError:
                pass

    def _build_widgets(self):
        self.root.title("Embedding Explorer — Top kMeans in xNN Quality Neighborhood")
        self.root.geometry("1400x880")

        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="k (clusters):").pack(side=tk.LEFT)
        self.k_var = tk.StringVar(value="8")
        ttk.Entry(top, textvariable=self.k_var, width=6).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(top, text="init:").pack(side=tk.LEFT)
        self.init_var = tk.StringVar(value="farthest")
        ttk.Combobox(top, textvariable=self.init_var, values=["farthest", "best_quality"],
                     state="readonly", width=12).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(top, text="xNN k:").pack(side=tk.LEFT)
        self.x_var = tk.StringVar(value="3")
        ttk.Entry(top, textvariable=self.x_var, width=6).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Button(top, text="Run", command=self._on_run).pack(side=tk.LEFT, padx=(0, 24))

        ttk.Label(top, text="Frame ID:").pack(side=tk.LEFT)
        self.frame_var = tk.StringVar(value="0")
        ttk.Entry(top, textvariable=self.frame_var, width=8).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(top, text="Show Frame", command=self._on_show_frame).pack(side=tk.LEFT)

        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=55)
        main.columnconfigure(1, weight=45)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew")
        self.fig = Figure(figsize=(7, 6.5), dpi=110, facecolor="#15171e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0d0f14")
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        right = ttk.Frame(main, padding=(0, 0, RIGHT_PADDING, RIGHT_PADDING))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)

        self.image_label = tk.Label(right, anchor="center", background="#000")
        self.image_label.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.text_out = scrolledtext.ScrolledText(
            right, wrap=tk.NONE, state=tk.DISABLED,
            font=(MONO_FONT_FAMILY, 11), background="#0d0f14", foreground="#e6e6e6",
            insertbackground="#e6e6e6",
        )
        self.text_out.grid(row=1, column=0, sticky="nsew")

    def run(self, k, init, x):
        self.result = algorithms.run_kmeans_xnn(self.embeddings, self.quality, k, init, x)
        mapping = plot_result(self.ax, self.coords, self.quality, self.result["labels"],
                              self.result, self.pool_ids)
        self._id_map = dict(mapping)
        self.canvas.draw_idle()
        self._set_text(algorithms.build_text(self.result, self.pool_ids, self.quality))

    def _set_text(self, text):
        self.text_out.configure(state=tk.NORMAL)
        self.text_out.delete("1.0", tk.END)
        self.text_out.insert("1.0", text)
        self.text_out.configure(state=tk.DISABLED)

    def _on_run(self):
        try:
            k = int(self.k_var.get())
            init = self.init_var.get()
            x = int(self.x_var.get())
        except ValueError:
            return
        self.run(k, init, x)

    def _on_pick(self, event):
        ids = self._id_map.get(id(event.artist))
        if ids is None or len(event.ind) == 0:
            return
        self.show_frame(int(ids[event.ind[0]]))

    def _on_show_frame(self):
        try:
            frame_id = int(self.frame_var.get())
        except ValueError:
            return
        self.show_frame(frame_id)

    def show_frame(self, frame_id):
        self.frame_var.set(str(frame_id))
        pil = self._load_composite(frame_id)
        if pil is None:
            return
        photo = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def _load_composite(self, frame_id):
        if frame_id in self._image_cache:
            return self._image_cache[frame_id]
        img_path = self._image_dir / f"{frame_id:05d}.png"
        mask_path = self._mask_dir / f"{frame_id:05d}.png"
        if not img_path.exists() or not mask_path.exists():
            return None
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        out = algorithms.compose_mask_overlay(image, mask)
        pil = Image.fromarray(out)
        pil.thumbnail((640, 640))
        self._image_cache[frame_id] = pil
        return pil


def main():
    parser = argparse.ArgumentParser(description="Embedding explorer desktop app")
    parser.add_argument("--output_dir", type=str, default="/tmp/opencode/10_verify_out")
    parser.add_argument("--data_root", type=str, default="",
                        help="Dataset root with images/ and masks/ (default: from report.json)")
    args = parser.parse_args()

    root = tk.Tk()
    ExplorerApp(root, args.output_dir, args.data_root)
    root.mainloop()


if __name__ == "__main__":
    main()
