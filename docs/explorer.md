# Embedding Explorer Tool

Interactive explorer for the **Top kMeans Embedding Selection in xNN quality
Neighborhood** (`TopKMeansXNN`, `selection/kmeans_xnn.py`) *before* top-k
selection is applied. It visualises the embedding pool, the k-means clusters,
the constrained xNN neighbourhoods and the final picks, and lets you inspect
the actual frames side by side.

Two frontends share the same algorithms (`embedding_explorer_tool/algorithms.py`):

- **Web app** — everything in one browser window, interactive Plotly **3D**.
- **tkinter app** — offline desktop mirror with an embedded matplotlib **2D** scatter.

## Inputs

The tool reads a pipeline snapshot (an `--output_dir` produced by `run.py`)
and the dataset root:

| Input | Source |
|-------|--------|
| `embeddings.npy`, `selection_pool_ids.npy`, `quality.csv`, `report.json` | pipeline `--output_dir` |
| `images/`, `masks/` | dataset `--data_root` (default taken from `report.json`) |

## Web app

```bash
python -m embedding_explorer_tool.webapp
python -m embedding_explorer_tool.webapp --output_dir ./outputs
python -m embedding_explorer_tool.webapp --output_dir ./outputs --data_root /path/to/bottle
python -m embedding_explorer_tool.webapp --port 9000 --no-browser
```

Arguments:

| Argument | Default | Effect |
|----------|---------|--------|
| `--output_dir` | last verified run | Pipeline snapshot directory |
| `--data_root` | from `report.json` | Dataset root with `images/` and `masks/` |
| `--port` | `8510` | Local server port |
| `--no-browser` | off | Do not auto-open the browser |

The page loads `plotly.min.js` from the vendored copy in
`embedding_explorer_tool/static/`, so it works fully offline.

### Layout

```
┌─────────────────────────────┬──────────────────────┐
│  k (clusters) | init | xNN  │  Frame ID | Show     │  <- controls
├─────────────────────────────┼──────────────────────┤
│  Interactive 3D MDS (plotly)│  Frame + mask viewer │
│                             │                      │
│                             ├──────────────────────┤
│                             │  Scrollable text     │
│                             │  output              │
└─────────────────────────────┴──────────────────────┘
```

### Markers in the 3D plot

| Marker | Meaning |
|--------|---------|
| Small dots (coloured per cluster) | Pool samples; dot **alpha = quality score** (max quality ⇒ 100 %, quality 0 ⇒ 0 %) |
| Larger dots with thick black outline | Constrained xNN candidates of a centroid |
| Ring + `★` in cluster colour | Centroid frame (medoid — nearest pool sample to the k-means centre) |
| Gold `★` | Final pick (best-quality sample in `{centroid} ∪ xNN`) |

Hovering any dot shows the frame in the right viewer with the mask overlay
(object region: strong green tint over the low-opacity frame; background
dimmed to 66 %). The tooltip shows the sample ID and quality.

### Controls

- **k (clusters)** — k-means `k = n`, one cluster per requested view.
- **init** — `farthest` (deterministic farthest-point seeds, starts at the
  highest-quality sample) or `best_quality` (top-quality seeds).
- **xNN k** — neighbourhood radius: candidate set is
  `{centroid} ∪ its x nearest neighbours`, constrained so a candidate is only
  considered for a centroid it is closer to than to any other centroid;
  medoid fallback when the whole neighbourhood is dropped.
- **Run** — recomputes k-means + xNN and redraws the plot.
- **Frame ID / Show Frame** — view any frame directly.

### Text output

The scrollable field shows the centroid frame IDs as a list and the
constrained xNN of each centroid as a dictionary, plus the final picks and
their qualities:

```
k = 8   init = farthest   xNN = 5

Centroid frame IDs (8):
[111, 48, 96, 146, 41, 49, 30, 0]

Constrained xNN per centroid (keys = centroid frame IDs):
{ "111": [111, 110, 112, 109, 106, 114], ... }

Final picks (best quality in xNN, 8):
[102, 47, 157, 145, 69, 49, 46, 6]
```

The picks reproduced exactly what `run.py --selector top_kmeans_xnn` outputs
for the same pool / init / xNN, so the explorer doubles as a debugging tool
for the selector.

## tkinter app

```bash
python -m embedding_explorer_tool.gui_tk
python -m embedding_explorer_tool.gui_tk --output_dir ./outputs --data_root /path/to/bottle
```

Same controls and semantics as the web app, with an embedded matplotlib **2D**
MDS scatter (the web app keeps the interactive 3D view). Requires a graphical
session (no Xvfb available on the current host).

## Module layout

| File | Purpose |
|------|---------|
| `algorithms.py` | Snapshot loading, quality/farthest seeds, k-means, constrained xNN, 3D MDS, mask overlay, text output |
| `webapp_plotting.py` | Plotly figure builder |
| `webapp.py` | Local HTTP server (`/`, `/api/run`, `/composite/`, `/image/`, `/mask/`) |
| `webapp_template.html` | Single-page frontend (HTML/CSS/JS) |
| `gui_tk.py` | tkinter + matplotlib mirror |
| `static/plotly.min.js` | Vendored plotly bundle (offline) |
