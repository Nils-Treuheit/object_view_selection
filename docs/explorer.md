# Embedding Explorer Tool

Interactive explorer for the **Top kMeans Embedding Selection in xNN quality
Neighborhood** (`TopKMeansXNN`, `selection/kmeans_xnn.py`) *before* top-k
selection is applied. It visualises the embedding pool, the k-means clusters,
the constrained xNN neighbourhoods and the final picks, and lets you inspect
the actual frames side by side.

Two frontends share the same algorithms (`embedding_explorer_tool/algorithms.py`):

- **Web app** — everything in one browser window, interactive Plotly figure with a
  **2D/3D** switch.
- **tkinter app** — offline desktop mirror with an embedded matplotlib scatter,
  also switchable between **2D** and **3D**, on a bright/white background.

The MDS projection itself is a **pure MDS** of the embedding space (metric MDS
over cosine distance, `algorithms.project_mds`) computed without any knowledge
of the k-means clusters. The cluster colours, xNN candidates, centroids and
final picks are drawn on top of that fixed projection afterwards, so switching
clusters (`k`, `init`, `xNN`) never changes the point positions.

## Inputs

The tool reads a pipeline snapshot (an `--output_dir` produced by `run.py`)
and the dataset root:

| Input | Source |
|-------|--------|
| `embeddings.npy`, `selection_pool_ids.npy`, `quality.csv`, `report.json` | pipeline `--output_dir` |
| `images/`, `masks/` | dataset `--data_root` (default taken from `report.json`) |

If `--output_dir` has **no snapshot** yet (`embeddings.npy` +
`selection_pool_ids.npy` missing), the app **generates it first** from
`--data_root`: it runs the same pre-filtering, quality scoring and embedding
stages as `run.py`, writes the snapshot into `--output_dir`, and only then
launches. That means you can point the tool at a dataset that was never
pipelined and explore it immediately.

## Web app

```bash
# Point at an existing pipeline output
python -m embedding_explorer_tool.webapp
python -m embedding_explorer_tool.webapp --output_dir ./outputs
python -m embedding_explorer_tool.webapp --output_dir ./outputs --data_root /path/to/bottle

# Generate a fresh snapshot from a dataset, then explore it
python -m embedding_explorer_tool.webapp \
    --output_dir ./outputs_embedding_explorer \
    --data_root /path/to/triprong \
    --embedding dinov2 --embedding_model dinov2_vitb14_reg

# Custom port / no auto-open
python -m embedding_explorer_tool.webapp --port 9000 --no-browser
```

Arguments:

| Argument | Default | Effect |
|----------|---------|--------|
| `--output_dir` | `outputs_embedding_explorer` | Snapshot directory. Created and populated from `--data_root` when it has no snapshot yet |
| `--data_root` | from `report.json` | Dataset root with `images/` and `masks/`. Required when `--output_dir` has no snapshot |
| `--embedding` | `auto` | Embedding type used only when generating a fresh snapshot (`auto` infers from `--embedding_model`). Same choices and default as `run.py` |
| `--embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model name or path for snapshot generation; type inferred automatically when `--embedding=auto`. Same default as `run.py` |
| `--port` | `8510` | Local server port |
| `--no-browser` | off | Do not auto-open the browser |

The embedding choices mirror `run.py`: `auto`, `dinov3`, `dinov2`, `siglip2`,
`siglip`, `moonvit`, `clip`, `eva_clip`. Re-running against an `--output_dir`
that already holds a snapshot ignores these flags and loads the saved
embeddings as-is.

The page loads `plotly.min.js` from the vendored copy in
`embedding_explorer_tool/static/`, so it works fully offline.

### Layout

```
┌─────────────────────────────┬──────────────────────┐
│  k (clusters) | init | xNN  │  Frame ID | Show     │  <- controls
│  Run | 2D/3D                │                      │
├─────────────────────────────┼──────────────────────┤
│  Interactive MDS (plotly)   │  Frame + mask viewer │
│                             │                      │
│                             ├──────────────────────┤
│                             │  Scrollable text     │
│                             │  output              │
└─────────────────────────────┴──────────────────────┘
```

### Markers in the plot

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
- **2D / 3D** — switches the plotly figure between a 2D scatter (first two MDS
  components, the default view) and the 3D scatter (first three components).
  The projection is the same MDS; only the view changes.
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
python -m embedding_explorer_tool.gui_tk \
    --output_dir ./outputs_embedding_explorer \
    --data_root /path/to/triprong \
    --embedding dinov2 --embedding_model dinov2_vitb14_reg
```

Same controls and semantics as the web app, with an embedded matplotlib MDS
scatter on a **white background**. The **View: 2D / 3D** radios in the top
bar re-run `project_mds` with `n_components = 2` or `3` (the embeddings are
unchanged) and re-draw the clusters/candidates/picks on the new projection.
It accepts the same `--output_dir`, `--data_root`, `--embedding` and
`--embedding_model` arguments (with the same auto-generation behaviour).
Requires a graphical session (no Xvfb available on the current host).

## Module layout

| File | Purpose |
|------|---------|
| `algorithms.py` | Snapshot loading, quality/farthest seeds, k-means, constrained xNN, pure MDS projection (2D or 3D), mask overlay, text output; `snapshot_exists` / `generate_snapshot` for producing a snapshot from a raw `--data_root` |
| `webapp_plotting.py` | Plotly figure builder |
| `webapp.py` | Local HTTP server (`/`, `/api/run`, `/composite/`, `/image/`, `/mask/`) |
| `webapp_template.html` | Single-page frontend (HTML/CSS/JS) |
| `gui_tk.py` | tkinter + matplotlib mirror |
| `static/plotly.min.js` | Vendored plotly bundle (offline) |
