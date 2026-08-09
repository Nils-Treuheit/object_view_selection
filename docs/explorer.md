# Embedding Explorer Tool

Interactive explorer for the **Top kMeans Embedding Selection in xNN quality
Neighborhood** (`TopKMeansXNN`, `selection/kmeans_xnn.py`) *before* top-k
selection is applied. It visualises the embedding pool, the k-means clusters,
the constrained xNN neighbourhoods and the final picks, and lets you inspect
the actual frames side by side.

The frontend lives in `embedding_explorer_tool/` (two web apps sharing the
algorithms in `embedding_explorer_tool/algorithms.py`):

- **Web app** — everything in one browser window, interactive Plotly figure with a
  **2D/3D** switch (port **8510**).
- **Pre-filter tuner** — tune the pre-filter thresholds on a dataset, preview the
  accept/reject outcome, then generate the snapshot the web app reads (port **8520**).

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

## Output structure (snapshot directory)

The tool operates on a **snapshot directory** — a pipeline `--output_dir`. It
reads the files below and, when generating a fresh snapshot, writes them too:

```
outputs_embedding_explorer/
├── embeddings.npy           # (N, D) embedding matrix, aligned to the pool ids
├── selection_pool_ids.npy   # (N,) int frame ids; pool_ids[i] <-> embeddings[i]
├── quality.csv              # id + quality per pool row (same alignment)
├── report.json              # {"data_root", "embedding", "embedding_model"}
└── selected_indices.npy     # optional: run.py's selected indices into embeddings.npy
```

- **Row alignment** is the contract: `embeddings[i]`, `pool_ids[i]` and
  `quality[i]` all refer to the same observation.
- **Selected views** — when a pipeline run wrote `selected_indices.npy`, the
  tool loads it to mark the run's `top_kmeans_xnn` picks; it is not required
  for the explorer to work.
- **`data_root`** defaults to the value stored in `report.json`; it is only
  needed explicitly when generating a snapshot for a dataset that was never
  pipelined.
- The explorer itself never writes back to the snapshot — it is read-only once
  loaded; `generate_snapshot` is the only writer and runs before launch.

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

## Pre-filter tuner web app

The second frontend (port **8520**) is the step before the explorer: it loads a
dataset once and lets you tune the **pre-filter** thresholds and see exactly how
many observations pass before any embedding runs.

```bash
python -m embedding_explorer_tool.prefilter_app
python -m embedding_explorer_tool.prefilter_app --data_root /path/to/object \
    --output_dir ./outputs_embedding_explorer --port 8520
```

Arguments:

| Argument | Default | Effect |
|----------|---------|--------|
| `--data_root` | `.../workspace/intresting_objects/elephant` | Dataset root with `images/` and `masks/` |
| `--output_dir` | `outputs_embedding_explorer` | Snapshot dir the explorer reads; "Run Embedding" writes here |
| `--port` | `8520` | Local server port (one off the explorer's 8510) |

Layout:

```
┌────────────────────────────┬─────────────────────────────┐
│ Garbage Thresholds         │  PRE-FILTER RUN (text)      │
│  blur_laplacian floor      │  observations / accepted /  │
│  blur_tenengrad floor      │  rejected                   │
│  artefacts ceiling  ...    │  applied thresholds         │
│ Outlier Thresholds         │  rejected-by-filter counts  │
│  z-cutoff per filter       │  accepted raw stats         │
│  (checkbox to disable)     │                             │
├────────────────────────────┼─────────────────────────────┤
│ [Apply Auto Thresholds]    │  [Run Embedding]            │
│ [Run Pre-Filter]           │                             │
└────────────────────────────┴─────────────────────────────┘
```

- **Garbage Thresholds** — absolute floors / ceilings on the raw stats that
  reject a sample regardless of the population
  (`hard_min_variance`, `hard_min_tenengrad`, `hard_max_fraction`,
  motion-blur floor, area floor).
- **Outlier Thresholds** — robust population z-cutoffs
  (`outlier_z` per filter); uncheck a row to disable that filter's outlier
  rejection.
- **Run Pre-Filter** — runs the actual `run.py` pre-filter pipeline
  (`build_filters` + `build_soft_filters`) with the current knobs and renders
  the outcome: applied thresholds, rejected-by-filter counts and the
  accepted-set raw stats.
- **Apply Auto Thresholds** — computes the data-driven thresholds
  (`utils.threshold_tuner.tune_thresholds`, the same percentile floors as
  `run.py --auto-thresholds`), writes them into the knob fields and runs the
  pre-filter with them.
- **Run Embedding** — generates the snapshot (`algorithms.generate_snapshot`)
  with the current knobs into `--output_dir`, so a reload of the explorer
  shows the newly-tuned selection pool.

## Module layout

| File | Purpose |
|------|---------|
| `algorithms.py` | Snapshot loading, quality/farthest seeds, k-means, constrained xNN, pure MDS projection (2D or 3D), mask overlay, text output; `snapshot_exists` / `generate_snapshot` for producing a snapshot from a raw `--data_root` |
| `webapp_plotting.py` | Plotly figure builder |
| `webapp.py` | Explorer HTTP server (`/`, `/api/run`, `/composite/`, `/image/`, `/mask/`) |
| `webapp_template.html` | Explorer frontend (HTML/CSS/JS) |
| `prefilter_app.py` | Pre-filter tuner HTTP server (`/`, `/api/config`, `/api/run`, `/api/run_auto`, `/api/embed`) |
| `prefilter_template.html` | Pre-filter tuner frontend (HTML/CSS/JS) |
| `static/plotly.min.js` | Vendored plotly bundle (offline) |
