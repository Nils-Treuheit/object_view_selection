# Object View Selection

Select the best **N image/mask pairs** that maximize **object identifiability** — high-quality, diverse, non-redundant viewpoints of a single object.

```
Dataset → Auto-Threshold → Pre-Filter → Quality Score → Embeddings → Subset Selection → Outputs
```

| Stage | What it does |
|-------|-------------|
| **Auto-Threshold** | Tunes legacy filters via dataset statistics; the default blur/artifact pre-filters use static relaxed floors + population-relative outlier rejection |
| **Pre-filter** | 5 conservative filters: empty mask, frame-touching mask, boundary-band Laplacian blur, boundary-band Tenengrad blur, mask artifacts (relaxed floor + robust outlier rejection) |
| **Quality Score** | Weighted combination of exactly 4 components: boundary-blur, mask-area, mask-artifacts, centerness |
| **Embeddings** | DINOv3 / DINOv2 / SigLIP / CLIP / EVA-CLIP features (or classical shape descriptors on CPU) |
| **Selection** | Greedy quality+diversity, FPS, Facility Location, DPP, or NBV |

See [`docs/pipeline.md`](docs/pipeline.md) for detailed module descriptions and [`docs/thresholds.md`](docs/thresholds.md) for the auto-tuning strategy.

## Installation

```bash
git clone <repo> && cd object_view_selection

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies via uv sync
uv sync 

# or: Install dependencies with pip
pip install -r requirements.txt

# or: Manual install dependencies
pip install numpy opencv-python scipy scikit-image scikit-learn Pillow tqdm pandas matplotlib

# For learned embeddings (optional):
pip install torch torchvision timm
pip install git+https://github.com/openai/CLIP.git        # CLIP
pip install transformers                                  # SigLIP, DINOv3
pip install open-clip-torch                               # EVA-CLIP
```

## Dataset Structure

```
bottle/
├── images/          # 00000.png, 00001.png, ...
├── masks/           # 00000.png, 00001.png, ...  (binary, same filename)
└── object_hands/    # 00000.png, 00001.png, ...  (binary hand occlusion, optional)
```

## Usage

```bash
# Run the full pipeline (auto-tuning enabled by default)
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs

# DINOv3 embeddings (default model: dinov3-vitb16)
python run.py --data_root /path/to/bottle --num_views 10

# Switch model (auto-detected from name)
python run.py --data_root /path/to/bottle \
  --embedding_model facebook/dinov3-vitl16-pretrain-lvd1689m

# SigLIP2 or MoonViT
python run.py --data_root /path/to/bottle \
  --embedding_model google/siglip2-base-patch16-224
python run.py --data_root /path/to/bottle \
  --embedding_model moonshotai/MoonViT-SO-400M

# Explicit type override (e.g. fall back to DINOv2)
python run.py --data_root /path/to/bottle \
  --embedding dinov2 --embedding_model dinov2_vitb14_reg

# Shape descriptors (CPU only, no GPU needed)
python run.py --data_root /path/to/bottle \
  --use_shape_descriptors --num_views 10
python run.py --data_root /path/to/bottle \
  --use_shape_descriptors --shape_descriptor zernike

# Different selection strategies
python run.py --data_root /path/to/bottle --selector fps --num_views 10
python run.py --data_root /path/to/bottle --selector dpp --num_views 10
python run.py --data_root /path/to/bottle \
  --selector quality_diversity --selector_alpha 0.3 --selector_beta 0.7

# Top kMeans Embedding Selection in xNN quality Neighborhood
#   --kmeans_init   farthest | best_quality  (cluster seeding)
#   --kmeans_xnn_k  3 | 5 | 10               (neighbourhood radius)
python run.py --data_root /path/to/bottle \
  --selector top_kmeans_xnn --kmeans_init farthest --kmeans_xnn_k 5
python run.py --data_root /path/to/bottle \
  --selector top_kmeans_xnn --kmeans_init best_quality --kmeans_xnn_k 10

# Custom pre-filter order (comma-separated, defaults to the config order)
python run.py --data_root /path/to/bottle \
  --filter_order "vincent_empty_mask,vincent_border_pixel,blur_laplacian,blur_tenengrad,vincents_artefacts"

# Disable auto-threshold tuning (use static config values instead)
python run.py --data_root /path/to/bottle --no-auto-thresholds
```

### Generating Plots

Use the `--plot` flag to generate diagnostic plots after selection:

```bash
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs --plot

# With debug mode (all DR methods, not just PCA + MDS):
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs --plot --debug
```

Plots are saved under `outputs/plots/` and example frames under `outputs/bad_examples/`. `--debug` additionally enables single-set violins, all dimensionality-reduction methods, and the embedding-neighbour diagnostics (`plots/selection/selected_neighbors_*.png`, `plots/selection/selected_clusters_pca.png`). See [`docs/plotting.md`](docs/plotting.md) for a complete reference.

### Standalone Plotting

Re-generate plots from a previous pipeline run without re-running the pipeline:

```bash
python -m plotting_process.wrapper --input_dir ./outputs [--output_dir ./plots] [--debug]
```

If `--output_dir` is omitted, the `plots/` folder is created inside `--input_dir`. The selected-sample folders are produced by the pipeline itself (`run.py`), not by the standalone plotter.

### Embedding Explorer

Interactive 3D visualisation of the kMeans + constrained-xNN selection *before* top-k selection — the embedding pool, cluster colours, quality-linked dot alpha, centroid stars, xNN candidates and final picks, with a live frame viewer (mask overlay):

```bash
# Web app (single browser window, offline-capable plotly)
python -m embedding_explorer_tool.webapp --output_dir ./outputs

# tkinter + matplotlib desktop mirror
python -m embedding_explorer_tool.gui_tk --output_dir ./outputs

# Explore a raw dataset directly: the app generates the snapshot first
python -m embedding_explorer_tool.webapp \
    --output_dir ./outputs_embedding_explorer \
    --data_root /path/to/triprong \
    --embedding dinov2 --embedding_model dinov2_vitb14_reg
```

If `--output_dir` has no snapshot yet, the app runs the same pre-filter +
quality + embedding stages as `run.py` to generate one from `--data_root`
before launching. `--embedding`/`--embedding_model` have the same choices and
default as `run.py` and are ignored once a snapshot already exists.

See [`docs/explorer.md`](docs/explorer.md) for layout, marker legend and controls.

### Arguments

| Argument | Default | Choices |
|----------|---------|---------|
| `--data_root` | `<repo>/../nit_object_onboarding/workspace/fmb_blocks/09_triprong` | Path to dataset |
| `--output_dir` | `outputs` | Output directory |
| `--num_views` | `10` | Number of views to select |
| `--embedding` | `auto` | `auto`, `dinov3`, `dinov2`, `siglip2`, `siglip`, `moonvit`, `clip`, `eva_clip` |
| `--embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model name or path; type inferred automatically when `--embedding=auto` |
| `--selector` | `quality_diversity` | `fps`, `quality_diversity`, `facility_location`, `dpp`, `next_best_view`, `top_kmeans_xnn` |
| `--selector_alpha` | `0.60` | Quality weight for GQD selector |
| `--selector_beta` | `0.40` | Diversity weight for GQD selector |
| `--kmeans_init` | `farthest` | k-means cluster-init for `top_kmeans_xnn`: `farthest` (farthest-point seeds) or `best_quality` (top-quality seeds) |
| `--kmeans_xnn_k` | `3` | xNN neighbourhood radius for `top_kmeans_xnn`: `3`, `5`, or `10` |
| `--filter_order` | config default | Comma-separated pre-filter order, e.g. `vincent_empty_mask,vincent_border_pixel,blur_laplacian,blur_tenengrad,vincents_artefacts`. Legacy filters available for custom orders: `border,area,occlusion,confidence,completeness` (not tested / likely not working as proper pre-filters) |
| `--use_shape_descriptors` | `False` | Use classical shape descriptors (CPU) |
| `--shape_descriptor` | `hu` | `hu`, `zernike`, `fourier`, `shape_context` |
| `--no-auto-thresholds` | `False` | Disable data-driven threshold tuning |

## Output Structure

```
outputs/
├── report.json            # Pipeline summary + selection metrics
├── quality.csv            # Per-observation quality metrics
├── embeddings.npy         # Embedding matrix (accepted pool)
├── selected_indices.npy   # Selected indices into embedding matrix
├── rejected.json          # Per-observation rejection reasons
├── rejected_metrics.csv   # Pre-filter raw metrics for rejected obs.
├── visualization.png      # Overview grid of selected views
│
├── selected_samples/      # Final selected tuples, re-organized by data type:
│   └── <obj_id>/          #   named exactly like the last component of data_root
│       ├── rgb/           #   selected object images
│       ├── mask/          #   selected object masks
│       ├── depth/         #   frame-wise depth (only when <data_root>/depth exists)
│       └── hand_mask/     #   hand masks (only when a hand mask is available)
│
├── accepted_samples/      # Accepted-but-unselected tuples (--debug), same layout
│   └── <obj_id>/
│       ├── rgb/
│       ├── mask/
│       └── hand_mask/
│
├── rejected_samples/      # Rejected tuples grouped by rejection reason:
│   └── <reason>/          #   e.g. vincent_border_pixel, blur_laplacian,
│       ├── threshold-based/  #   <reason>_threshold variants (below the
│       │   └── <obj_id>/     #   relaxed absolute floor)
│       │       ├── rgb/
│       │       ├── mask/
│       │       ├── depth/    #   only when <data_root>/depth exists
│       │       └── hand_mask/#   only when a hand mask is available
│       └── outlier-based/    #   <reason>_outlier variants (extreme bad
│           └── <obj_id>/     #   outliers relative to the population)
│               ├── rgb/
│               ├── mask/
│               ├── depth/
│               └── hand_mask/
│
├── bad_examples/          # Per-stage example frames (if --plot)
│   ├── pre-filter_stage/  # <feature>_filtered.png (reason-matched) or
│   │   └── {<feature>_filtered,lower_<feature>_quality}.png
│   └── selection_stage/   # lower_<feature>_quality.png (prob-sampled)
│       └── lower_<feature>_quality.png
│
└── plots/                 # Diagnostic plots (if --plot)
    ├── pre-filter/
    │   ├── violin_rejected_vs_accepted.png
    │   ├── violin_rejected_vs_accepted_scaled.png
    │   ├── pre_filter_raw_stats.png       # All pre-filter elements (accepted vs rejected)
    │   ├── pre_filter_soft_weights.png    # Vincent soft-filter population weights
    │   ├── rejection_reasons.png          # Occlusion and truncation kept as separate bars
    │   └── data_set_overview/             # Raw pre-filter stats, two variants each:
    │       └── <feature>_filter_{fixed,relative}.png
    │
    └── selection/
        ├── violin_*.png                   # Quality-score violins
        ├── data_set_overview/             # Quality scores, two variants each:
        │   └── quality_score_<feature>_{fixed,relative}.png
        │
        ├── embedding_space/               # DR of the embedding space
        │   ├── 2D_DR_plots/
        │   │   ├── selection_embedding.png        # PCA (jet)
        │   │   ├── selection_embedding_scaled.png # PCA (viridis)
        │   │   ├── embedding_mds.png
        │   │   ├── embedding_{tsne,umap,...}.png  # (debug only)
        │   │   └── clusters_embedding_<method>.png  # same coords, k-means colour
        │   └── 3D_DR_plots/
        │       ├── selection_embedding_3d.html    # PCA (plotly)
        │       ├── embedding_mds_3d.html
        │       ├── embedding_{tsne,umap,...}_3d.html  # (debug only)
        │       └── clusters_embedding_<method>_3d.html
        │
        ├── quality_criteria/              # DR of the normalised metric space
        │   └── DR_plots/
        │       ├── 2D_DR_plots/
        │       │   ├── selection_criteria.png / selection_criteria_scaled.png
        │       │   ├── criteria_{mds,tsne,umap,...}.png  # (debug only)
        │       │   └── clusters_criteria_<method>.png
        │       └── 3D_DR_plots/
        │           ├── selection_criteria_3d.html
        │           ├── criteria_{mds,tsne,umap,...}_3d.html  # (debug only)
        │           └── clusters_criteria_<method>_3d.html
        │
        └── debug (--debug only):
            ├── selected_neighbors_knn.png      # 5-NN of each selected view (embedding)
            ├── selected_neighbors_kmeans.png   # 5 neighbours from the selected view's k-means cluster
            ├── selected_clusters_pca.png       # PCA scatter coloured by k-means cluster
            └── embedded_samples/samples_<NN>.png  # original, mask, 224×224 cut-out on contrast bg, + original mask
```

## Configuration

All pipeline parameters are controlled via `config.py`. See [`docs/thresholds.md`](docs/thresholds.md) for the auto-tuning strategy and [`docs/pipeline.md`](docs/pipeline.md) for full module documentation.

### Pre-Filtering

The default pre-filter set (port from `nit_view_selection/select_best_views.py`, reworked for this pipeline) is deliberately small and conservative. The 5 default filters never hard-reject on their own; each one either has a **very relaxed absolute floor** (`threshold_min` in `config.py`) or removes **extreme bad outliers** relative to the population (`outlier_z`, fit on the accepted population via robust median/MAD statistics). Rejections are grouped under `<reason>/threshold-based/` and `<reason>/outlier-based/`.

- `vincent_empty_mask` → `VincentEmptyMaskFilter` (empty masks; pure hard reject)
- `vincent_border_pixel` → `VincentBorderPixelFilter` (masks touching the image frame; pure hard reject)
- `blur_laplacian` → `BorderLaplacianBlurFilter` (boundary-band Laplacian variance; default floor `0.01`, `outlier_z=3.0`)
- `blur_tenengrad` → `BorderTenengradBlurFilter` (boundary-band mean Sobel magnitude; default floor `0.10`, `outlier_z=3.0`)
- `vincents_artefacts` → `VincentsArtifactsFilter` (mask speckle/holes/ragged edges; default floor `0.05`, `outlier_z=3.0`)

Their raw stats (`laplacian`, `tenengrad`, `vincent_area_fraction`, `vincent_artifact_fraction`, `vincent_boundary_blur_variance`) are exported to `quality.csv` and `rejected_metrics.csv`.

The quality score combines exactly 4 components: boundary-blur (reads the `laplacian` pre-filter stat), mask-area, mask-artifacts, and centerness, weighted by `QualityWeights` (`blur` 0.30, `area` 0.20, `vincents_artefacts` 0.20, `centerness` 0.30). `confidence` is exported for diagnostics but not used by the scorer.

**Legacy filters** (`area`, `border`, `occlusion`, `confidence`, `completeness`) are kept only for custom `--filter_order` runs and are NOT part of the default set — they are not tested / likely not working as proper pre-filters. Two Vincent soft filters remain available as population-adapted weights: `VincentsAreaFilter` → `vincents_area` and `VincentsMotionBlurFilter` → `vincents_motion_blur` (both never reject).

## Testing

The project has **223 correctness test functions** (779 check assertions) and **55 smoke test checks** (including the new pre-filters, the 4-component quality scorer, and run.py wiring).

### Correctness Tests

Each filter, module, and selector is validated against **synthetic data** with known correct
outputs — no external labeled dataset needed:

```bash
# Run all correctness tests (delegates to tests/run_correctness.py)
python test_correctness.py
# or
python tests/run_correctness.py

# Expected output: Results: 223 passed, 0 failed out of 223
```

### Smoke Tests

Run the full pipeline against a real dataset to verify end-to-end integration:

```bash
python test_smoke.py --data_root /path/to/bottle
# or
python tests/run_smoke.py --data_root /path/to/bottle

# Expected output: Results: 55 passed, 0 failed out of 55
```

All suites must pass with **0 failures** before changes are considered complete.

## Project Structure

```
object_view_selection/
├── run.py                    # Pipeline entry point
├── config.py                 # All configuration dataclasses
│
├── data_io/
│   ├── observation.py        # Observation dataclass (image, mask, hand, quality, embedding)
│   ├── dataset.py            # Dataset loader (loads images/, masks/, object_hands/)
│   └── metrics.py            # ObservationMetrics dataclass
│
├── preprocessing/
│   ├── base.py               # Abstract BaseFilter
│   ├── filter_pipeline.py    # Chains filters, rejects on first failure
│   ├── variants.py           # FilterVariant: relaxed floor + robust-outlier rejection
│   ├── border_blur_filter.py # Default: boundary-band Laplacian + Tenengrad blur pre-filters
│   ├── vincents_artefacts.py # Default: mask artifact score pre-filter (never hard-rejects)
│   ├── vincent_utils.py      # Vincent helpers (boundary band, artifacts, robust scoring)
│   ├── vincent_empty_mask.py # Hard: reject empty masks
│   ├── vincent_border_pixel.py  # Hard: reject masks touching the frame
│   ├── vincents_base.py      # Abstract VincentSoftFilter (population-adapted)
│   ├── vincents_area_filter.py   # Soft: mask area weight
│   ├── vincents_motion_blur.py   # Soft: boundary-blur weight
│   └── (legacy, custom orders only) blur_filter.py, area_filter.py,
│       border_truncation.py, occlusion_filter.py, confidence.py,
│       completeness_filter.py
│
├── quality/
│   ├── base.py               # Abstract QualityMetric
│   ├── quality_scorer.py     # Weighted sum scorer
│   ├── blur.py               # BorderBlurQuality (boundary-band Laplacian, self-contained fallback)
│   ├── area.py               # AreaQuality (ratio up to 20%)
│   ├── centerness.py         # CenternessQuality (mask centredness)
│   ├── vincent.py            # VincentsArtifactsQuality (pass-through of artifact fraction)
│   └── (legacy, unused by scorer) occlusion.py, completeness.py, confidence.py
│
├── embeddings/
│   ├── base.py               # Abstract EmbeddingModel (+ optional RGBA alpha flag)
│   ├── crop.py               # Bbox/masked/square crops + contrast background input
│   ├── dinov3.py             # DINOv3 (ViT)
│   ├── dinov2.py             # DINOv2 (ViT)
│   ├── siglip2.py            # SigLIP2
│   ├── siglip.py             # SigLIP
│   ├── moonvit.py            # MoonViT
│   ├── clip.py               # OpenAI CLIP
│   └── eva_clip.py           # EVA-CLIP
│
├── descriptors/
│   ├── hu.py                 # Hu moments (7-dim)
│   ├── zernike.py            # Zernike moments (27-dim)
│   ├── fourier.py            # Fourier descriptors (32-dim)
│   └── shape_context.py      # Shape Context (60-dim)
│
├── selection/
│   ├── selector.py           # Abstract SubsetSelector
│   ├── fps.py                # FarthestPointSampling
│   ├── greedy_quality_diversity.py  # GQD (default)
│   ├── facility_location.py  # Facility Location
│   ├── dpp.py                # Determinantal Point Process
│   └── next_best_view.py     # Next Best View
│
├── utils/
│   ├── threshold_tuner.py    # Data-driven auto-thresholding
│   ├── visualization.py      # Overview grid saving
│   └── ...
│
├── plotting_process/              # Diagnostic plotting submodule
│   ├── wrapper.py                 # plot_all() + standalone CLI
│   ├── misc_plot.py               # Rejection-reasons bar chart
│   ├── pre_filter_plots.py        # Per-element pre-filter histograms
│   ├── feature_plots.py           # Per-feature overview + bad-example plots
│   ├── neighbor_plots.py          # Debug k-means / k-NN neighbour diagnostics
│   ├── embedding_plots/           # 2D/3D DR scatter plots
│   │   ├── base.py                # Shared scatter-drawing helpers
│   │   └── {pca,mds,tsne,...}.py  # One file per DR method
│   └── quality_score_plots/
│       └── violins.py             # Quality-score & pre-filter violins
│
├── tests/
│   ├── run_correctness.py    # Correctness test runner
│   ├── run_smoke.py          # Smoke test runner
│   ├── test_utils.py         # Shared helpers (make_circle_mask, make_flower, check)
│   ├── smoke_test_utils.py   # Smoke test helpers
│   ├── correctness_test_units/  # Test modules (779 checks)
│   └── smoke_test_units/        # Smoke test modules (55 checks)
│
├── embedding_explorer_tool/  # Interactive kMeans + xNN explorer
│   ├── algorithms.py         # Snapshot loading, seeds, k-means, xNN, MDS, overlay, text
│   ├── webapp_plotting.py    # Plotly 3D figure builder
│   ├── webapp.py             # Local HTTP server (/api/run, /composite/, ...)
│   ├── webapp_template.html  # Single-page frontend
│   ├── gui_tk.py             # tkinter + matplotlib mirror
│   └── static/plotly.min.js  # Vendored plotly bundle (offline)
│
├── docs/
│   ├── pipeline.md              # Detailed pipeline documentation
│   ├── plotting.md              # Plotting module reference
│   ├── selection_algorithms.md  # Selection algorithm deep-dive
│   ├── explorer.md              # Embedding explorer tool reference
│   └── thresholds.md            # Threshold auto-tuning reference
│
└── README.md
```

## Further Reading

- [`docs/pipeline.md`](docs/pipeline.md) — Detailed module descriptions, algorithm explanations, configuration reference
- [`docs/thresholds.md`](docs/thresholds.md) — Auto-tuning strategy, safety limits, percentile rules, override instructions
- [`docs/plotting.md`](docs/plotting.md) — Diagnostic plot reference, output structure, standalone usage
- [`docs/selection_algorithms.md`](docs/selection_algorithms.md) — Selection algorithm deep-dive with pseudocode and comparison
- [`docs/explorer.md`](docs/explorer.md) — Embedding explorer tool reference (web app + tkinter)
