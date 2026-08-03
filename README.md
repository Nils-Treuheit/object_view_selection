# Object View Selection

Select the best **N image/mask pairs** that maximize **object identifiability** — high-quality, diverse, non-redundant viewpoints of a single object.

```
Dataset → Auto-Threshold → Pre-Filter → Quality Score → Embeddings → Subset Selection → Outputs
```

| Stage | What it does |
|-------|-------------|
| **Auto-Threshold** | Computes data-driven filter thresholds from dataset statistics (percentile + safety clamp) |
| **Pre-filter** | Rejects blurry, truncated, occluded, tiny, or incomplete observations (6 filter modules) |
| **Quality Score** | Weighted combination of blur, area, occlusion, completeness scores |
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

# Disable auto-threshold tuning (use static config values instead)
python run.py --data_root /path/to/bottle --num_views 10 --no-auto-thresholds

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
```

### Generating Plots

Use the `--plot` flag to generate diagnostic plots after selection:

```bash
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs --plot

# With debug mode (all DR methods, not just PCA + MDS):
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs --plot --debug
```

Plots are saved under `outputs/plots/`. See [`docs/plotting.md`](docs/plotting.md) for a complete reference.

### Standalone Plotting

Re-generate plots from a previous pipeline run without re-running the pipeline:

```bash
python -m plotting_process.wrapper --input_dir ./outputs [--output_dir ./plots] [--debug]
```

If `--output_dir` is omitted, the `plots/` folder is created inside `--input_dir`.

### Arguments

| Argument | Default | Choices |
|----------|---------|---------|
| `--data_root` | `""` | Path to dataset |
| `--output_dir` | `outputs` | Output directory |
| `--num_views` | `10` | Number of views to select |
| `--embedding` | `auto` | `auto`, `dinov3`, `dinov2`, `siglip2`, `siglip`, `moonvit`, `clip`, `eva_clip` |
| `--embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model name or path; type inferred automatically when `--embedding=auto` |
| `--selector` | `quality_diversity` | `fps`, `quality_diversity`, `facility_location`, `dpp`, `next_best_view` |
| `--selector_alpha` | `0.4` | Quality weight for GQD selector |
| `--selector_beta` | `0.6` | Diversity weight for GQD selector |
| `--use_shape_descriptors` | `False` | Use classical shape descriptors (CPU) |
| `--shape_descriptor` | `hu` | `hu`, `zernike`, `fourier`, `shape_context` |
| `--no-auto-thresholds` | `False` | Disable data-driven threshold tuning |

## Output Structure

```
outputs/
├── selected/              # Selected images (PNG)
├── selected_masks/        # Corresponding masks
├── selected_object_hands/ # Corresponding hand masks (if available)
├── rejected/              # Rejected images (if save_rejected=True)
├── rejected_masks/        # Rejected masks
├── report.json            # Pipeline summary + selection metrics
├── quality.csv            # Per-observation quality metrics
├── embeddings.npy         # Embedding matrix (accepted pool)
├── selected_indices.npy   # Selected indices into embedding matrix
├── rejected.json          # Per-observation rejection reasons
├── rejected_metrics.csv   # Pre-filter raw metrics for rejected obs.
├── visualization.png      # Overview grid of selected views
│
└── plots/                 # Diagnostic plots (if --plot)
    ├── pre-filter/
    │   ├── violin_rejected_vs_accepted.png
    │   ├── violin_rejected_vs_accepted_scaled.png
    │   └── rejection_reasons.png
    │
    └── selection/
        ├── violin_*.png                   # Quality-score violins
        │
        ├── 2D_DR_plots/
        │   ├── selection_embedding.png    # PCA (jet)
        │   ├── selection_embedding_scaled.png  # PCA (viridis)
        │   ├── embedding_mds.png
        │   └── embedding_{tsne,umap,...}.png   # (debug only)
        │
        └── 3D_DR_plots/
            ├── selection_embedding_3d.html     # PCA (plotly)
            ├── embedding_mds_3d.html
            └── embedding_{tsne,umap,...}_3d.html  # (debug only)
```

## Configuration

All pipeline parameters are controlled via `config.py`. See [`docs/thresholds.md`](docs/thresholds.md) for the auto-tuning strategy and [`docs/pipeline.md`](docs/pipeline.md) for full module documentation.

## Testing

The project has **101 correctness tests** (86 original + 15 plotting) and **51 smoke tests**.

### Correctness Tests

Each filter, module, and selector is validated against **synthetic data** with known correct
outputs — no external labeled dataset needed:

```bash
# Run all correctness tests
python tests/run_correctness.py

# Expected output: 101 passed, 0 failed out of 101
```

### Smoke Tests

Run the full pipeline against a real dataset to verify end-to-end integration:

```bash
python tests/run_smoke.py --data_root /path/to/bottle
```

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
│   ├── blur_filter.py        # Laplace + Tenengrad sharpness
│   ├── area_filter.py        # Min object size
│   ├── border_truncation.py  # Edge-touching detection
│   ├── occlusion_filter.py   # Hand/mask overlap
│   ├── confidence.py         # Detector confidence gate (disabled by default)
│   └── completeness_filter.py# Solidity, extent, convexity
│
├── quality/
│   ├── base.py               # Abstract QualityMetric
│   ├── quality_scorer.py     # Weighted sum scorer
│   ├── blur.py               # BlurQuality (normalized by 2× threshold)
│   ├── area.py               # AreaQuality (ratio up to 20%)
│   ├── occlusion.py          # OcclusionQuality (1 - overlap)
│   ├── completeness.py       # CompletenessQuality (pass-through)
│   └── confidence.py         # ConfidenceQuality (pass-through, unused by scorer)
│
├── embeddings/
│   ├── base.py               # Abstract EmbeddingModel
│   ├── crop.py               # Bbox / masked / square cropping
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
│   ├── correctness_test_units/  # Test modules (101 tests)
│   └── smoke_test_units/        # Smoke test modules (51 tests)
│
├── docs/
│   ├── pipeline.md              # Detailed pipeline documentation
│   ├── plotting.md              # Plotting module reference
│   ├── selection_algorithms.md  # Selection algorithm deep-dive
│   └── thresholds.md            # Threshold auto-tuning reference
│
└── README.md
```

## Further Reading

- [`docs/pipeline.md`](docs/pipeline.md) — Detailed module descriptions, algorithm explanations, configuration reference
- [`docs/thresholds.md`](docs/thresholds.md) — Auto-tuning strategy, safety limits, percentile rules, override instructions
- [`docs/plotting.md`](docs/plotting.md) — Diagnostic plot reference, output structure, standalone usage
- [`docs/selection_algorithms.md`](docs/selection_algorithms.md) — Selection algorithm deep-dive with pseudocode and comparison
