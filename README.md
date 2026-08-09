# Object View Selection

Select the best **N image/mask pairs** that maximize **object identifiability** — high-quality, diverse, non-redundant viewpoints of a single object.

```
Dataset → Auto-Threshold → Pre-Filter → Quality Score → Embeddings → Subset Selection → Outputs
```

| Stage | What it does |
|-------|-------------|
| **Auto-Threshold** | Tunes legacy filters via dataset statistics; the default blur/artifact pre-filters use static relaxed floors + population-relative outlier rejection |
| **Pre-filter** | 5 conservative filters: empty mask, frame-touching mask, boundary-band Laplacian blur, boundary-band Tenengrad blur, mask artifacts |
| **Quality Score** | Weighted combination of 4 components: boundary-blur, mask-area, mask-artifacts, centerness |
| **Embeddings** | DINOv3 / DINOv2 / SigLIP2 / SigLIP / MoonViT / CLIP / EVA-CLIP features (or classical shape descriptors on CPU) |
| **Selection** | Greedy quality+diversity (default), FPS, Facility Location, DPP, Next-Best-View, or top-kMeans-xNN |

## Installation

```bash
git clone <repo> && cd object_view_selection

python3 -m venv .venv && source .venv/bin/activate

uv sync                  # or: pip install -r requirements.txt
```

Optional extras for learned embeddings: `torch torchvision timm`, `transformers` (DINOv3/SigLIP), `open-clip-torch` (EVA-CLIP), `git+https://github.com/openai/CLIP.git`.

## Usage

Minimum command:

```bash
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs
```

`--data_root` must contain `images/` and `masks/` (binary masks, same filenames), plus optional `object_hands/` and `depth/`:

```
bottle/
├── images/          # 00000.png, 00001.png, ...
├── masks/           # 00000.png, 00001.png, ...  (binary, same filename)
├── object_hands/    # optional, binary hand occlusion
└── depth/           # optional, copied through to outputs
```

### Key parameters

| Argument | Default | Choices / notes |
|----------|---------|-----------------|
| `--num_views` | `10` | Number of views to select |
| `--embedding` | `auto` | `auto`, `dinov3`, `dinov2`, `siglip2`, `siglip`, `moonvit`, `clip`, `eva_clip` |
| `--embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model name/path; type inferred when `--embedding=auto` |
| `--selector` | `quality_diversity` | `fps`, `quality_diversity`, `facility_location`, `dpp`, `next_best_view`, `top_kmeans_xnn` |
| `--selector_alpha` / `--selector_beta` | `0.60` / `0.40` | Quality / diversity weights (GQD) |
| `--kmeans_init` | `farthest` | `farthest` or `best_quality` (top_kmeans_xnn) |
| `--kmeans_xnn_k` | `3` | `3`, `5`, or `10` (top_kmeans_xnn) |
| `--use_shape_descriptors` | off | CPU shape descriptors (`hu`, `zernike`, `fourier`, `shape_context`) |
| `--filter_order` | config default | Comma-separated pre-filter order; legacy filters `border,area,occlusion,confidence,completeness` for custom orders only |
| `--no-auto-thresholds` | off | Use static config thresholds |
| `--plot` / `--debug` | off | Generate diagnostic plots / verbose per-step stats |
| `--only_pre_filter` | off | Stop after pre-filtering (dump accepted/rejected samples + `rejected.json`) |

Examples:

```bash
python run.py --data_root /path/to/bottle --embedding dinov2 --embedding_model dinov2_vitb14_reg
python run.py --data_root /path/to/bottle --selector dpp --num_views 10
python run.py --data_root /path/to/bottle --use_shape_descriptors --shape_descriptor zernike
python run.py --data_root /path/to/bottle --plot --debug
```

### Outputs

Results are written to `--output_dir`: `report.json`, `quality.csv`, `rejected.json`, `rejected_metrics.csv`, `embeddings.npy` (+ `selected_indices.npy`, `selection_pool_ids.npy`), `selected_samples/`, `rejected_samples/`, `accepted_samples/` (`--debug`), `visualization.png`, and optional `plots/`, `bad_examples/`, `embedded_samples/`. See [`docs/pipeline.md`](docs/pipeline.md) for the full structure.

### Related tools

```bash
# Diagnostic plots from a previous run (or --plot during the run)
python -m plotting_process.wrapper --input_dir ./outputs

# Interactive 3D explorer of the kMeans + xNN selection pool
python -m embedding_explorer_tool.webapp --output_dir ./outputs

# Tune the pre-filter thresholds on a dataset, preview the accept/reject
# outcome, then run the embedding (feeds the explorer's snapshot)
python -m embedding_explorer_tool.prefilter_app

# Or start both together (tuner on 8520, explorer on 8510)
python run_webapps.py -i /path/to/dataset -o ./outputs_embedding_explorer
```

## Testing

Run all correctness tests (synthetic data) and smoke tests (real dataset):

```bash
python test_correctness.py        # or: python tests/run_correctness.py
python test_smoke.py --data_root /path/to/bottle   # or: tests/run_smoke.py
```

See [`docs/testing.md`](docs/testing.md) for details and counts.

## Project Structure

```
object_view_selection/
├── run.py                  # Pipeline entry point (argparse)
├── config.py               # All configuration dataclasses
├── data_io/                # Observation/dataset loaders + metrics
├── preprocessing/          # Pre-filters: ScoreFilter/OutlierFilter rejection + legacy
├── quality/                # 4-component weighted quality scorer
├── embeddings/             # Learned embedding models + crop helpers
├── descriptors/            # CPU shape descriptors (hu/zernike/fourier/shape_context)
├── selection/              # 6 subset-selection algorithms
├── utils/                  # Threshold tuner, visualization helpers
├── plotting_process/       # Diagnostic plotting (run.py + standalone wrapper)
├── embedding_explorer_tool/# kMeans/xNN explorer (web) + pre-filter threshold tuner (web)
├── tests/                  # Correctness + smoke test suites
└── docs/                   # Reference documentation (see below)
```

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md) — Pipeline stages, output structure, configuration reference
- [`docs/pre-filter/`](docs/pre-filter/README.md) — Pre-filter algorithms, metrics, rejection behaviour
- [`docs/scoring.md`](docs/scoring.md) — Quality scoring components
- [`docs/thresholds.md`](docs/thresholds.md) — Auto-threshold tuning strategy
- [`docs/selection_algorithms.md`](docs/selection_algorithms.md) — Selection algorithm deep-dive
- [`docs/plotting.md`](docs/plotting.md) — Diagnostic plots and standalone usage
- [`docs/explorer.md`](docs/explorer.md) — Embedding explorer (web) reference
- [`docs/testing.md`](docs/testing.md) — Test suites and how to run them
