# Object View Selection

Select the best **N image/mask pairs** that maximize **object identifiability** — high-quality, diverse, non-redundant viewpoints.

## Pipeline

```
Dataset → Pre-filter → Quality Score → Embeddings → Subset Selection → Outputs
```

| Stage | What it does |
|-------|-------------|
| **Pre-filter** | Rejects blurry, truncated, occluded, or tiny observations |
| **Quality Score** | Weighted combination of blur, area, occlusion, confidence, completeness |
| **Embeddings** | DINOv3 / DINOv2 / SigLIP / CLIP / EVA-CLIP features (or shape descriptors) |
| **Selection** | Greedy quality+diversity, FPS, Facility Location, DPP, or NBV |

## Installation

```bash
git clone <repo> && cd object_view_selection

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
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
# DINOv3 embeddings (default, requires torch + transformers)
python run.py --data_root /path/to/bottle --num_views 10

# Switch DINOv3 model variant (auto-detected from model name)
python run.py --data_root /path/to/bottle --embedding_model facebook/dinov3-vitl16-pretrain-lvd1689m
python run.py --data_root /path/to/bottle --embedding_model facebook/dinov3-vits16-pretrain-lvd1689m

# SigLIP2 or MoonViT (auto-detected)
python run.py --data_root /path/to/bottle --embedding_model google/siglip2-base-patch16-224
python run.py --data_root /path/to/bottle --embedding_model moonshotai/MoonViT-SO-400M

# Explicitly override type (e.g. fall back to DINOv2)
python run.py --data_root /path/to/bottle --embedding dinov2 --embedding_model dinov2_vitb14_reg

# Shape descriptors (no GPU needed, no torch required)
python run.py --data_root /path/to/bottle --use_shape_descriptors --num_views 10
python run.py --data_root /path/to/bottle --use_shape_descriptors --shape_descriptor zernike

# Custom selector
python run.py --data_root /path/to/bottle --selector dpp --num_views 10
```

### Arguments

| Argument | Default | Choices |
|----------|---------|---------|
| `--data_root` | `""` | Path to dataset |
| `--output_dir` | `outputs` | Output directory |
| `--num_views` | `10` | Number of views to select |
| `--embedding` | `auto` | `auto`, `dinov3`, `dinov2`, `siglip2`, `siglip`, `moonvit`, `clip`, `eva_clip` |
| `--embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model name or path; type inferred automatically when `--embedding=auto` |
| `--selector` | `quality_diversity` | `fps`, `quality_diversity`, `facility_location`, `dpp`, `next_best_view` |
| `--use_shape_descriptors` | `False` | Use classical shape descriptors |
| `--shape_descriptor` | `hu` | `hu`, `zernike`, `fourier`, `shape_context` |

## Output Structure

```
outputs/
├── selected/              # Selected images
├── selected_masks/        # Corresponding masks
├── selected_object_hands/ # Corresponding hand masks
├── rejected/              # Rejected images (if save_rejected=True)
├── rejected_masks/        # Rejected masks
├── report.json            # Pipeline summary
├── quality.csv            # Per-observation metrics
├── embeddings.npy         # Embedding matrix
├── selected_indices.npy   # Selected indices
├── rejected.json          # Rejection reasons
└── visualization.png      # Overview grid
```

## Configuration

All pipeline parameters are controlled via `config.py`. Key settings:

```python
PipelineConfig(
    filters=FilterConfig(
        filter_order=["border", "area", "confidence", "blur", "occlusion", "completeness"],
        blur=BlurConfig(threshold=120.0),
        area=AreaConfig(minimum_ratio=0.01),
        border=BorderConfig(maximum_ratio=0.05),
        occlusion=OcclusionConfig(maximum_overlap=0.15),
    ),
    embedding="dinov3",
    embedding_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    selector="quality_diversity",
    num_views=10,
    quality_weights=QualityWeights(
        blur=0.25, area=0.20, occlusion=0.20, completeness=0.25
    ),
)
```

## Testing

Each filter, module, and selector is validated against synthetic data with known correct
outputs (49 tests). The test data is **generated programmatically** in
`test_correctness.py` — no external labeled dataset is needed:

- **Blur filter**: sharp circle vs. Gaussian-blurred circle — sharp has higher Laplacian/Tenengrad
- **Area filter**: 50%-area mask passes, 1%-area mask fails a 2% threshold
- **Border filter**: centered vs. edge-touching masks
- **Occlusion filter**: non-overlapping vs. overlapping hand masks
- **Confidence filter**: confidence above/below threshold
- **Completeness filter**: solid vs. perforated mask
- **Filters pipeline**: all six filters in the correct order
- **Quality scorer**: weighted combination with known values
- **Embedding models**: DINOv3, SigLIP2, DINOv2, SigLIP, CLIP, EVA-CLIP encode random input to correct dimension
- **Shape descriptors**: Hu, Zernike, Fourier, Shape Context return fixed-size vectors
- **Selectors**: FPS, quality+diversity, facility location, DPP, NBV produce exactly N views
- **Metrics dataclass**: defaults and setters

```bash
# Run all correctness tests
python test_correctness.py

# Expected output: 49 passed, 0 failed out of 49
```

## Project Structure

```
object_view_selection/
├── run.py              # Pipeline entry point
├── config.py           # Configuration
├── data_io/            # Dataset loader + Observation dataclass
├── preprocessing/      # Blur, truncation, area, occlusion, confidence, completeness
├── quality/            # Quality metrics + weighted scorer
├── embeddings/         # DINOv2, SigLIP, CLIP, EVA-CLIP
├── descriptors/        # Hu, Zernike, Fourier, Shape Context
├── selection/          # FPS, Quality+Diversity, Facility Location, DPP, NBV
└── utils/              # Geometry, math, visualization
```
