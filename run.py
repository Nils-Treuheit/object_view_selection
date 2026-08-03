import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from config import PipelineConfig
from data_io.dataset import Dataset
from preprocessing.area_filter import AreaFilter
from preprocessing.blur_filter import BlurFilter
from preprocessing.border_truncation import BorderFilter
from preprocessing.completeness_filter import CompletenessFilter
from preprocessing.confidence import ConfidenceFilter
from preprocessing.filter_pipeline import FilterPipeline
from preprocessing.occlusion_filter import OcclusionFilter
from quality.area import AreaQuality
from quality.blur import BlurQuality
from quality.completeness import CompletenessQuality
from quality.occlusion import OcclusionQuality
from quality.quality_scorer import QualityScorer


def build_filters(cfg: PipelineConfig, tuned=None):
    if tuned is None:
        tuned = {}

    available = {
        "blur": BlurFilter(
            laplacian_threshold=tuned.get("laplacian_threshold", cfg.filters.blur.threshold),
            tenengrad_threshold=tuned.get("tenengrad_threshold", cfg.filters.blur.tenengrad_threshold),
            enabled=cfg.filters.blur.enabled,
        ),
        "area": AreaFilter(
            minimum_ratio=tuned.get("area_minimum_ratio", cfg.filters.area.minimum_ratio),
            enabled=cfg.filters.area.enabled,
        ),
        "border": BorderFilter(
            maximum_ratio=tuned.get("border_maximum_ratio", cfg.filters.border.maximum_ratio),
            edge_maximum_ratio=tuned.get("border_edge_maximum_ratio", cfg.filters.border.edge_maximum_ratio),
            enabled=cfg.filters.border.enabled,
        ),
        "occlusion": OcclusionFilter(
            maximum_overlap=tuned.get("occlusion_maximum_overlap", cfg.filters.occlusion.maximum_overlap),
            enabled=cfg.filters.occlusion.enabled,
        ),
        "confidence": ConfidenceFilter(
            minimum_confidence=cfg.filters.confidence.minimum_confidence,
            enabled=cfg.filters.confidence.enabled,
        ),
        "completeness": CompletenessFilter(
            minimum_score=tuned.get("completeness_minimum_score", cfg.filters.completeness.minimum_score),
            enabled=cfg.filters.completeness.enabled,
        ),
    }
    filters = []
    for name in cfg.filters.filter_order:
        if name in available:
            filters.append(available[name])
    return FilterPipeline(filters)


def build_quality_scorer(cfg: PipelineConfig, tuned=None):
    if tuned is None:
        tuned = {}
    metrics = []
    weights = {}
    if cfg.filters.blur.enabled:
        lap_thresh = tuned.get("laplacian_threshold", cfg.filters.blur.threshold)
        metrics.append(BlurQuality(max_lap=lap_thresh * 2))
        weights["blur"] = cfg.quality_weights.blur
    if cfg.filters.area.enabled:
        metrics.append(AreaQuality())
        weights["area"] = cfg.quality_weights.area
    if cfg.filters.occlusion.enabled:
        metrics.append(OcclusionQuality())
        weights["occlusion"] = cfg.quality_weights.occlusion
    if cfg.filters.completeness.enabled:
        metrics.append(CompletenessQuality())
        weights["completeness"] = cfg.quality_weights.completeness
    return QualityScorer(metrics, weights)


def infer_embedding_type(model_name: str) -> str:
    """Infer the embedding type from a model name or path."""
    name_lower = model_name.lower()
    if "dinov3" in name_lower or name_lower.startswith("dinov3_"):
        return "dinov3"
    if "dinov2" in name_lower or name_lower.startswith("dinov2_"):
        return "dinov2"
    if "siglip2" in name_lower:
        return "siglip2"
    if "siglip" in name_lower and "siglip2" not in name_lower:
        return "siglip"
    if "moonvit" in name_lower:
        return "moonvit"
    if "clip" in name_lower and "eva" not in name_lower:
        return "clip"
    if name_lower.startswith("vit-"):
        return "clip"
    if "eva" in name_lower:
        return "eva_clip"
    if name_lower.startswith("vit_"):
        return "dinov2"
    raise ValueError(f"Cannot infer embedding type from model name: {model_name}")


def build_embedding_model(cfg: PipelineConfig):
    if cfg.use_shape_descriptors:
        return None

    embedding_type = cfg.embedding
    variant = cfg.embedding_model
    if embedding_type == "auto":
        embedding_type = infer_embedding_type(variant)

    if embedding_type == "dinov3":
        from embeddings.dinov3 import DINOv3Embedding
        try:
            return DINOv3Embedding(model_name=variant)
        except Exception as e:
            print(f"Warning: DINOv3 failed to load ({e}). Falling back to DINOv2.")
            embedding_type = "dinov2"
            variant = "dinov2_vitb14_reg"
    if embedding_type == "dinov2":
        from embeddings.dinov2 import DINOv2Embedding
        return DINOv2Embedding(model_name=variant)
    elif embedding_type == "siglip2":
        from embeddings.siglip2 import SigLIP2Embedding
        return SigLIP2Embedding(model_name=variant)
    elif embedding_type == "siglip":
        from embeddings.siglip import SigLIPEmbedding
        return SigLIPEmbedding(model_name=variant)
    elif embedding_type == "moonvit":
        from embeddings.moonvit import MoonViTEmbedding
        return MoonViTEmbedding(model_name=variant)
    elif embedding_type == "clip":
        from embeddings.clip import CLIPEmbedding
        return CLIPEmbedding(model_name=variant)
    elif embedding_type == "eva_clip":
        from embeddings.eva_clip import EvaCLIPEmbedding
        return EvaCLIPEmbedding(model_name=variant)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def extract_shape_descriptor(observation, descriptor_type: str) -> np.ndarray:
    mask = observation.mask

    if descriptor_type == "hu":
        from descriptors.hu import hu_moments
        return hu_moments(mask)
    elif descriptor_type == "zernike":
        from descriptors.zernike import zernike_moments
        return zernike_moments(mask)
    elif descriptor_type == "fourier":
        from descriptors.fourier import fourier_descriptors
        return fourier_descriptors(mask)
    elif descriptor_type == "shape_context":
        from descriptors.shape_context import shape_context_descriptor
        return shape_context_descriptor(mask)
    else:
        raise ValueError(f"Unknown shape descriptor: {descriptor_type}")


def build_selector(cfg: PipelineConfig):
    name = cfg.selector
    if name == "fps":
        from selection.fps import FarthestPointSampling
        return FarthestPointSampling()
    elif name == "quality_diversity":
        from selection.greedy_quality_diversity import GreedyQualityDiversity
        return GreedyQualityDiversity(alpha=cfg.selector_alpha, beta=cfg.selector_beta)
    elif name == "facility_location":
        from selection.facility_location import FacilityLocation
        return FacilityLocation()
    elif name == "dpp":
        from selection.dpp import DPPSelector
        return DPPSelector(sigma=cfg.dpp_sigma)
    elif name == "next_best_view":
        from selection.next_best_view import NextBestView
        return NextBestView()
    else:
        raise ValueError(f"Unknown selector: {name}")


def run_pipeline(cfg: PipelineConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected").mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_object_hands").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected_masks").mkdir(parents=True, exist_ok=True)

    dataset = Dataset(cfg.data_root)
    dataset.load_images()
    print(f"Loaded {len(dataset)} observations")

    tuned = {}
    if cfg.auto_thresholds:
        from utils.threshold_tuner import tune_thresholds
        print("Computing data-driven thresholds...")
        tuned = tune_thresholds(dataset.observations)
        print(f"  area_minimum_ratio={tuned['area_minimum_ratio']}")
        print(f"  border_maximum_ratio={tuned['border_maximum_ratio']}")
        print(f"  border_edge_maximum_ratio={tuned['border_edge_maximum_ratio']}")
        print(f"  laplacian_threshold={tuned['laplacian_threshold']}")
        print(f"  tenengrad_threshold={tuned['tenengrad_threshold']}")
        print(f"  occlusion_maximum_overlap={tuned['occlusion_maximum_overlap']}")
        print(f"  completeness_minimum_score={tuned['completeness_minimum_score']}")

    filter_pipeline = build_filters(cfg, tuned)
    quality_scorer = build_quality_scorer(cfg, tuned)
    embedding_model = build_embedding_model(cfg)
    selector = build_selector(cfg)

    accepted = []
    rejected = []

    for obs in tqdm(dataset.observations, desc="Pre-filtering"):
        if not filter_pipeline.run(obs):
            rejected.append(obs)
            continue
        accepted.append(obs)

    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    if cfg.debug:
        rejection_counts = {}
        for obs in rejected:
            r = obs.rejection_reason or "unknown"
            rejection_counts[r] = rejection_counts.get(r, 0) + 1
        print("  Rejection breakdown:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        if accepted:
            raw_metrics = ["laplacian", "tenengrad", "area_ratio", "border_ratio", "edge_ratio", "hand_overlap", "completeness"]
            print("  Accepted raw metrics:")
            for key in raw_metrics:
                vals = np.array([getattr(o.metrics, key, 0) for o in accepted])
                print(f"    {key}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")

    if len(accepted) == 0:
        print("No observations passed filtering. Exiting.")
        return

    for obs in tqdm(accepted, desc="Scoring quality"):
        quality_scorer.score(obs)

    for obs in accepted:
        obs.metrics.confidence = min(
            obs.metrics.blur,
            obs.metrics.area,
            obs.metrics.occlusion,
            obs.metrics.completeness,
        )

    if cfg.debug:
        quality_keys = ["blur", "area", "occlusion", "completeness", "confidence"]
        print("  Quality scores:")
        for key in quality_keys:
            vals = np.array([getattr(o.metrics, key, 0) for o in accepted])
            print(f"    {key}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")
        qvals = np.array([o.quality for o in accepted])
        print(f"    score:  min={qvals.min():.4f}  max={qvals.max():.4f}  mean={qvals.mean():.4f}  median={np.median(qvals):.4f}")

    if cfg.use_shape_descriptors or embedding_model is None:
        for obs in tqdm(accepted, desc="Extracting descriptors"):
            feat = extract_shape_descriptor(obs, cfg.shape_descriptor)
            obs.embedding = feat
    else:
        for obs in tqdm(accepted, desc="Extracting embeddings"):
            obs.embedding = embedding_model.encode(obs.image, obs.mask)

    embeddings = np.array([obs.embedding for obs in accepted])
    quality_scores = np.array([obs.quality for obs in accepted])

    selected_idx = selector.select(
        embeddings=embeddings,
        quality_scores=quality_scores,
        n=cfg.num_views,
    )

    selected = [accepted[i] for i in selected_idx]
    print(f"Selected {len(selected)} views")

    if cfg.debug:
        sel_qual = quality_scores[selected_idx]
        print(f"  Selected quality: min={sel_qual.min():.4f}  max={sel_qual.max():.4f}  mean={sel_qual.mean():.4f}")
        print(f"  Pool quality:     min={quality_scores.min():.4f}  max={quality_scores.max():.4f}  mean={quality_scores.mean():.4f}")

    selected_set = {s.id for s in selected}

    quality_csv = []
    for i, obs in enumerate(accepted):
        row = {
            "id": obs.id,
            "quality": obs.quality,
            "score": obs.quality,
            "laplacian": obs.metrics.laplacian,
            "tenengrad": obs.metrics.tenengrad,
            "area_ratio": obs.metrics.area_ratio,
            "border_ratio": obs.metrics.border_ratio,
            "border_free": 1.0 - obs.metrics.border_ratio,
            "edge_ratio": obs.metrics.edge_ratio,
            "hand_overlap": obs.metrics.hand_overlap,
            "solidity": obs.metrics.solidity,
            "extent": obs.metrics.extent,
            "convexity": obs.metrics.convexity,
            "completeness": obs.metrics.completeness,
            "blur": obs.metrics.blur,
            "area": obs.metrics.area,
            "occlusion": obs.metrics.occlusion,
            "confidence": obs.metrics.confidence,
            "selected": obs.id in selected_set,
        }
        quality_csv.append(row)

    import pandas as pd
    df = pd.DataFrame(quality_csv)
    df.to_csv(output_dir / "quality.csv", index=False)

    if cfg.save_embeddings:
        np.save(output_dir / "embeddings.npy", embeddings)
        np.save(output_dir / "selected_indices.npy", selected_idx)

    for obs in selected:
        stem = f"{obs.id:05d}"
        cv2.imwrite(str(output_dir / "selected" / f"{stem}.png"),
                     cv2.cvtColor(obs.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "selected_masks" / f"{stem}.png"), obs.mask)
        if obs.object_hand is not None:
            cv2.imwrite(str(output_dir / "selected_object_hands" / f"{stem}.png"), obs.object_hand)

    if cfg.save_rejected:
        for obs in rejected:
            stem = f"{obs.id:05d}"
            cv2.imwrite(str(output_dir / "rejected" / f"{stem}.png"),
                         cv2.cvtColor(obs.image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / "rejected_masks" / f"{stem}.png"), obs.mask)

    rejected_data = [
        {"id": obs.id, "reason": obs.rejection_reason}
        for obs in rejected
    ]
    with open(output_dir / "rejected.json", "w") as f:
        json.dump(rejected_data, f, indent=2)

    rejected_metrics_csv = []
    for obs in rejected:
        m = obs.metrics
        rejected_metrics_csv.append({
            "id": obs.id,
            "laplacian": m.laplacian,
            "tenengrad": m.tenengrad,
            "area_ratio": m.area_ratio,
            "border_ratio": m.border_ratio,
            "edge_ratio": m.edge_ratio,
            "hand_overlap": m.hand_overlap,
            "completeness": m.completeness,
        })
    pd.DataFrame(rejected_metrics_csv).to_csv(output_dir / "rejected_metrics.csv", index=False)

    effective_embedding = cfg.embedding
    if effective_embedding == "auto":
        effective_embedding = infer_embedding_type(cfg.embedding_model)

    # --- selection metrics ---
    from sklearn.metrics import pairwise_distances
    dist = pairwise_distances(embeddings, metric="cosine")
    sel_dist = dist[np.ix_(selected_idx, selected_idx)]
    n_sel = len(selected_idx)
    triu_sel = sel_dist[np.triu_indices(n_sel, k=1)]
    sel_qual = quality_scores[selected_idx]

    non_sel_mask = np.ones(len(embeddings), dtype=bool)
    non_sel_mask[selected_idx] = False
    non_sel_idx = np.where(non_sel_mask)[0]
    coverage_dists = dist[non_sel_idx][:, selected_idx].min(axis=1) if len(non_sel_idx) > 0 else np.array([])

    selection_log = []
    if cfg.selector == "quality_diversity":
        from selection.greedy_quality_diversity import GreedyQualityDiversity
        s = GreedyQualityDiversity(alpha=cfg.selector_alpha, beta=cfg.selector_beta)
        steps = []
        pool = set(range(len(embeddings)))
        first = int(quality_scores.argmax())
        steps.append(first)
        selection_log.append({
            "step": 0, "id": int(accepted[first].id),
            "quality": float(quality_scores[first]),
            "min_cosine_dist_to_set": None, "score": None,
        })
        pool.remove(first)
        while len(steps) < len(selected_idx):
            best_score = -np.inf
            best_i = -1
            for i in pool:
                diversity = dist[i, steps].min()
                score = cfg.selector_alpha * quality_scores[i] + cfg.selector_beta * diversity
                if score > best_score:
                    best_score = score
                    best_i = i
            steps.append(best_i)
            pool.remove(best_i)
            selection_log.append({
                "step": len(steps)-1, "id": int(accepted[best_i].id),
                "quality": float(quality_scores[best_i]),
                "min_cosine_dist_to_set": float(dist[best_i, steps[:-1]].min()),
                "score": float(best_score),
            })

    selection_metrics = {
        "selector": cfg.selector,
        "num_views": cfg.num_views,
        "selected_count": n_sel,
        "intra_set": {
            "mean_pairwise_cosine_distance": float(triu_sel.mean()) if len(triu_sel) > 0 else 0.0,
            "min_pairwise_cosine_distance": float(triu_sel.min()) if len(triu_sel) > 0 else 0.0,
            "max_pairwise_cosine_distance": float(triu_sel.max()) if len(triu_sel) > 0 else 0.0,
            "mean_similarity": float((1 - triu_sel).mean()) if len(triu_sel) > 0 else 0.0,
        },
        "quality": {
            "selected_mean": float(sel_qual.mean()),
            "selected_min": float(sel_qual.min()),
            "selected_max": float(sel_qual.max()),
            "pool_mean": float(quality_scores.mean()),
        },
        "coverage": {
            "mean_min_cosine_dist_to_selected": float(coverage_dists.mean()) if len(coverage_dists) > 0 else 0.0,
            "median_min_cosine_dist_to_selected": float(np.median(coverage_dists)) if len(coverage_dists) > 0 else 0.0,
            "pool_covered_within_05": int((coverage_dists <= 0.5).sum()) if len(coverage_dists) > 0 else 0,
            "pool_covered_within_03": int((coverage_dists <= 0.3).sum()) if len(coverage_dists) > 0 else 0,
            "total_unselected": int(len(non_sel_idx)),
        },
        "selection_log": selection_log,
    }

    report = {
        "total": len(dataset),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "selected": len(selected),
        "num_views": cfg.num_views,
        "embedding": effective_embedding,
        "embedding_model": cfg.embedding_model,
        "selector": cfg.selector,
        "accepted_ids": [obs.id for obs in accepted],
        "selected_ids": [obs.id for obs in selected],
        "rejected_ids": [obs.id for obs in rejected],
        "selection_metrics": selection_metrics,
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Selected IDs: {[obs.id for obs in selected]}")

    if cfg.save_visualization and len(selected) > 0:
        try:
            from utils.visualization import save_overview_grid
            imgs = [obs.image for obs in selected]
            masks = [obs.mask for obs in selected]
            titles = [f"ID:{obs.id} Q:{obs.quality:.2f}" for obs in selected]
            save_overview_grid(imgs, masks, str(output_dir / "visualization.png"), titles)
            print(f"Visualization saved to {output_dir / 'visualization.png'}")
        except Exception as e:
            print(f"Visualization failed: {e}")

    if cfg.save_plots:
        try:
            from utils.plotting import plot_all
            plot_all(accepted, rejected, selected, embeddings, selected_idx, quality_scores, output_dir, single_set_plots=cfg.debug, debug=cfg.debug)
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object View Selection Pipeline")
    parser.add_argument("--data_root", type=str, default="",
                        help="Path to dataset root (images/, masks/, object_hands/)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--embedding", type=str, default="auto",
                        choices=["auto", "dinov3", "dinov2", "siglip2", "siglip", "moonvit", "clip", "eva_clip"],
                        help="Embedding type (auto=infer from --embedding_model)")
    parser.add_argument("--embedding_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="Model name or path; type inferred automatically when --embedding=auto")
    parser.add_argument("--selector", type=str, default="quality_diversity",
                        choices=["fps", "quality_diversity", "facility_location", "dpp", "next_best_view"])
    parser.add_argument("--use_shape_descriptors", action="store_true",
                        help="Use classical shape descriptors instead of learned embeddings")
    parser.add_argument("--shape_descriptor", type=str, default="hu",
                        choices=["hu", "zernike", "fourier", "shape_context"])
    parser.add_argument("--plot", action="store_true", dest="save_plots",
                        help="Generate pipeline diagnostic plots")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose terminal output with per-step statistics; also enables single-set violin plots (requires --plot)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_root=args.data_root or "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/bottle",
        output_dir=args.output_dir,
        num_views=args.num_views,
        embedding=args.embedding,
        embedding_model=args.embedding_model,
        selector=args.selector,
        use_shape_descriptors=args.use_shape_descriptors,
        shape_descriptor=args.shape_descriptor,
        save_plots=args.save_plots,
        debug=args.debug,
    )
    run_pipeline(cfg)