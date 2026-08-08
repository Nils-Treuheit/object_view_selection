import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from config import PipelineConfig
from data_io.dataset import Dataset
from preprocessing.legacy.area_filter import AreaFilter
from preprocessing.border_blur_filter import (
    BorderLaplacianBlurFilter,
    BorderTenengradBlurFilter,
)
from preprocessing.legacy.border_truncation import BorderFilter
from preprocessing.future_work.completeness_filter import CompletenessFilter
from preprocessing.future_work.confidence import ConfidenceFilter
from preprocessing.filter_pipeline import FilterPipeline
from preprocessing.future_work.occlusion_filter import OcclusionFilter
from preprocessing.vincent_border_pixel import VincentBorderPixelFilter
from preprocessing.vincent_empty_mask import VincentEmptyMaskFilter
from preprocessing.vincents_area_filter import VincentsAreaFilter
from preprocessing.vincents_artefacts import VincentsArtifactsFilter
from preprocessing.vincents_motion_blur import VincentsMotionBlurFilter
from quality.area import AreaQuality
from quality.blur import BorderBlurQuality
from quality.centerness import CenternessQuality
from quality.quality_scorer import QualityScorer
from quality.vincent import VincentsArtifactsQuality


def _maybe_outlier(f, conf):
    """Wrap a pre-filter in a population-outlier rejection when configured.

    Filters that implement their own absolute threshold criterion (the legacy
    ``AreaFilter`` / ``BorderFilter`` / ...) get the population-based
    extreme-bad-outlier rejection layered on by ``OutlierFilter`` when the
    config sets ``outlier_z``.
    """
    outlier_z = getattr(conf, "outlier_z", None)
    if outlier_z is None:
        return f
    from preprocessing.variants import OutlierFilter
    return OutlierFilter(f, outlier_z=outlier_z)


def build_filters(cfg: PipelineConfig, tuned=None):
    if tuned is None:
        tuned = {}

    # NOTE: the default pre-filter set is deliberately small and deliberately
    # conservative. Every default filter removes (a) awful-quality samples
    # below a very relaxed bare-minimum absolute threshold and (b) extreme bad
    # outliers relative to the population. Both criteria are implemented by
    # each filter itself via the shared ScoreFilter base (see
    # preprocessing/base.py + preprocessing/filter_utils.py).
    available = {
        "vincent_empty_mask": VincentEmptyMaskFilter(
            enabled=cfg.filters.vincent_empty_mask.enabled,
        ),
        "vincent_border_pixel": VincentBorderPixelFilter(
            enabled=cfg.filters.vincent_border_pixel.enabled,
        ),
        "blur_laplacian": BorderLaplacianBlurFilter(
            stroke_width=cfg.filters.blur_laplacian.stroke_width,
            max_variance=cfg.filters.blur_laplacian.max_variance,
            hard_min_variance=cfg.filters.blur_laplacian.hard_min_variance,
            outlier_z=cfg.filters.blur_laplacian.outlier_z,
            enabled=cfg.filters.blur_laplacian.enabled,
        ),
        "blur_tenengrad": BorderTenengradBlurFilter(
            stroke_width=cfg.filters.blur_tenengrad.stroke_width,
            max_tenengrad=cfg.filters.blur_tenengrad.max_tenengrad,
            hard_min_tenengrad=cfg.filters.blur_tenengrad.hard_min_tenengrad,
            outlier_z=cfg.filters.blur_tenengrad.outlier_z,
            enabled=cfg.filters.blur_tenengrad.enabled,
        ),
        "vincents_artefacts": VincentsArtifactsFilter(
            kernel_size=cfg.filters.vincents_artefacts.kernel_size,
            max_fraction=cfg.filters.vincents_artefacts.max_fraction,
            hard_max_fraction=cfg.filters.vincents_artefacts.hard_max_fraction,
            outlier_z=cfg.filters.vincents_artefacts.outlier_z,
            enabled=cfg.filters.vincents_artefacts.enabled,
        ),
        # ------------------------------------------------------------------ #
        # Legacy pre-filters, kept for custom --filter_order only.
        # NOT part of the default set and NOT tested / likely not working as
        # proper pre-filters (occlusion, completeness, area, confidence).
        # ------------------------------------------------------------------ #
        "area": _maybe_outlier(AreaFilter(
            minimum_ratio=tuned.get("area_minimum_ratio", cfg.filters.area.minimum_ratio),
            enabled=cfg.filters.area.enabled,
        ), cfg.filters.area),
        "border": _maybe_outlier(BorderFilter(
            maximum_ratio=tuned.get("border_maximum_ratio", cfg.filters.border.maximum_ratio),
            edge_maximum_ratio=tuned.get("border_edge_maximum_ratio", cfg.filters.border.edge_maximum_ratio),
            enabled=cfg.filters.border.enabled,
        ), cfg.filters.border),
        "occlusion": _maybe_outlier(OcclusionFilter(
            maximum_overlap=tuned.get("occlusion_maximum_overlap", cfg.filters.occlusion.maximum_overlap),
            enabled=cfg.filters.occlusion.enabled,
        ), cfg.filters.occlusion),
        "confidence": _maybe_outlier(ConfidenceFilter(
            minimum_confidence=cfg.filters.confidence.minimum_confidence,
            enabled=cfg.filters.confidence.enabled,
        ), cfg.filters.confidence),
        "completeness": _maybe_outlier(CompletenessFilter(
            minimum_score=tuned.get("completeness_minimum_score", cfg.filters.completeness.minimum_score),
            enabled=cfg.filters.completeness.enabled,
        ), cfg.filters.completeness),
    }
    filters = []
    for name in cfg.filters.filter_order:
        if name in available:
            filters.append(available[name])
    return FilterPipeline(filters)


def build_soft_filters(cfg: PipelineConfig):
    """Population-adapted soft pre-filters (ported from nit_view_selection).

    These compute raw per-observation stats and then a population pass turns
    those stats into robust selection weights in (0, 1]. They are kept as
    diagnostics — none of them feeds the default pre-filter set or the
    4-component quality score. Both ``VincentsAreaFilter`` and
    ``VincentsMotionBlurFilter`` implement both ``BaseFilter`` rejection
    criteria themselves (absolute threshold floor + outlier_z population
    outlier via the shared ``ScoreFilter`` base), so they can act as working
    pre-filters.
    """
    return {
        "vincents_area": VincentsAreaFilter(
            softness=cfg.filters.vincents_area.softness,
            hard_min_area_fraction=cfg.filters.vincents_area.hard_min_area_fraction,
            outlier_z=cfg.filters.vincents_area.outlier_z,
            enabled=cfg.filters.vincents_area.enabled,
        ),
        "vincents_motion_blur": VincentsMotionBlurFilter(
            softness=cfg.filters.vincents_motion_blur.softness,
            stroke_width=cfg.filters.vincents_motion_blur.stroke_width,
            hard_min_variance=cfg.filters.vincents_motion_blur.hard_min_variance,
            outlier_z=cfg.filters.vincents_motion_blur.outlier_z,
            enabled=cfg.filters.vincents_motion_blur.enabled,
        ),
    }


def apply_soft_filters(soft_filters, accepted, rejected=None):
    """Run soft pre-filter pass: raw stats per observation, then population weights.

    Raw stats are computed for all observations (accepted + rejected) so the
    diagnostic plots can compare them; population weights are fit only on the
    accepted set (rejected observations do not compete for selection).

    Soft filters that implement the ``BaseFilter`` rejection criteria
    (``need_fitting`` + ``evaluate`` returning ``passed=False``, e.g. the
    motion-blur threshold/outlier modes) run their population fit first and
    move the observations they reject out of ``accepted`` with the annotated
    reason; already-rejected observations keep their original reason.
    """
    if rejected is None:
        rejected = []
    all_observations = list(accepted) + list(rejected)
    for soft_filter in soft_filters.values():
        # population pass for outlier-mode filters (before per-observation evaluate)
        if getattr(soft_filter, "need_fitting", lambda: False)():
            soft_filter.fit(all_observations)

        for obs in all_observations:
            _score, passed, reason = soft_filter.evaluate(obs)
            if passed or obs.rejection_reason is not None:
                continue
            obs.rejected = True
            obs.rejection_reason = reason
            if obs in accepted:
                accepted.remove(obs)
                rejected.append(obs)

        soft_filter.fit_weights(accepted)


def build_quality_scorer(cfg: PipelineConfig, tuned=None):
    if tuned is None:
        tuned = {}
    metrics = []
    weights = {}
    anchors = cfg.quality_anchors
    # 4 quality components: border blur, mask artifact, area, centerness.
    if cfg.filters.blur_laplacian.enabled:
        metrics.append(BorderBlurQuality(max_variance=anchors.blur_max_variance))
        weights["blur"] = cfg.quality_weights.blur
    if cfg.filters.vincents_artefacts.enabled:
        metrics.append(VincentsArtifactsQuality(max_fraction=anchors.artifacts_max_fraction))
        weights["vincents_artefacts"] = cfg.quality_weights.vincents_artefacts
    metrics.append(AreaQuality())
    weights["area"] = cfg.quality_weights.area
    metrics.append(CenternessQuality())
    weights["centerness"] = cfg.quality_weights.centerness
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
    elif name == "top_kmeans_xnn":
        from selection.kmeans_xnn import TopKMeansXNN
        return TopKMeansXNN(init=cfg.kmeans_init, xnn_k=cfg.kmeans_xnn_k)
    else:
        raise ValueError(f"Unknown selector: {name}")


def compute_quality_floor(quality_scores, num_views: int, cfg) -> float:
    """Adaptive minimum-quality floor for the embedding selection pool.

    Drops the worst tail of the accepted pool so low-quality samples are
    excluded from the selection set, while guaranteeing enough candidates
    remain for a diverse sample-set selection:

    - ``quality_floor.percentile``: drop the bottom ``percentile`` of the pool.
    - ``quality_floor.minimum_pool``: never leave fewer than this many
      candidates (unless the pool itself is smaller).
    - ``quality_floor.absolute_min``: never let samples below this absolute
      quality into the selection pool.
    - the floor never drops the pool below ``num_views`` candidates.
    """
    scores = np.asarray(quality_scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0
    sorted_q = np.sort(scores)
    if num_views >= n:
        return float(sorted_q[0])

    # target tail drop: reject the bottom `percentile`, plus the absolute min
    floor = max(
        float(np.quantile(scores, cfg.quality_floor.percentile)),
        cfg.quality_floor.absolute_min,
    )

    # guarantee at least minimum_pool candidates survive: cap the floor
    min_pool = max(cfg.quality_floor.minimum_pool, num_views)
    if n >= min_pool:
        floor = min(floor, float(sorted_q[-min_pool]))

    # never drop the pool below num_views candidates
    floor = min(floor, float(sorted_q[-num_views]))
    return float(floor)


def _save_samples(observations, kind, data_root, output_dir, group=None):
    """Copy observations into ``{kind}_samples/<obj_id>/{rgb,mask,depth?,hand_mask?}``.

    The ``<obj_id>`` folder is named exactly like the last component of
    ``data_root``. Under it the frames are re-organized into:

      rgb/          object images
      mask/         object masks
      depth/        frame-wise depth information (only when a matching file
                    exists in ``<data_root>/depth``)
      hand_mask/    hand masks (only when a hand mask exists for the frame)

    When ``group`` is given the observations are additionally nested under a
    ``{kind}_samples/<group>/<obj_id>/...`` subfolder (used to split rejected
    observations by their rejection reason).
    """
    root = Path(data_root)
    obj_id = root.name or "dataset"
    if group:
        base = Path(output_dir) / f"{kind}_samples" / group / obj_id
    else:
        base = Path(output_dir) / f"{kind}_samples" / obj_id

    rgb_dir = base / "rgb"
    mask_dir = base / "mask"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    depth_src = root / "depth"
    hand_src = root / "object_hands"

    depth_map = {}
    if depth_src.is_dir():
        for p in depth_src.iterdir():
            if p.suffix.lower() in (".png", ".npy", ".jpg", ".jpeg", ".tiff") and p.stem.isdigit():
                depth_map[int(p.stem)] = p

    hand_map = {}
    if hand_src.is_dir():
        for p in hand_src.glob("*.png"):
            if p.stem.isdigit():
                hand_map[int(p.stem)] = p

    for obs in observations:
        stem = f"{obs.id:05d}"
        if obs.image is not None:
            cv2.imwrite(str(rgb_dir / f"{stem}.png"),
                        cv2.cvtColor(obs.image, cv2.COLOR_RGB2BGR))
        elif obs.image_path is not None and Path(obs.image_path).exists():
            shutil.copy(str(obs.image_path), str(rgb_dir / f"{stem}.png"))
        if obs.mask is not None:
            cv2.imwrite(str(mask_dir / f"{stem}.png"), obs.mask)
        elif obs.mask_path is not None and Path(obs.mask_path).exists():
            shutil.copy(str(obs.mask_path), str(mask_dir / f"{stem}.png"))

        if depth_map:
            depth_dir = base / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            dp = depth_map.get(obs.id)
            if dp is not None:
                shutil.copy(str(dp), str(depth_dir / dp.name))

        if hand_map:
            hand_mask_dir = base / "hand_mask"
            hand_mask_dir.mkdir(parents=True, exist_ok=True)
            hp = hand_map.get(obs.id)
            if hp is not None:
                shutil.copy(str(hp), str(hand_mask_dir / hp.name))

    print(f"{kind.capitalize()} samples saved to {base}")
    return base


def save_selected_samples(selected, data_root, output_dir):
    """Copy the final selected tuples into ``selected_samples/<obj_id>/``."""
    return _save_samples(selected, "selected", data_root, output_dir)


def save_rejected_samples(rejected, data_root, output_dir):
    """Copy the rejected tuples into ``rejected_samples/<obj_id>/``.

    Mirrors ``save_selected_samples``: same ``rgb/``, ``mask/``, ``depth/``
    and ``hand_mask/`` layout under a folder named after ``data_root``.
    """
    return _save_samples(rejected, "rejected", data_root, output_dir)


_REASON_SUFFIXES = ("_threshold", "_outlier")


def _base_reason(reason):
    """Strip the threshold/outlier suffix from a rejection reason.

    ``blur_laplacian_threshold`` -> ``blur_laplacian`` so both rejection modes
    of one filter group into the same reason folder.
    """
    for suffix in _REASON_SUFFIXES:
        if reason.endswith(suffix):
            return reason[: -len(suffix)]
    return reason


def _rejection_mode(reason):
    """Map a rejection reason to its ``{threshold,outlier}-based`` subfolder.

    Reasons carrying the ``_threshold`` / ``_outlier`` suffix (the two
    ``BaseFilter`` rejection criteria) map to their mode; pure hard reasons
    (empty mask, border pixel) are absolute structural rejects and fall under
    ``threshold-based``.
    """
    if reason.endswith("_outlier"):
        return "outlier-based"
    return "threshold-based"


def save_rejected_samples_by_reason(rejected, data_root, output_dir):
    """Group rejected tuples by rejection reason into per-filter subfolders.

    Writes
    ``rejected_samples/<reason>/threshold-based/<obj_id>/{rgb,mask,depth?,hand_mask?}``
    and ``rejected_samples/<reason>/outlier-based/<obj_id>/...`` so a run shows
    *why* every frame was rejected at a glance and which rejection mode
    (relaxed absolute threshold vs extreme-bad outlier) triggered it. The
    reason is sanitised so it can never escape the ``rejected_samples`` folder.
    """
    by_reason = {}
    for obs in rejected:
        reason = (obs.rejection_reason or "unknown").replace("/", "_")
        by_reason.setdefault(reason, []).append(obs)

    bases = []
    base_root = Path(output_dir) / "rejected_samples"
    for reason in sorted(by_reason):
        mode = _rejection_mode(reason)
        group = f"{_base_reason(reason)}/{mode}"
        bases.append(_save_samples(by_reason[reason], "rejected", data_root, output_dir, group=group))
        # ensure every reason folder has both subfolders (even when empty)
        for other_mode in ("threshold-based", "outlier-based"):
            (base_root / _base_reason(reason) / other_mode).mkdir(parents=True, exist_ok=True)
    if rejected:
        print(f"Rejected samples grouped by reason saved to {base_root}")
    return bases


def save_accepted_samples(unselected, data_root, output_dir):
    """Copy the accepted-but-unselected tuples into ``accepted_samples/<obj_id>/``.

    Same ``rgb/``, ``mask/``, ``depth/`` and ``hand_mask/`` layout as the
    selected/rejected dumps; these frames passed the pre-filter and quality
    floor but were not picked by the selection step.
    """
    if not unselected:
        return None
    return _save_samples(unselected, "accepted", data_root, output_dir)


def run_pipeline(cfg: PipelineConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  occlusion_maximum_overlap={tuned['occlusion_maximum_overlap']}")
        print(f"  completeness_minimum_score={tuned['completeness_minimum_score']}")
        print("  (default blur/artifact pre-filters use static relaxed floors +")
        print("   population-relative outlier rejection, no tuning needed)")

    filter_pipeline = build_filters(cfg, tuned)
    quality_scorer = build_quality_scorer(cfg, tuned)
    soft_filters = build_soft_filters(cfg)
    embedding_model = build_embedding_model(cfg)
    selector = build_selector(cfg)

    accepted = []
    rejected = []

    if filter_pipeline.need_fitting:
        print("Fitting pre-filter outlier statistics on the population...")
        filter_pipeline.fit_observations(dataset.observations)

    for obs in tqdm(dataset.observations, desc="Pre-filtering"):
        if not filter_pipeline.run(obs):
            rejected.append(obs)
            continue
        accepted.append(obs)

    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    apply_soft_filters(soft_filters, accepted, rejected)

    if cfg.debug:
        rejection_counts = {}
        for obs in rejected:
            r = obs.rejection_reason or "unknown"
            rejection_counts[r] = rejection_counts.get(r, 0) + 1
        print("  Rejection breakdown:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        if accepted:
            raw_metrics = ["laplacian", "tenengrad", "vincent_area_fraction",
                           "vincent_artifact_fraction", "vincent_boundary_blur_variance"]
            print("  Accepted raw metrics:")
            for key in raw_metrics:
                vals = np.array([getattr(o.metrics, key, 0) for o in accepted])
                print(f"    {key}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")

    if len(accepted) == 0:
        print("No observations passed filtering. Exiting.")
        return

    if cfg.only_pre_filter:
        if cfg.save_rejected:
            save_rejected_samples_by_reason(rejected, cfg.data_root, output_dir)
        save_accepted_samples(accepted, cfg.data_root, output_dir)

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
                "reason": obs.rejection_reason,
                "laplacian": m.laplacian,
                "tenengrad": m.tenengrad,
                "area_ratio": m.area_ratio,
                "border_ratio": m.border_ratio,
                "edge_ratio": m.edge_ratio,
                "hand_overlap": m.hand_overlap,
                "completeness": m.completeness,
                "vincent_pixel_count": m.vincent_pixel_count,
                "vincent_touches_border": m.vincent_touches_border,
                "vincent_area_fraction": m.vincent_area_fraction,
                "vincent_artifact_fraction": m.vincent_artifact_fraction,
                "vincent_boundary_blur_variance": m.vincent_boundary_blur_variance,
            })
        import pandas as pd
        pd.DataFrame(rejected_metrics_csv).to_csv(output_dir / "rejected_metrics.csv", index=False)

        print(f"\nPre-filter only (--only_pre_filter): accepted {len(accepted)}, rejected {len(rejected)}")
        print("Stopped before quality scoring / embedding / selection / plots.")
        return

    for obs in tqdm(accepted, desc="Scoring quality"):
        quality_scorer.score(obs)

    for obs in accepted:
        obs.metrics.confidence = min(
            obs.metrics.blur,
            obs.metrics.area,
            obs.metrics.vincents_artefacts,
            obs.metrics.centerness,
        )

    if cfg.debug:
        quality_keys = ["blur", "area", "vincents_artefacts", "centerness", "confidence"]
        print("  Quality scores:")
        for key in quality_keys:
            vals = np.array([getattr(o.metrics, key, 0) for o in accepted])
            print(f"    {key}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")
        qvals = np.array([o.quality for o in accepted])
        print(f"    score:  min={qvals.min():.4f}  max={qvals.max():.4f}  mean={qvals.mean():.4f}  median={np.median(qvals):.4f}")

    quality_scores = np.array([obs.quality for obs in accepted])

    pool = accepted
    floor = 0.0
    if cfg.quality_floor.enabled:
        floor = compute_quality_floor(quality_scores, cfg.num_views, cfg)
        if floor > 0.0:
            pool_mask = quality_scores >= floor
            pool = [obs for obs, keep in zip(accepted, pool_mask) if keep]
            print(f"Quality floor {floor:.3f}: selection pool {len(pool)} of {len(accepted)}")

    if len(pool) == 0:
        print("No observations above the quality floor. Exiting.")
        return

    if cfg.use_shape_descriptors or embedding_model is None:
        for obs in tqdm(pool, desc="Extracting descriptors"):
            feat = extract_shape_descriptor(obs, cfg.shape_descriptor)
            obs.embedding = feat
    else:
        from embeddings.crop import compute_contrast_background
        background = compute_contrast_background(pool)
        embedding_model.set_background(background)
        if cfg.debug:
            print(f"  Static contrast background over {len(pool)} pool samples: "
                  f"{'black' if background == 0 else 'white'}")
        for obs in tqdm(pool, desc="Extracting embeddings"):
            obs.embedding = embedding_model.encode(obs.image, obs.mask)

    embeddings = np.array([obs.embedding for obs in pool])
    pool_quality = np.array([obs.quality for obs in pool])

    selected_idx = selector.select(
        embeddings=embeddings,
        quality_scores=pool_quality,
        n=cfg.num_views,
    )

    selected = [pool[i] for i in selected_idx]
    print(f"Selected {len(selected)} views")

    if cfg.debug:
        sel_qual = pool_quality[selected_idx]
        print(f"  Selected quality: min={sel_qual.min():.4f}  max={sel_qual.max():.4f}  mean={sel_qual.mean():.4f}")
        print(f"  Pool quality:     min={pool_quality.min():.4f}  max={pool_quality.max():.4f}  mean={pool_quality.mean():.4f}")

    selected_set = {s.id for s in selected}
    pool_set = {obs.id for obs in pool}

    quality_csv = []
    for i, obs in enumerate(accepted):
        row = {
            "id": obs.id,
            "quality": obs.quality,
            "score": obs.quality,
            "in_selection_pool": obs.id in pool_set,
            "below_quality_floor": obs.id not in pool_set,
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
            "centerness": obs.metrics.centerness,
            "confidence": obs.metrics.confidence,
            "vincent_area_fraction": obs.metrics.vincent_area_fraction,
            "vincent_artifact_fraction": obs.metrics.vincent_artifact_fraction,
            "vincent_boundary_blur_variance": obs.metrics.vincent_boundary_blur_variance,
            "vincents_area": obs.metrics.vincents_area,
            "vincents_artefacts": obs.metrics.vincents_artefacts,
            "selected": obs.id in selected_set,
        }
        quality_csv.append(row)

    import pandas as pd
    df = pd.DataFrame(quality_csv)
    df.to_csv(output_dir / "quality.csv", index=False)

    if cfg.save_embeddings:
        np.save(output_dir / "embeddings.npy", embeddings)
        np.save(output_dir / "selected_indices.npy", selected_idx)
        np.save(output_dir / "selection_pool_ids.npy", np.array([obs.id for obs in pool]))

    if selected:
        save_selected_samples(selected, cfg.data_root, output_dir)

    if cfg.debug:
        unselected = [obs for obs in accepted if obs.id not in selected_set]
        save_accepted_samples(unselected, cfg.data_root, output_dir)

    if cfg.save_rejected:
        save_rejected_samples_by_reason(rejected, cfg.data_root, output_dir)
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
            "reason": obs.rejection_reason,
            "laplacian": m.laplacian,
            "tenengrad": m.tenengrad,
            "area_ratio": m.area_ratio,
            "border_ratio": m.border_ratio,
            "edge_ratio": m.edge_ratio,
            "hand_overlap": m.hand_overlap,
            "completeness": m.completeness,
            "vincent_pixel_count": m.vincent_pixel_count,
            "vincent_touches_border": m.vincent_touches_border,
            "vincent_area_fraction": m.vincent_area_fraction,
            "vincent_artifact_fraction": m.vincent_artifact_fraction,
            "vincent_boundary_blur_variance": m.vincent_boundary_blur_variance,
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
    sel_qual = pool_quality[selected_idx]

    non_sel_mask = np.ones(len(embeddings), dtype=bool)
    non_sel_mask[selected_idx] = False
    non_sel_idx = np.where(non_sel_mask)[0]
    coverage_dists = dist[non_sel_idx][:, selected_idx].min(axis=1) if len(non_sel_idx) > 0 else np.array([])

    selection_log = []
    if cfg.selector == "quality_diversity":
        steps = []
        remaining = set(range(len(embeddings)))
        first = int(pool_quality.argmax())
        steps.append(first)
        selection_log.append({
            "step": 0, "id": int(pool[first].id),
            "quality": float(pool_quality[first]),
            "min_cosine_dist_to_set": None, "score": None,
        })
        remaining.remove(first)
        while len(steps) < len(selected_idx):
            best_score = -np.inf
            best_i = -1
            for i in remaining:
                diversity = dist[i, steps].min()
                score = cfg.selector_alpha * pool_quality[i] + cfg.selector_beta * diversity
                if score > best_score:
                    best_score = score
                    best_i = i
            steps.append(best_i)
            remaining.remove(best_i)
            selection_log.append({
                "step": len(steps)-1, "id": int(pool[best_i].id),
                "quality": float(pool_quality[best_i]),
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
            "pool_mean": float(pool_quality.mean()),
            "pool_min": float(pool_quality.min()),
            "quality_floor": float(floor),
            "selection_pool_count": int(len(pool)),
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
        "quality_floor": float(floor),
        "selection_pool_count": len(pool),
        "data_root": str(cfg.data_root),
        "accepted_ids": [obs.id for obs in accepted],
        "selection_pool_ids": [obs.id for obs in pool],
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
            plot_all(accepted, rejected, selected, embeddings, selected_idx, pool_quality, output_dir, single_set_plots=cfg.debug, debug=cfg.debug, pool_obs=pool, n_clusters=cfg.kmeans_xnn_k)
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
                        choices=["fps", "quality_diversity", "facility_location", "dpp",
                                 "next_best_view", "top_kmeans_xnn"])
    parser.add_argument("--selector_alpha", type=float, default=None,
                        help="Quality weight for the quality_diversity (GQD) selector "
                             "(default: config value 0.60)")
    parser.add_argument("--selector_beta", type=float, default=None,
                        help="Diversity weight for the quality_diversity (GQD) selector "
                             "(default: config value 0.40)")
    parser.add_argument("--kmeans_init", type=str, default=None,
                        choices=["farthest", "best_quality"],
                        help="k-means cluster-init for the top_kmeans_xnn selector: "
                             "farthest-point seeds or best-quality seeds "
                             "(default: config value 'farthest')")
    parser.add_argument("--kmeans_xnn_k", type=int, default=None,
                        choices=[3, 5, 10],
                        help="xNN neighbourhood radius for the top_kmeans_xnn selector: "
                             "pick the best-quality sample among the centroid plus its "
                             "x nearest neighbours (default: config value 3)")
    parser.add_argument("--filter_order", type=str, default=None,
                        help="Comma-separated pre-filter application order, e.g. "
                             "'vincent_empty_mask,vincent_border_pixel,blur_laplacian,"
                             "blur_tenengrad,vincents_artefacts'. "
                             "Legacy filters available for custom orders: "
                             "'border,area,occlusion,confidence,completeness' "
                             "(NOT tested / likely not working as proper pre-filters). "
                             "Defaults to the config default (the current setup).")
    parser.add_argument("--use_shape_descriptors", action="store_true",
                        help="Use classical shape descriptors instead of learned embeddings")
    parser.add_argument("--shape_descriptor", type=str, default="hu",
                        choices=["hu", "zernike", "fourier", "shape_context"])
    parser.add_argument("--no-auto-thresholds", action="store_true", dest="no_auto_thresholds",
                        help="Disable data-driven threshold tuning (use static config values)")
    parser.add_argument("--plot", action="store_true", dest="save_plots",
                        help="Generate pipeline diagnostic plots")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose terminal output with per-step statistics; also enables single-set violin plots (requires --plot)")
    parser.add_argument("--only_pre_filter", action="store_true",
                        help="Stop right after the pre-filter stage: dump accepted_samples/ "
                             "and per-reason rejected_samples/, write rejected.json + "
                             "rejected_metrics.csv, then exit before selection")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_root=args.data_root or "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/09_triprong",
        output_dir=args.output_dir,
        num_views=args.num_views,
        embedding=args.embedding,
        embedding_model=args.embedding_model,
        selector=args.selector,
        use_shape_descriptors=args.use_shape_descriptors,
        shape_descriptor=args.shape_descriptor,
        auto_thresholds=not args.no_auto_thresholds,
        save_plots=args.save_plots,
        debug=args.debug,
        only_pre_filter=args.only_pre_filter,
    )
    if args.selector_alpha is not None:
        cfg.selector_alpha = args.selector_alpha
    if args.selector_beta is not None:
        cfg.selector_beta = args.selector_beta
    if args.kmeans_init is not None:
        cfg.kmeans_init = args.kmeans_init
    if args.kmeans_xnn_k is not None:
        cfg.kmeans_xnn_k = args.kmeans_xnn_k
    if args.filter_order:
        order = [name.strip() for name in args.filter_order.split(",") if name.strip()]
        if order:
            cfg.filters.filter_order = order
    run_pipeline(cfg)