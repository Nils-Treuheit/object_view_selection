"""
Automatic threshold tuning from dataset statistics.

Usage:
    from utils.threshold_tuner import tune_thresholds

    thresholds = tune_thresholds(observations)
"""

import numpy as np

from preprocessing.legacy.area_filter import AreaFilter
from preprocessing.border_blur_filter import (
    BorderLaplacianBlurFilter,
    BorderTenengradBlurFilter,
)
from preprocessing.legacy.border_truncation import BorderFilter
from preprocessing.future_work.completeness_filter import CompletenessFilter
from preprocessing.future_work.occlusion_filter import OcclusionFilter


def compute_metric_stats(observations):
    areas = []
    borders = []
    edges = []
    laplacians = []
    tenengrads = []
    overlaps = []
    completenesses = []

    for obs in observations:
        AreaFilter().evaluate(obs)
        BorderFilter().evaluate(obs)
        # boundary-band sharpness (same band the default blur pre-filters use)
        BorderLaplacianBlurFilter().evaluate(obs)
        BorderTenengradBlurFilter().evaluate(obs)
        OcclusionFilter().evaluate(obs)
        CompletenessFilter().evaluate(obs)

        areas.append(obs.metrics.area_ratio)
        borders.append(obs.metrics.border_ratio)
        edges.append(obs.metrics.edge_ratio)
        laplacians.append(obs.metrics.laplacian)
        tenengrads.append(obs.metrics.tenengrad)
        overlaps.append(obs.metrics.hand_overlap)
        completenesses.append(obs.metrics.completeness)

    return dict(
        area_ratio=np.array(areas),
        border_ratio=np.array(borders),
        edge_ratio=np.array(edges),
        laplacian=np.array(laplacians),
        tenengrad=np.array(tenengrads),
        hand_overlap=np.array(overlaps),
        completeness=np.array(completenesses),
    )


DEFAULT_PERCENTILES = dict(
    area_ratio=1,
    border_ratio=95,
    edge_ratio=95,
    laplacian=5,
    tenengrad=5,
    hand_overlap=95,
    completeness=1,
)


SAFETY_LIMITS = dict(
    area_minimum_ratio=dict(min=0.01, max=0.05),
    border_maximum_ratio=dict(min=0.001, max=0.05),
    # boundary-band sharpness (the scale the default blur pre-filters use)
    laplacian_threshold=dict(min=5.0, max=1000.0),
    tenengrad_threshold=dict(min=3.0, max=100.0),
    border_edge_maximum_ratio=dict(min=0.05, max=0.5),
    occlusion_maximum_overlap=dict(min=0.001, max=0.30),
    completeness_minimum_score=dict(min=0.50, max=0.80),
)


def tune_thresholds(observations, percentiles=None):
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    stats = compute_metric_stats(observations)

    p_area = np.percentile(stats["area_ratio"], percentiles["area_ratio"])
    p_border = np.percentile(stats["border_ratio"], percentiles["border_ratio"])
    p_edge = np.percentile(stats["edge_ratio"], percentiles["edge_ratio"])
    p_lap = np.percentile(stats["laplacian"], percentiles["laplacian"])
    p_ten = np.percentile(stats["tenengrad"], percentiles["tenengrad"])
    p_overlap = np.percentile(stats["hand_overlap"], percentiles["hand_overlap"])
    p_comp = np.percentile(stats["completeness"], percentiles["completeness"])

    area_min = np.clip(
        p_area,
        SAFETY_LIMITS["area_minimum_ratio"]["min"],
        SAFETY_LIMITS["area_minimum_ratio"]["max"],
    )
    border_max = np.clip(
        p_border,
        SAFETY_LIMITS["border_maximum_ratio"]["min"],
        SAFETY_LIMITS["border_maximum_ratio"]["max"],
    )
    border_edge_max = np.clip(
        p_edge,
        SAFETY_LIMITS["border_edge_maximum_ratio"]["min"],
        SAFETY_LIMITS["border_edge_maximum_ratio"]["max"],
    )
    lap_thresh = np.clip(
        p_lap,
        SAFETY_LIMITS["laplacian_threshold"]["min"],
        SAFETY_LIMITS["laplacian_threshold"]["max"],
    )
    ten_thresh = np.clip(
        p_ten,
        SAFETY_LIMITS["tenengrad_threshold"]["min"],
        SAFETY_LIMITS["tenengrad_threshold"]["max"],
    )
    occ_max = np.clip(
        p_overlap,
        SAFETY_LIMITS["occlusion_maximum_overlap"]["min"],
        SAFETY_LIMITS["occlusion_maximum_overlap"]["max"],
    )
    comp_min = np.clip(
        p_comp,
        SAFETY_LIMITS["completeness_minimum_score"]["min"],
        SAFETY_LIMITS["completeness_minimum_score"]["max"],
    )

    return dict(
        area_minimum_ratio=float(round(area_min, 4)),
        border_maximum_ratio=float(round(border_max, 4)),
        border_edge_maximum_ratio=float(round(border_edge_max, 4)),
        laplacian_threshold=float(round(lap_thresh, 1)),
        tenengrad_threshold=float(round(ten_thresh, 1)),
        occlusion_maximum_overlap=float(round(occ_max, 4)),
        completeness_minimum_score=float(round(comp_min, 4)),
    )
