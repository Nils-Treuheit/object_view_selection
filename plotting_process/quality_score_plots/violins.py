import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def _build_df(observations, score_keys, label):
    rows = []
    for obs in observations:
        row = {"id": obs.id}
        m = obs.metrics
        for k in score_keys:
            if k == "border_free":
                row[k] = 1.0 - getattr(m, "border_ratio", 0.0)
            elif k == "hand_overlap_free":
                row[k] = 1.0 - getattr(m, "hand_overlap", 0.0)
            else:
                row[k] = getattr(m, k, None)
        row["quality"] = obs.quality
        row["score"] = obs.quality
        rows.append(row)
    df = pd.DataFrame(rows)
    df["group"] = label
    return df


def _create_violin_figure(n, sep_index):
    has_spacer = sep_index is not None and sep_index < n - 1
    if has_spacer:
        n_cols = n + 1
        ratios = [1.0] * n_cols
        ratios[sep_index + 1] = 0.08
    else:
        n_cols = n
        ratios = None

    extra_w = 0.08 if has_spacer else 0
    fig = plt.figure(figsize=(3 * n + 1 + extra_w, 4.5))

    if has_spacer:
        gs = GridSpec(1, n_cols, width_ratios=ratios, figure=fig)
        axes_raw = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        sp = axes_raw[sep_index + 1]
        for spine in sp.spines.values():
            spine.set_visible(False)
        sp.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        sp.set_xlim(0, 1)
        sp.set_ylim(0, 1)
        sp.axvline(0.5, color="gray", linewidth=0.8, linestyle="-")
        axes = []
        for i in range(n):
            if i <= sep_index:
                axes.append(axes_raw[i])
            else:
                axes.append(axes_raw[i + 1])
        fig._ax_spacer = axes_raw[sep_index + 1]
    else:
        axes_raw = [fig.add_subplot(1, n_cols, i + 1) for i in range(n_cols)]
        axes = axes_raw
        fig._ax_spacer = None

    return fig, axes


def _plot_violin_set(
    df, score_keys, title, path, group_col=None, palette=None, sep_index=None,
):
    n = len(score_keys)
    fig, axes = _create_violin_figure(n, sep_index)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, score_keys):
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            positions = []
            data = []
            labels = []
            for i, g in enumerate(groups):
                vals = df.loc[df[group_col] == g, key].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(i + 1)
                    labels.append(g)
            if data:
                vp = ax.violinplot(data, positions, showmeans=True, showmedians=False)
                for j, body in enumerate(vp["bodies"]):
                    color = palette.get(labels[j], "#4ecdc4") if palette else "#4ecdc4"
                    body.set_facecolor(color)
                    body.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
        else:
            vals = df[key].dropna().values
            if len(vals) > 0:
                ax.violinplot(vals, positions=[1], showmeans=True, showmedians=False)
                ax.set_xticks([1])
                ax.set_xticklabels([key], fontsize=8)

        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Score" if key == score_keys[0] else "")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_violin_set_scaled(
    df, score_keys, title, path, palette=None, sep_index=None,
):
    n = len(score_keys)
    fig, axes = _create_violin_figure(n, sep_index)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, score_keys):
        vals = df[key].dropna().values
        if len(vals) == 0:
            continue
        ax.violinplot(vals, positions=[1], showmeans=True, showmedians=False)

        vmin, vmax = vals.min(), vals.max()
        margin = max((vmax - vmin) * 0.15, 0.01)
        ax.set_ylim(vmin - margin, vmax + margin)

        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Score" if key == score_keys[0] else "")
        ax.set_xticks([1])
        ax.set_xticklabels([key], fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_quality_violins(accepted, rejected, selected, output_dir_pre, output_dir_sel, single_set_plots=True):
    score_keys = ["blur", "area", "vincents_artefacts", "centerness",
                  "confidence", "score"]
    palette = {"selected": "#e76f51", "non-selected": "#264653"}

    selected_ids = {s.id for s in selected}
    non_sel = [o for o in accepted if o.id not in selected_ids]

    sep_idx = 4

    df_ns = _build_df(non_sel, score_keys, "non-selected")
    df_sel = _build_df(selected, score_keys, "selected")

    if single_set_plots:
        if len(df_ns) > 1:
            _plot_violin_set(df_ns, score_keys,
                             "Non-Selected — Quality Score Distribution",
                             output_dir_sel / "violin_non_selected.png", sep_index=sep_idx)
            _plot_violin_set_scaled(df_ns, score_keys,
                                    "Non-Selected — Quality Score Distribution (scaled)",
                                    output_dir_sel / "violin_non_selected_scaled.png", sep_index=sep_idx)
        if len(df_sel) > 1:
            _plot_violin_set(df_sel, score_keys,
                             "Selected — Quality Score Distribution",
                             output_dir_sel / "violin_selected.png", sep_index=sep_idx)
            _plot_violin_set_scaled(df_sel, score_keys,
                                    "Selected — Quality Score Distribution (scaled)",
                                    output_dir_sel / "violin_selected_scaled.png", sep_index=sep_idx)

    combined = pd.concat([df_ns, df_sel], ignore_index=True)
    if len(combined) > 1:
        n = len(score_keys)
        fig, axes = _create_violin_figure(n, sep_idx)
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, score_keys):
            data = []
            positions = []
            labels = []
            for i, g in enumerate(["non-selected", "selected"]):
                vals = combined.loc[combined["group"] == g, key].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(i + 1)
                    labels.append(g)
            if data:
                vp = ax.violinplot(data, positions, showmeans=True, showmedians=False)
                for j, body in enumerate(vp["bodies"]):
                    body.set_facecolor(palette.get(labels[j], "#4ecdc4"))
                    body.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(key, fontsize=10)
            ax.set_ylabel("Score" if key == score_keys[0] else "")
            ax.set_ylim(-0.05, 1.05)
        fig.suptitle("Selected vs Non-Selected — Quality Scores", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir_sel / "violin_selected_vs_non_selected.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir_sel / 'violin_selected_vs_non_selected.png'}")

    combined_scaled = pd.concat([df_ns, df_sel], ignore_index=True)
    if len(combined_scaled) > 1:
        n = len(score_keys)
        fig, axes = _create_violin_figure(n, sep_idx)
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, score_keys):
            data = []
            positions = []
            labels = []
            for i, g in enumerate(["non-selected", "selected"]):
                vals = combined_scaled.loc[combined_scaled["group"] == g, key].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(i + 1)
                    labels.append(g)
            if data:
                vp = ax.violinplot(data, positions, showmeans=True, showmedians=False)
                for j, body in enumerate(vp["bodies"]):
                    body.set_facecolor(palette.get(labels[j], "#4ecdc4"))
                    body.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)

            all_key_vals = combined_scaled[key].dropna().values
            if len(all_key_vals) > 0:
                vmin, vmax = all_key_vals.min(), all_key_vals.max()
                margin = max((vmax - vmin) * 0.15, 0.005)
                ax.set_ylim(vmin - margin, vmax + margin)

            ax.set_title(key, fontsize=10)
            ax.set_ylabel("Score" if key == score_keys[0] else "")
        fig.suptitle("Selected vs Non-Selected — Quality Scores (scaled)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir_sel / "violin_selected_vs_non_selected_scaled.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir_sel / 'violin_selected_vs_non_selected_scaled.png'}")

    rejected_raw_keys = ["laplacian", "tenengrad", "area_ratio", "border_free", "hand_overlap_free", "completeness",
                         "vincent_area_fraction", "vincent_artifact_fraction", "vincent_boundary_blur_variance"]
    readable_labels = {
        "laplacian": "Laplacian", "tenengrad": "Tenengrad",
        "area_ratio": "Area Ratio", "border_free": "Border-Free Ratio",
        "hand_overlap_free": "Hand-Free Ratio", "completeness": "Completeness",
        "vincent_area_fraction": "Mask Area Fraction",
        "vincent_artifact_fraction": "Artifact Fraction",
        "vincent_boundary_blur_variance": "Boundary Blur Variance",
    }
    df_rej = _build_df(rejected, rejected_raw_keys, "rejected")
    df_acc = _build_df(accepted, rejected_raw_keys, "accepted")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined_raw = pd.concat([df_rej, df_acc], ignore_index=True)

    if len(combined_raw) > 1:
        for suffix, scaled in [("", False), ("_scaled", True)]:
            n_raw = len(rejected_raw_keys)
            fig, axes = plt.subplots(1, n_raw, figsize=(3 * n_raw + 1, 4.5))
            for ax, key in zip(axes, rejected_raw_keys):
                all_vals = combined_raw[key].dropna().values
                if len(all_vals) == 0:
                    continue
                vmin_all, vmax_all = all_vals.min(), all_vals.max()
                for i, (g, color) in enumerate([("rejected", "#e9c46a"), ("accepted", "#2a9d8f")]):
                    vals = combined_raw.loc[combined_raw["group"] == g, key].dropna().values
                    if len(vals) > 1:
                        if vmax_all > vmin_all:
                            vals_norm = (vals - vmin_all) / (vmax_all - vmin_all)
                        else:
                            vals_norm = vals * 0.0
                        vp = ax.violinplot(vals_norm, positions=[i + 1], showmeans=False, showmedians=True)
                        vp["bodies"][0].set_facecolor(color)
                        vp["bodies"][0].set_alpha(0.6)
                        vp["cmedians"].set_color(color)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["rejected", "accepted"], fontsize=8)
                ax.set_title(readable_labels.get(key, key), fontsize=10)

                if scaled:
                    local_vals = combined_raw[key].dropna().values
                    if vmax_all > vmin_all:
                        local_norm = (local_vals - vmin_all) / (vmax_all - vmin_all)
                    else:
                        local_norm = local_vals * 0.0
                    vmin_z, vmax_z = local_norm.min(), local_norm.max()
                    margin = max((vmax_z - vmin_z) * 0.15, 0.005)
                    ax.set_ylim(vmin_z - margin, vmax_z + margin)
                else:
                    ax.set_ylim(-0.05, 1.05)

            title_text = "Rejected vs Accepted — Pre-Filter Metrics (normalised [0,1])"
            if scaled:
                title_text += " — scaled"
            fig.suptitle(title_text, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fname = f"violin_rejected_vs_accepted{suffix}.png"
            fig.savefig(output_dir_pre / fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {output_dir_pre / fname}")
