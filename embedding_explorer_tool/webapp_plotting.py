"""Plotly figure builder for the web app (2D or 3D).

The MDS projection itself is computed cluster-independently (``algorithms.project_mds``);
the cluster labels, xNN candidates, centroids and final picks are only post-hoc
markers/colors drawn on top of those fixed coordinates.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

CLUSTER_COLORS = px.colors.qualitative.Plotly
PICK_COLOR = "#FFD700"


def _rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _quality_colors(hex_color, quality):
    return [_rgba(hex_color, float(a)) for a in quality]


def _trace(dims, x, y, z, **kwargs):
    if dims == "3d":
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)
    return go.Scatter(x=x, y=y, **kwargs)


def build_figure(coords, quality, labels, result, pool_ids, dims="2d"):
    """Build the interactive MDS figure for a kMeans + xNN run.

    ``dims`` is ``"3d"`` (Scatter3d over the first three MDS components) or
    ``"2d"`` (Scatter over the first two).  Pool samples are drawn as dots
    coloured per cluster with opacity linked to their quality score; each
    cluster's constrained xNN candidates are shown as slightly larger dots
    with a thick black outline; the centroid (medoid frame) of every cluster
    is a larger star; the final pick is a gold star.
    """
    if dims not in ("2d", "3d"):
        dims = "3d"
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ids = np.asarray(pool_ids, dtype=int)
    quality = np.asarray(quality, dtype=float)
    clusters = result["clusters"]
    k = result["k"]

    x, y = coords[:, 0], coords[:, 1]
    z = coords[:, 2] if coords.shape[1] > 2 else None

    fig = go.Figure()

    legend_done = {"pool": False, "candidate": False, "centroid": False, "pick": False}

    for c in range(k):
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        mask = labels == c
        if not mask.any():
            continue

        custom = np.stack([ids[mask], quality[mask]], axis=1)
        fig.add_trace(
            _trace(
                dims, x[mask], y[mask], z[mask] if z is not None else None,
                mode="markers",
                marker=dict(
                    size=5,
                    color=_quality_colors(color, quality[mask]),
                ),
                customdata=custom,
                hovertemplate=_hovertemplate(),
                name="Pool samples" if not legend_done["pool"] else f"Cluster {c}",
                showlegend=not legend_done["pool"],
            )
        )
        legend_done["pool"] = True

        cand = clusters[c]["candidates"]
        if cand:
            cm = np.asarray(cand, dtype=int)
            fig.add_trace(
                _trace(
                    dims, x[cm], y[cm], z[cm] if z is not None else None,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=3, color="black"),
                    ),
                    customdata=np.stack([ids[cm], quality[cm]], axis=1),
                    hovertemplate=_hovertemplate(),
                    name="xNN candidates" if not legend_done["candidate"] else f"Cluster {c} xNN",
                    showlegend=not legend_done["candidate"],
                )
            )
            legend_done["candidate"] = True

    medoid_ids = [clusters[c]["medoid"] for c in range(k)]
    medoid_colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in range(k)]
    m = np.asarray(medoid_ids, dtype=int)
    fig.add_trace(
        _trace(
            dims, x[m], y[m], z[m] if z is not None else None,
            mode="markers+text",
            text=["\u2605"] * k,
            textposition="middle center",
            textfont=dict(size=18, color=medoid_colors),
            marker=dict(
                symbol="circle-open",
                size=20,
                color=medoid_colors,
                line=dict(width=3, color="black"),
            ),
            customdata=np.stack([ids[m], quality[m]], axis=1),
            hovertemplate=_hovertemplate(),
            name="Centroid (medoid frame)",
            showlegend=True,
        )
    )

    pick_ids = np.asarray(result["picks"], dtype=int)
    fig.add_trace(
        _trace(
            dims, x[pick_ids], y[pick_ids], z[pick_ids] if z is not None else None,
            mode="markers+text",
            text=["\u2605"] * len(pick_ids),
            textposition="middle center",
            textfont=dict(size=15, color=PICK_COLOR),
            marker=dict(
                symbol="circle-open",
                size=14,
                color=PICK_COLOR,
                line=dict(width=3, color="black"),
            ),
            customdata=np.stack([ids[pick_ids], quality[pick_ids]], axis=1),
            hovertemplate=_hovertemplate(),
            name="Final pick",
            showlegend=True,
        )
    )

    layout = dict(
        autosize=True,
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"),
    )
    if dims == "3d":
        layout["scene"] = dict(
            xaxis_title="MDS-1",
            yaxis_title="MDS-2",
            zaxis_title="MDS-3",
            bgcolor="rgba(0,0,0,0)",
        )
    else:
        layout["xaxis"] = dict(title="MDS-1", gridcolor="#3a3f55")
        layout["yaxis"] = dict(title="MDS-2", gridcolor="#3a3f55")
    fig.update_layout(**layout)
    return fig


def _hovertemplate():
    return "Frame %{customdata[0]}<br>quality %{customdata[1]:.3f}<extra></extra>"
