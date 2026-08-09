"""Embedding-space explorer for the object-view-selection pipeline.

Visualises the kMeans + constrained-xNN selection (``TopKMeansXNN``) before
top-k selection: an interactive 3D MDS of the kMeans clusters with quality
linked to dot opacity, plus a frame viewer with mask overlay.

The algorithms in :mod:`algorithms` are shared by two frontends:

* :mod:`webapp` — single-browser-window plotly app (``python -m
  embedding_explorer_tool.webapp``).
* :mod:`prefilter_app` — pre-filter threshold tuner on a separate port
  (``python -m embedding_explorer_tool.prefilter_app``): tune the garbage /
  outlier thresholds, preview the accept/reject outcome, then generate the
  snapshot the webapp reads.
"""
