"""Embedding-space explorer for the object-view-selection pipeline.

Visualises the kMeans + constrained-xNN selection (``TopKMeansXNN``) before
top-k selection: an interactive 3D MDS of the kMeans clusters with quality
linked to dot opacity, plus a frame viewer with mask overlay.

The algorithms in :mod:`algorithms` are shared by the single frontend:

* :mod:`webapp` — single-browser-window plotly app (``python -m
  embedding_explorer_tool.webapp``).
"""
