# Task 1 — Explorer: pure MDS + white background + 2D/3D toggle

## Goal
- The explorer MDS projection must be a **pure MDS** DR of the embedding space
  (no knowledge of found clusters). The found clusters and xNN candidates are
  only *colored/marked into* that fixed projection afterwards.
- Bright/white background in the tkinter app so axes/labels are readable.
- **2D** view in the webapp (the "shady" 3D is replaced by a clean 2D scatter),
  with a UI element in **both** apps to switch between 2D and 3D.

## Current state
- `embedding_explorer_tool/gui_tk.py`: tkinter + embedded matplotlib,
  `matplotlib.use("TkAgg")`; `plot_result(ax, coords, quality, labels, result, pool_ids, dims="2d")`
  draws clusters + xNN + centroids + gold final pick. Has `View: 2D / 3D` radios
  that re-run `project_mds(n_components=2|3)`. **White** `rcParams` + white
  Figure/axes + light right panel already applied.
- `embedding_explorer_tool/webapp_plotting.py`: `build_figure(coords, quality, labels,
  result, pool_ids, dims="3d")` branches between `Scatter3d` / `Scatter`.
- `embedding_explorer_tool/webapp.py`: `/api/run?k&init&x&dims=`; response echoes `dims`.
- `embedding_explorer_tool/webapp_template.html`: dark theme, topbar with a
  **2D/3D segmented switch** (`dims_2d` / `dims_3d` buttons), client re-render.
- `embedding_explorer_tool/algorithms.py`: `project_mds(embeddings, n_components=3)`
  — metric MDS over cosine distance, already cluster-independent.

## Work items
1. **Pure MDS guarantee** — DONE. `project_mds` computes MDS on `embeddings`
   alone; clusters/candidates/picks are only marker/colour overlays.
2. **White background (tkinter)** — DONE (rcParams + Figure/axes facecolor +
   tk widget backgrounds).
3. **2D/3D toggle (webapp)** — DONE (`build_figure(dims=...)`, template switch,
   `dims` query param).
4. **2D/3D toggle (tkinter)** — DONE (radios re-project MDS to 2/3 dims).
5. **Webapp defaults to 2D** — DONE. `currentDims = "2d"` in the template with
   `dims_2d` active on first load; `webapp.py` `api_run(..., dims="2d")` default
   and `_serve_api` `params.get("dims", ["2d"])`; `build_figure(..., dims="2d")`.
   Verified over HTTP: no `dims` param → `scatter` traces + `"dims":"2d"`;
   `dims=3d` → `scatter3d`.
6. **Docs** — DONE. `docs/explorer.md` updated (pure-MDS behaviour, white theme,
   2D/3D switch, 2D is the default view).

## Verification
- `build_figure(..., dims="2d")` returns `scatter` traces; `dims="3d"` returns `scatter3d`.
- Template default is 2D on first load.
- tkinter white background + toggles verified via headless `plot_result` on
  both a rectilinear and a 3D axes.
- Live HTTP check confirmed the 2D default and the `dims=3d` switch.
