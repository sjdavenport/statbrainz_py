# StatBrainz port — functions with issues

This report lists every MATLAB function encountered so far that **could not be
ported faithfully (or at all)**, and why. "Faithful" means producing the same
results as the MATLAB source; a function whose MATLAB source is broken or calls
code that does not exist in the package cannot be faithfully reproduced.

All "missing dependency" claims below were verified by searching the entire
MATLAB StatBrainz tree (`find . -name '<fn>.m'`) — the named functions are
defined in **zero** files in the package.

Legend:
- **BROKEN** — the MATLAB source itself is incomplete / will error.
- **MISSING DEP** — calls a function not present anywhere in the package.
- **EXTERNAL** — depends on an external tool / library with no clean Python equivalent.
- **SCRIPT** — a demo/example script, not a callable function.
- **DEFERRED** — portable, but waiting on an earlier wave or a data/decision step.

---

## 1. BROKEN in the MATLAB source

| Function | Area | Problem |
|---|---|---|
| `fdr_simul_cs` | Inference/CopeSets | Source is incomplete: contains an empty `for` line, and references undefined variables (`thresh`, `data_tstat`, `data_std`) that are never assigned in the in-mask code path. Cannot run as written. |

## 2. MISSING DEPENDENCY (callee absent from the whole package)

| Function | Area | Missing callee(s) | Notes |
|---|---|---|---|
| `scopes` | Inference/CopeSets | `fastperm`, `fastperm_mean` | Bootstrap/permutation helpers — defined nowhere in the package. |
| `scopes_lm` | Inference/CopeSets | `fastlmperm`, `lmthresh2scb` | Linear-model SCB permutation — callees absent. |
| `perm_thresh` (mask path only) | Inference/Resampling | `lmindices` | The **global-maximum path IS ported**; only the optional `mask` branch needs `lmindices` (local-maxima indices), which is absent. Passing a `mask` raises `NotImplementedError`. |
| `sss_cope_sets` | Inference/CopeSets | (not yet audited line-by-line) | Flagged for review when CopeSets is revisited. |

> Note: several of these also reference `fconv`, `noisegen`, `wfield` — these
> are RFTtoolbox functions assumed on the MATLAB path, not part of StatBrainz.
> Where StatBrainz ships its own equivalent (`fast_conv`), the port uses it.

## 3. EXTERNAL tool / library (no clean Python equivalent)

| Function | Area | Depends on | Decision |
|---|---|---|---|
| `fit_bayesglm` | Inference/BayesGLM | external **INLA** (R/C library) | Not portable as-is. Needs a user decision (skip, stub, or reimplement with PyMC/numpyro). |
| `runbayesglm` | Inference/BayesGLM | `fit_bayesglm` | Blocked on the above. |
| `bayespw`, `fit_classicalglm` | Inference/BayesGLM | (partial) | Revisit with the BayesGLM decision. |
| `clustertdp` (`heuristic`/`greedy` mode) | Inference/ClusterInference | `fgreedy` CLI jobs, `monitor_cTDP` | The **`lowerbound` mode IS ported**; the heuristic mode dispatches external compute jobs and parses their logs. Raises `NotImplementedError`. |
| `ctp_scores`, `ctp_extract_score`, `fgreedy`, `cluster2csv` | Inference/ClusterInference | external `fgreedy` job runner + its log files | Skipped (tooling glue, not numerical methods). |
| `extract_brain` | Statistics_Functions/FirstLevel | shells out to Python **HD-BET** | Port later as a thin `subprocess` wrapper if needed. |
| `mgzwrite` | Surface | system `gzip` + custom mgh writer | Use `nibabel` mgh writer instead when reached (Wave 4). |
| `imgsave` (SPM specifics) | ImageViewing | `spm_vol`, `spm_write_vol`, global path vars | Ported as a **nibabel writer** with explicit path — intentional, documented divergence from the SPM-tied original. |

## 3b. INTERACTIVE GUI / FIGURE-SHAPING (no faithful Python analog)

These use MATLAB `uicontrol` sliders, `figure`/`gcf` window shaping, or live
`patch` callbacks. They were intentionally **not ported**; the renderable parts
were re-expressed on matplotlib in `statbrainz.viewing.display` where it made
sense (viewthresh, viewdata, viewbrain, overlay_brain).

| Function(s) | Area | Reason |
|---|---|---|
| `brainmove`, `pan3`, `viewdata2` | ImageViewing | Interactive `uicontrol` navigation. |
| `sliderGUI`, `slidergui3`, `srfgui` | Aux/Surface | Interactive slider GUIs. |
| `fullscreen`, `fullscreen2`, `screenshape`, `spherescreen`, `squarescreen`, `surfscreen` | ImageViewing/Shape_screen | Figure-window shaping. |
| `plot_compact`, `mytiles`, `plotImagesInTile`, `animatefun`, `BigFont`, `add_region` | ImageViewing/Aux | Figure layout / animation / patch overlays. |
| `srfplot`, `srfplot2`, `newfun` (surf4), `srf_colour`, `cope_display`, `srf_cope_display*`, `playcr` | Surface/CopeSets | 3D `trisurf` rendering (a nilearn/matplotlib rewrite — TODO, not yet done). |

## 4. SCRIPTS (not functions — nothing to port)

| Name | Area | Notes |
|---|---|---|
| `bh_control` | Inference/MHT | FDR-control simulation/demo script. |
| `permutation_power` | Inference/Permutation | 2-line stub script. |
| `test_ica` | Inference/ICA | Demo script. |
| `store_atlases`, `ex_loadatlases`, `HO_derived_round_linearinterp` | Atlases | Setup/example scripts. |
| `dilmask`, `fullLMcalc` | Brain Functions | Demo scripts (the real functions live elsewhere, e.g. `dilate_mask`). |

## 5. DEFERRED (portable; blocked on an earlier step)

| Function | Area | Waiting on |
|---|---|---|
| `histpdf` | Statistics_Functions | Plotting-adjacent (wraps MATLAB `histogram`); Wave 5/viewing. Numerical dep `fast_conv` is ported, so a numpy rewrite is trivial when wanted. |

> **Wave 6 — now DONE (no longer deferred):**
> - `perm_tfce`, `real_tfce_clusters`, `perm_cluster`, `localized_csi`
>   (Inference/ClusterInference) — sign-flip permutation cluster/TFCE inference.
>   RNG is numpy-based (seedable via `rng=`); the permutation stream differs
>   from MATLAB by construction, but the deterministic structure and the
>   observed-statistic entries are validated.
> - `loadmask` (Surface) — fsaverage3-7 `lh/rh.aparc.annot` bundled under
>   `statbrainz/data/Surface/fsaverageN/`; returns `{'lh','rh'}` boolean masks.
>
> **Confirmed NOT portable (verified against the real `.m` source):**
> - `peakgen` (Statistics_Functions/Signal_generation) — calls `SpheroidSignal`,
>   an RFTtoolbox function present in **zero** files of the StatBrainz package.
> - `fs_smooth` (Surface) — the MATLAB function body is empty (a stub).
>
> `spintest` is **already ported** (Wave 4, after `spin_surface`).

> Atlases (`getBrainRegionNames`, `get_mask`, `getregion`, `atlas_masks`) are now
> DONE — the Harvard-Oxford 2mm atlas was bundled into `statbrainz/data/Atlases/`.
> Only the larger/derived atlas variants (1mm, Juelich, Talairach, JHU, the `.mat`
> precomputed masks) were not copied; add them if needed.

---

_Last updated after Wave 6. See `PORTING.md` for the full wave-by-wave status._
