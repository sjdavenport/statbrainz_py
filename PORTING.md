# StatBrainz MATLAB → Python porting map

Source: `/Users/samd/Documents/Code/Packages/Matlab/StatBrainz/`
Target: this package (`statbrainz`).

Goal: full, faithful port. Stack: numpy / scipy / nibabel / nilearn(+matplotlib for viewing).

Port **leaf functions first** (no internal StatBrainz deps), then work up the dependency
tree. Plotting/GUI functions port last and will NOT be 1:1 — they become matplotlib/nilearn
rewrites preserving the same inputs/outputs where sensible.

Package layout mirrors the MATLAB areas:

| MATLAB area          | Python module            |
|----------------------|--------------------------|
| Statistics_Functions | `statbrainz.statistics`  |
| Brain Functions      | `statbrainz.brain`       |
| Surface              | `statbrainz.surface`     |
| Inference            | `statbrainz.inference`   |
| Atlases              | `statbrainz.atlases`     |
| ImageViewing         | `statbrainz.viewing`     |

MATLAB→Python gotchas to apply throughout:
- 1-based → 0-based indexing (convind, plane2index, index2mask, etc. need care).
- Column-major (MATLAB) vs row-major (numpy): be explicit with `order='F'` when
  reshaping/vectorizing voxel data (vec_data, unwrap) to match MATLAB ordering.
- `end` slicing, `:` linearization → numpy `.ravel(order='F')`.
- Stats fns (norminv/normcdf/tinv/tcdf) → `scipy.stats`.
- imdilate/imfill/bwconncomp → `scipy.ndimage`.
- NIfTI I/O (imgload/imgsave) → `nibabel`.
- gifti / FreeSurfer surface reads → `nibabel.freesurfer` / `nibabel.gifti`.

## Porting order (waves)

### Wave 1 — pure-math leaves (no deps)  [STATUS: DONE — validated vs MATLAB R2024b]
DONE statistics: apower, asinh_trans, asinh_data_trans, convind, convindall,
FWHM2sigma, sigma2fwhm, nan2zero, zero2nan, lcdf, bernstd, distbn2pval,
tstat2pval, pval2tstat, mask_bounds, pad_im, square_signal, gmcdf, gmrnd
DONE statistics/aux: loader, modul, filesindir, statbrainz_maindir
DONE brain: plane2index, nifti_type, unwrap, vec_data
DONE inference/TDP: linear_template, inverse_linear_template, get_pivotal_stats
DONE inference/MHT: fdp_calc, fdrBH, fdrthresh
DONE inference/FirstLevel: prewhiten (faithful — reproduces MATLAB's buggy loop)
DEFER inference/FirstLevel: Xgen2 (needs fconv -> Wave 2)

### Wave 2 — one-hop (depend only on Wave 1)  [STATUS: core DONE — validated vs MATLAB]
DONE statistics: Gker, GkerMV, GkerMV2, Gkerderiv, Gkerderiv2, mvtstat,
dilate_mask, mask_bndry, expand2mask, doubleim, region_bndry2D, fast_conv,
gen_noise
DONE inference/MHT: imBH, imBH_data
DONE inference/cluster: numOfConComps, getlargestcluster, bestclusterslice,
index2mask, cluster_im
DONE inference/FirstLevel: Xgen2 (was deferred from Wave 1)
DONE brain I/O: imgload, imgsave (nibabel; imgsave diverges from SPM original)
DONE brain (MNI data bundled in statbrainz/data/Volume/): getMNImask, voxinMNI,
gen_mask, mvtstat_dep
TODO statistics: histpdf(fast_conv)  [plotting-ish]
TODO inference/MHT: spatialBH, bh_control, localized_vi
TODO inference/Resampling: perm_thresh (needs lmindices — Wave 3)

NOTE: bundled MNI data files copied into statbrainz/data/Volume/: MNImask.nii,
MNIbrain.nii.gz, MNIbrainmask.nii.gz. (ExData.nii / MNImaskNAN.nii @7MB not yet
copied — add if a function needs them.)

### Wave 3 — multi-hop / domain logic  [STATUS: core DONE — validated vs MATLAB]
DONE inference/cluster TDP+TFCE: rkval, clustertp_lowerbound, cluster_tp2tdp,
clustertdp (lowerbound method), tfce, voxLCE, LCE
DONE inference/MHT: spatialBH, localized_vi
DONE inference/CopeSets: fdr_crs
DONE inference/Resampling: perm_thresh (global-max path; mask path needs missing
lmindices)

NOT PORTABLE (broken / missing upstream deps in the MATLAB source itself):
- scopes, scopes_lm -> call fastperm/fastperm_mean/fastlmperm/lmthresh2scb,
  NONE of which exist anywhere in the StatBrainz MATLAB package.
- fdr_simul_cs -> MATLAB source is incomplete (empty `for`, undefined vars).
- bh_control, permutation_power, test_ica -> demo SCRIPTS, not functions.
- clustertdp 'heuristic'/'greedy', ctp_scores, fgreedy, cluster2csv,
  ctp_extract_score -> dispatch/parse EXTERNAL jobs (fgreedy CLI). Skipped.
- perm_tfce, real_tfce_clusters, perm_cluster, localized_csi -> TODO (need the
  permutation harness; revisit after surface so spintest can come too).

TODO inference/Permutation: spintest (needs spin_surface -> Wave 4)
TODO inference/BayesGLM: fit_classicalglm, fit_bayesglm, bayespw, runbayesglm
  (fit_bayesglm wraps external INLA — NOT portable; needs user decision)
DONE atlases (Harvard-Oxford 2mm bundled in statbrainz/data/Atlases/):
getBrainRegionNames, get_mask, getregion, atlas_masks — validated vs MATLAB
(48 cortical regions; Frontal Pole mask sum 15397; getregion lookups match).
findstrings/capstr (missing from MATLAB pkg) reimplemented inline.

### Wave 4 — surface I/O + ops  [STATUS: core DONE — validated vs MATLAB]
DONE surface core: SurfStatEdg, SurfStatSmooth, adjacency_matrix, graph_cc,
srf_face_area, srf_fwhm2niters, smooth_surface, resample_srf, resample_srf_nn
  (+ make_srf helper; srf is a dict with 0-based faces, nibabel convention)
DONE surface ops: srf_dilate_mask, srf_contour, srf_noise, spin_surface
DONE surface I/O (via nibabel): read_fs_geometry, freesurfer_read_surf,
read_annotation, fsannot2mask, fs2surf, gifti2surf, load_gifti, dpvread
DONE inference/Permutation: spintest (was deferred — needs spin_surface)

DONE surface loaders: loadsrf — validated vs MATLAB (fs5 white: 10242 verts,
20480 faces, face-area sum 66661.7988; hcp 32492 verts). Surface geometry was
extracted from the MATLAB .mat files and bundled as compressed .npz (faces
pre-converted to 0-based) under statbrainz/data/Surface/ (fsaverage3-7 + hcp).
TODO surface: loadmask -> needs the bilateral annotation files (fsaverage .annot)
SKIP surface: SurfStatReadSurf / SurfStatReadSurf1 (.obj/.fs readers — nibabel
covers these formats), mgzwrite (use nibabel mgh writer), fs_smooth (stub)

### Wave 5 — viewing  [STATUS: core DONE — helpers validated vs MATLAB]
DONE viewing helpers (pure arrays, 1:1, statbrainz.viewing.helpers):
colorRegion, peak2circle, viewthresh_image, combine_brains, custom_colormap
DONE viewing render (matplotlib rewrites, NOT 1:1, statbrainz.viewing.display):
viewthresh, viewdata, viewbrain, overlay_brain
  (matplotlib is an optional dep — the `viz` extra)

NOT PORTED — interactive GUIs / figure-window shaping with no Python analog:
brainmove, pan3, viewdata2, sliderGUI, slidergui3, srfgui, fullscreen,
fullscreen2, screenshape, spherescreen, squarescreen, surfscreen, plot_compact,
add_region, loadbrains, loadsubs, mytiles, plotImagesInTile, animatefun, BigFont
TODO (optional, later): surface plotting (srfplot/srfplot2/newfun/srf_colour) and
cope display (cope_display, srf_cope_display*) — 3D trisurf rendering, would be
a nilearn/matplotlib rewrite; not yet done.

## Not directly portable (flag for discussion)
- fit_bayesglm / runbayesglm — wrap external INLA (R/C library). No clean Python equiv.
- extract_brain — shells out to Python HD-BET tool; keep as subprocess wrapper.
- mgzwrite — uses system gzip; use nibabel mgh writer instead.
- SurfStat* — vendored from external SurfStat package; reimplement minimally.
- GUI/slider/screen functions — interactive MATLAB uicontrol; no faithful Python analog.

## NOTE on imgload/imgsave
`imgload`/`imgsave` are used by many Wave 2 brain functions. In practice port these
FIRST (thin nibabel wrappers) even though MATLAB lists them under ImageViewing.
