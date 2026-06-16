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

### Wave 3 — multi-hop / domain logic
inference/cluster: clustertdp, cluster_tp2tdp, clustertp_lowerbound, ctp_scores,
fgreedy, LCE, voxLCE, tfce, perm_tfce, real_tfce_clusters, perm_cluster,
localized_csi, cluster2csv, ctp_extract_score, rkval
inference/CopeSets (non-display): fdr_crs, fdr_crs_dep, fdr_simul_cs, scopes(loader),
scopes_lm(loader), sss_cope_sets, srf_fdr_crs, srf_scopes, srf_scb2cope,
srf_blue_crs, srf_color_crs, srf_redblue_crs
inference/Permutation: permutation_power, spintest(spin_surface,loader)
inference/BayesGLM: fit_classicalglm, fit_bayesglm, bayespw, runbayesglm
  (NOTE: fit_bayesglm wraps external INLA — flag as not-portable / needs discussion)
atlases: getBrainRegionNames, get_mask, getregion, atlas_masks

### Wave 4 — surface I/O + ops
surface: read_fs_geometry, freesurfer_read_surf, read_annotation, dpvread,
fs2surf, gifti2surf, load_gifti, fsannot2mask, loadmask, loadsrf, mgzwrite,
SurfStatEdg, SurfStatReadSurf, SurfStatReadSurf1, SurfStatSmooth, fs_smooth,
adjacency_matrix, graph_cc, resample_srf, resample_srf_nn, srf_face_area,
srf_fwhm2niters, smooth_surface, srf_dilate_mask, srf_contour, srf_noise,
spin_surface

### Wave 5 — viewing / GUI (NOT 1:1 — matplotlib/nilearn rewrites, port last)
statistics/aux: BigFont, animatefun, custom_colormap, mytiles, plotImagesInTile,
sliderGUI, slidergui3
brain I/O bridge: imgload, imgsave  (these are I/O, port early in practice via nibabel)
viewing: add_region, brainmove, colorRegion, combine_brains, overlay_brain,
overlay_brainslice, peak2circle, viewbrain, viewdata, viewdata2, viewthresh,
pan3, plot_compact, loadbrains, loadsubs
viewing/shape_screen: fullscreen, fullscreen2, screenshape, spherescreen,
squarescreen, surfscreen
surface plotting: srfplot, srfplot2, newfun(surf4), srfgui, srf_colour
copesets display: cope_display, srf_cope_display, srf_cope_display2, playcr

## Not directly portable (flag for discussion)
- fit_bayesglm / runbayesglm — wrap external INLA (R/C library). No clean Python equiv.
- extract_brain — shells out to Python HD-BET tool; keep as subprocess wrapper.
- mgzwrite — uses system gzip; use nibabel mgh writer instead.
- SurfStat* — vendored from external SurfStat package; reimplement minimally.
- GUI/slider/screen functions — interactive MATLAB uicontrol; no faithful Python analog.

## NOTE on imgload/imgsave
`imgload`/`imgsave` are used by many Wave 2 brain functions. In practice port these
FIRST (thin nibabel wrappers) even though MATLAB lists them under ImageViewing.
