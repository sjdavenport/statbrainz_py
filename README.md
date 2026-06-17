# statbrainz (Python)

Python port of the MATLAB [StatBrainz](https://github.com/sjdavenport/StatBrainz)
package: statistical inference and visualization of brain imaging data
(clustersize inference, TDP bounds, CoPE/confidence sets, volumetric and surface
reading/visualization).

Stack: numpy / scipy / nibabel, with matplotlib + nilearn for visualization.

## Install (development)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[test]"
```

## Status

This is an in-progress, faithful port. See [`PORTING.md`](PORTING.md) for the full
function inventory, dependency map, and porting order (leaf functions first).

Ported so far (Waves 1 & 2 — validated against MATLAB R2024b):
- `statbrainz.statistics`: transforms (`apower`, `asinh_trans`, `fwhm2sigma`, ...),
  distributions (`lcdf`, `bernstd`, `distbn2pval`, `tstat2pval`, `pval2tstat`,
  `gmcdf`, `gmrnd`), indexing (`convind`, `convindall`), masks (`nan2zero`,
  `zero2nan`, `mask_bounds`, `pad_im`, `square_signal`), kernels (`Gker`, ...),
  smoothing (`fast_conv`, `gen_noise`), morphology (`dilate_mask`, `mask_bndry`,
  `doubleim`, `region_bndry2D`, `expand2mask`), `mvtstat`, and utils.
- `statbrainz.brain`: `vec_data`, `unwrap`, `plane2index`, `nifti_type`,
  `imgload`, `imgsave`, `getMNImask`, `voxinMNI`, `gen_mask`, `mvtstat_dep`.
- `statbrainz.inference`: MHT (`fdrBH`, `fdrthresh`, `fdp_calc`, `imBH`,
  `imBH_data`, `spatialBH`, `localized_vi`), clusters (`numOfConComps`,
  `cluster_im`, `getlargestcluster`, `index2mask`, `bestclusterslice`),
  cluster-TDP (`tfce`, `voxLCE`, `LCE`, `rkval`, `clustertp_lowerbound`,
  `clustertdp`, `cluster_tp2tdp`), CoPE (`fdr_crs`), `perm_thresh`, `spintest`,
  TDP templates, `Xgen2`, `prewhiten`.
- `statbrainz.surface`: I/O (`fs2surf`, `gifti2surf`, `load_gifti`,
  `read_fs_geometry`, `read_annotation`, `fsannot2mask`, `dpvread`), ops
  (`SurfStatEdg`, `SurfStatSmooth`, `smooth_surface`, `adjacency_matrix`,
  `graph_cc`, `srf_face_area`, `srf_fwhm2niters`, `resample_srf`,
  `srf_dilate_mask`, `srf_contour`, `srf_noise`, `spin_surface`).

- `statbrainz.atlases`: `getBrainRegionNames`, `get_mask`, `getregion`,
  `atlas_masks` (Harvard-Oxford cortical + subcortical, 2mm).

Bundled data lives in `statbrainz/data/` (`Volume/` for MNI, `Atlases/` for the
Harvard-Oxford atlas).

Functions that could not be faithfully ported (broken/missing upstream MATLAB
deps, external tools, or demo scripts) are catalogued in
[`PORTING_ISSUES.md`](PORTING_ISSUES.md).

## Tests

```bash
python -m pytest
```
