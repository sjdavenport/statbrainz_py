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
  `imBH_data`), clusters (`numOfConComps`, `cluster_im`, `getlargestcluster`,
  `index2mask`, `bestclusterslice`), TDP templates, `Xgen2`, `prewhiten`.

Bundled MNI data lives in `statbrainz/data/Volume/`.

## Tests

```bash
python -m pytest
```
