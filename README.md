# statbrainz

A Python port of the MATLAB [StatBrainz](https://github.com/sjdavenport/StatBrainz)
package for **statistical inference and visualization of brain imaging data** —
clustersize inference and TDP (true discovery proportion) bounds, CoPE /
confidence sets, TFCE, multiple-testing (FDR) control, permutation testing, and
reading/visualization of volumetric and surface data.

The port is faithful: numerical functions are cross-checked against the original
MATLAB (R2024b) output. It is built on the standard scientific-Python stack
(numpy / scipy / nibabel) rather than mirroring MATLAB idioms.

## Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .            # core
uv pip install -e ".[viz]"     # + matplotlib/nilearn for visualization
uv pip install -e ".[test]"    # + pytest
```

Core dependencies: `numpy`, `scipy`, `nibabel`. Visualization (`statbrainz.viewing`
rendering) requires the optional `viz` extra.

## Package layout

The package mirrors the MATLAB StatBrainz folder structure **one function per
file**, with each `.py` named exactly as the corresponding MATLAB `.m` (folder
names with spaces become underscores so they are importable):

```
statbrainz/
  Statistics_Functions/        apower.py, convind.py, gen_noise.py, ...
    ImageOperations/           FWHM2sigma.py, fast_conv.py, doubleim.py
      Kernels/                 Gker.py, GkerMV.py, ...
    Mask_functions/            nan2zero.py, dilate_mask.py, mask_bndry.py, ...
    Signal_generation/         square_signal.py
    Stats_functions/           mvtstat.py, lcdf.py, tstat2pval.py, ...
    Aux_functions/             Plotting/, SystemFunctions/
  Brain_Functions/             getMNImask.py, vec_data.py, voxinMNI.py, ...
  ImageViewing/                imgload.py, viewdata.py, combine_brains.py, ...
  Inference/
    MHT/                       fdrBH.py, imBH.py, spatialBH.py, ...
    ClusterInference/          numOfConComps.py, cluster_im.py,
      TFCE/                    tfce.py, voxLCE.py, LCE.py
      ClusterTDP/              clustertp_lowerbound.py, clustertdp.py, ...
    CopeSets/                  fdr_crs.py
    TDP_inference/templates/   linear_template.py, get_pivotal_stats.py, ...
    FirstLevel/  Resampling/  Permutation/
  Surface/                     loadsrf.py, smooth_surface.py, spin_surface.py, ...
    SurfStat/                  SurfStatEdg.py, SurfStatSmooth.py
    ReadSurfaceFiles/          fs2surf.py, read_fs_geometry.py, gifti/, freesurferfiles/
  Atlases/                     get_mask.py, getregion.py, atlas_masks.py, ...
```

A function can be imported from its mirror path, **or** flat from the top level:

```python
from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz import mvtstat        # same function, flat convenience API
```

## Quick start

```python
import numpy as np
import statbrainz as sb

# --- smoothing + voxelwise t-statistic -----------------------------------
rng = np.random.default_rng(0)
data = rng.standard_normal((20, 20, 30))      # [x, y, subjects]
data[5:10, 5:10, :] += 1.5                     # plant a signal blob
smoothed, ss = sb.fast_conv(data, FWHM=3, D=3)
tstat, mean, sd, cohensd = sb.mvtstat(data)

# --- FDR control ----------------------------------------------------------
pvals = np.array([0.001, 0.2, 0.04, 0.5])
rej_ind, n_rej, locs, maxp = sb.fdrBH(pvals, alpha=0.1)

# --- cluster inference + TDP bounds --------------------------------------
n_clusters, occ, sizes, idx = sb.numOfConComps(tstat, thresh=2, connectivity_criterion=8)
tfce_img = sb.tfce(tstat, H=2, E=0.5)

# --- CoPE confidence sets for {mu > thresh} ------------------------------
lower, upper = sb.fdr_crs(data, thresh=0.5, alpha_quant=0.05)

# --- MNI mask + atlas region ---------------------------------------------
mni = sb.imgload("MNImask")                    # bundled 2mm MNI mask
mask, indices, names = sb.get_mask("HOc", "Frontal Pole")
region_names = sb.getregion([39, 82, 23])      # 1-based voxel -> region name
```

Surface example:

```python
import numpy as np
import statbrainz as sb

srf = sb.fs2surf("lh.white")                   # or gifti2surf(...)
data = sb.srf_noise(srf, FWHM=10, nsubj=1)     # smoothed surface noise
smoothed = sb.smooth_surface(srf, data, FWHM=5)
adj = sb.adjacency_matrix(srf)
n_clusters, *_ = sb.graph_cc(data, thresh=0, adj_matrix=adj)
```

## Conventions / notes

- **Indexing.** Linear voxel indices and atlas coordinates follow MATLAB's
  **1-based, column-major (Fortran)** convention at the API boundary, so results
  match cross-referenced MATLAB code. `convind`, `index2mask`, `getregion`,
  cluster `PixelIdxList`-style indices, etc. are all 1-based. (Surfaces are the
  exception: `srf["faces"]` are 0-based, the nibabel convention.)
- **Bundled data.** `statbrainz/data/Volume/` holds the MNI volumes used by
  `imgload("MNImask")` etc.; `statbrainz/data/Atlases/` holds the Harvard-Oxford
  2mm atlas.
- **Visualization is not 1:1.** `statbrainz.viewing` re-expresses the MATLAB
  figure code on matplotlib. The pure image-building helpers (`combine_brains`,
  `colorRegion`, `custom_colormap`, ...) are faithful 1:1 ports; the interactive
  GUI/figure-shaping functions are intentionally not ported.

## What's not ported

Some MATLAB functions cannot be faithfully ported — their source is broken,
depends on code that is absent from the package, wraps an external tool (e.g.
INLA, FSL/SPM utilities), or is a demo script. Every such case is catalogued, with
the reason, in [`PORTING_ISSUES.md`](PORTING_ISSUES.md).

For the full per-function status and the leaf-first porting map, see
[`PORTING.md`](PORTING.md).

## Tests

```bash
python -m pytest
```

The test suite checks ported functions against values produced by the original
MATLAB implementation.

## License

See [`LICENSE`](LICENSE). Ported from StatBrainz by Samuel Davenport.
