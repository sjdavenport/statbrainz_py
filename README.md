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

## Modules

| Module | What's in it |
|---|---|
| `statbrainz.statistics` | transforms, distributions, Gaussian kernels, separable smoothing (`fast_conv`), mask morphology, voxelwise t-stats (`mvtstat`) |
| `statbrainz.brain` | voxel vectorization (`vec_data`/`unwrap`), NIfTI I/O (`imgload`/`imgsave`), MNI mask helpers |
| `statbrainz.inference` | FDR/MHT, connected-component clusters, TFCE, cluster-TDP bounds, CoPE sets, permutation thresholds, spin test |
| `statbrainz.surface` | FreeSurfer/GIfTI I/O, edge-based smoothing, adjacency, resampling, surface noise/spin |
| `statbrainz.atlases` | Harvard-Oxford region masks and coordinate lookups |
| `statbrainz.viewing` | image-montage helpers + matplotlib rendering |

## Quick start

```python
import numpy as np
from statbrainz import statistics as st, brain, inference as inf, atlases as at

# --- smoothing + voxelwise t-statistic -----------------------------------
rng = np.random.default_rng(0)
data = rng.standard_normal((20, 20, 30))      # [x, y, subjects]
data[5:10, 5:10, :] += 1.5                     # plant a signal blob
smoothed, ss = st.fast_conv(data, FWHM=3, D=3)
tstat, mean, sd, cohensd = st.mvtstat(data)

# --- FDR control ----------------------------------------------------------
pvals = np.array([0.001, 0.2, 0.04, 0.5])
rej_ind, n_rej, locs, maxp = inf.fdrBH(pvals, alpha=0.1)

# --- cluster inference + TDP bounds --------------------------------------
n_clusters, occ, sizes, idx = inf.numOfConComps(tstat, thresh=2, connectivity_criterion=8)
tfce_img = inf.tfce(tstat, H=2, E=0.5)

# --- CoPE confidence sets for {mu > thresh} ------------------------------
lower, upper = inf.fdr_crs(data, thresh=0.5, alpha_quant=0.05)

# --- MNI mask + atlas region ---------------------------------------------
mni = brain.imgload("MNImask")                 # bundled 2mm MNI mask
mask, indices, names = at.get_mask("HOc", "Frontal Pole")
region_names = at.getregion([39, 82, 23])      # 1-based voxel -> region name
```

Surface example:

```python
import numpy as np
from statbrainz import surface as sf

srf = sf.fs2surf("lh.white")                   # or gifti2surf(...)
data = sf.srf_noise(srf, FWHM=10, nsubj=1)     # smoothed surface noise
smoothed = sf.smooth_surface(srf, data, FWHM=5)
adj = sf.adjacency_matrix(srf)
n_clusters, *_ = sf.graph_cc(data, thresh=0, adj_matrix=adj)
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
