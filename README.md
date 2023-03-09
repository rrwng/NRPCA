## Manifold Denoising by Nolinear Robust Principal Analysis (NRPCA)

This is the repository for code related to 'Manifold Denoising by Nolinear Robust Principal Analysis'. The paper is available on the Arxiv at: https://arxiv.org/abs/1911.03831.

A Python implementation can be found at https://github.com/lyuhe95/NRPCA_python.

***

### Overview
We extend robust principal component analysis to nolinear manifolds, where we assume that the data matrix contains a sparse component and a component drawn from some low dimensional manifold. We aim at separating both components from noisy data by proposing an optimization framework.

***

### Descriptions
**data**: contains data for numerical simulation

**dependencies**: contains other pacakges used in the implementation

**result**: contains results for the two examples in the paper

**src**: contains source codes for NRPCA

**Example_MNIST.m**: Code for MNIST digits 4&9 classification using NRPCA

**Example_SwissRoll.m**: Code for 20 dimenssional SwissRoll dataset using NRPCA

**setup.m**: add paths to run examples.

