Project Based Learning in Climat : Data Analysis
================================================

This part introduces to Empirical Orthogonal Functions that is tool to explore the variability of a dataset from the computation of the leading modes of variability.


The work is organized as a [notebook](./notebook/pbl_app-climat.ipynb).

This document reminds basic knowledge in dynamical system to focus on time series and phase space representations of the time evolution of a system.

Then Principal Analysis Components or Empirical Orthoogonal Functions are introduced for a random vector, then used to determine the leading mode of variability from a dataset.

The climate dataset used comes from a long time integration of 30 years from the Global Circulation Model PUMA. This dataset is used to calculate the univariate EOFs of the geopotential, then the multivariate EOFs of the combined fields of geopotential and temperature. We also consider the use of a metric to take into account the influence of the coordinate system.
