# sfs_leja_trained_flow
This repository includes the trained normalizing flow describing the galaxy star-forming sequence from Leja et al. 2021. The normalizing flow is trained using code from Green & Ting 2020.

The trained flow is stored in `data`. The code to interact with this flow is in `sample_nf_probability_density.py`, including stellar mass-complete limits.
You can interact with the trained data using the `do_all` function in this file. A second file, `torch_light.py`, contains all of the helper functions
needed to interpret the flow. Normal usage will not require interacting with this code.

The package dependencies include torch, astropy, scipy, numpy, and matplotlib.
