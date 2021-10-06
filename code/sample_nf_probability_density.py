import torch
import torch_light
import numpy as np
from pathlib import Path
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_density

### sample the trained flow over some regular parameter grid
### to compute the probability density
### the spacing of this grid can be altered as desired
nlogm, nsfr, dz = 20, 60, 0.2
ndummy = 31
mmin, mmax = 8.6, 11.3
sfrmin, sfrmax = -2,2.5
zmin, zmax = 0.2, 3.0

### generate parameter grids
sfrgrid = np.linspace(sfrmin,sfrmax,nsfr)
mgrid = np.linspace(mmin,mmax,nlogm)
zgrid = np.arange(zmin,zmax+dz,dz)
nz = zgrid.shape[0]
dummy = np.linspace(-3.,3.,ndummy)

def do_all():
    """This function loads the trained normalizing flow, samples the probability density,
    optionally performs redshift smoothing, then plots the density at z=1 as a 
    simple working example.
    """

    flow = load_nf()
    prob_density = sample_density(flow,redshift_smoothing=True)
    plot_logm_logsfr(prob_density)

def plot_logm_logsfr(prob_density, ztarget=1.0):
    """Plot the logM--logSFR density at z=1 using the flow likelihood
    This is a simple working example of how to operate on the density
    """

    # find density at target redshift
    # note that prob_density has dimensions [n_mass, n_z, n_sfr]
    zidx = np.abs(zgrid-ztarget).argmin()
    density = prob_density[:,:,zidx]

    # Set density below the mass-complete limit to zero
    # This is an extrapolation of the density field, since no
    # galaxies of this mass were in the training set
    # We use 3D-HST since 3D-HST covers 0.5 < z < 3 while 
    # COSMOS15 covers 0.2 < z < 0.8; in the overlap region,
    # 3D-HST has a deeper mass-complete limit
    below_mcomp = mgrid < threedhst_mass_completeness(ztarget)
    density[below_mcomp,:] = np.nan

    # generate color between 'grid' points and transpose
    density = (density[1:,1:] + density[:-1,:-1])/2.

    # plot density
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    im = ax.pcolormesh(mgrid,sfrgrid,density.T,cmap='binary')

    # remove white lines
    im.set_edgecolor('face')

    # axis labels
    ax.set_xlabel('log(M/M$_{\odot}$)',fontsize=14)
    ax.set_ylabel('log(SFR/M$_{\odot}$/yr$^{-1}$)',fontsize=14)

    plt.tight_layout()
    plt.show()

def sample_density(flow, redshift_smoothing=True):
    """ This function computes probability density from the trained flow
    Optionally, it smooths over redshift with a Gaussian kernel of dz=0.1
    to smooth over spatial homogeneities (recommended)
    """

    ### loop over redshift grid and marginalize in each loop to save memory
    prob_full = np.zeros(shape=(nlogm,nsfr,nz))
    for i,zred in enumerate(zgrid):

        ### transform our regular grid into torch-appropriate inputs
        ### note that the ordering of variables must match the order during training
        x, y, z, d = zred,mgrid-10.,sfrgrid,dummy
        x,y,z,d = np.meshgrid(x, y, z, d)
        pgrid = np.stack([x,y,z,d], axis=-1)
        pgrid.shape = (-1, 4)
        pgrid = torch.from_numpy(pgrid.astype('f4')).to('cpu')

        ### calculate probability (== density)
        _ , prior_logprob, log_det = flow(pgrid)
        prob = np.exp((prior_logprob + log_det).detach().numpy())

        ### reshape output to match dimensions (logM, logSFR, z, dummy)
        prob = prob.reshape(nlogm,1,nsfr,ndummy)
        prob = np.swapaxes(prob,1,2)

        ### marginalize over `dummy' variable
        weights = norm_density.pdf(dummy, loc=0.0,scale=1.0)
        prob_full[:,:,i] = (weights[None,None,None,:] * prob).sum(axis=-1).squeeze() / weights.sum()

    ### Smooth in redshift to smooth over spatial inhomogeneities
    ### using a Gaussian kernel of width dz = 0.1
    if redshift_smoothing:
        kernel = Gaussian1DKernel(stddev=0.1/dz)
        prob_smooth = np.zeros(shape=(nlogm,nsfr,nz))
        for i in range(nlogm):
            for j in range(nsfr):
                prob_smooth[i,j,:] = convolve(prob_full[i,j,:],kernel,boundary='extend',preserve_nan=True)
        return prob_full
    else:
        return prob_smooth

def load_nf():
    """ This function instantiates the NormalizingFlow class
        and then loads the trained flow
    """

    ### use a trick to define a relative path
    loc = Path(__file__).parent / '../data/trained_flow_nsamp100.pth'

    ### fixed dimensions of the problem
    n_units = 5
    n_dim = 4

    ### instantiate flow
    flow = torch_light.NormalizingFlow(n_dim, n_units)

    #### load trained flow ###
    state_dict = torch.load(loc)
    flow.load_state_dict(state_dict)

    return flow


def cosmos15_mass_completeness(zred):
    """Returns log10(stellar mass) at which the COSMOS-2015 survey is considered complete
       at a given input redshift (zred)
       From Table 6 of Laigle+16, corrected to Prospector stellar mass.
    """

    zcosmos = np.array([0.175, 0.5, 0.8, 1.125, 1.525, 2.0])
    mcomp_prosp = np.array([8.57779419, 9.13212072, 9.54630419, 9.75007079,
                            10.10434753, 10.30023359])

    return np.interp(zred, zcosmos, mcomp_prosp)


def threedhst_mass_completeness(zred):
    """Returns log10(stellar mass) at which the 3D-HST survey is considered complete 
       at a given input redshift (zred)
       From Table 1 of Tal+14, corrected to Prospector stellar mass.
    """
    ztal = np.array([0.65, 1.0, 1.5, 2.1, 3.0])
    mcomp_prosp = np.array([8.86614882, 9.07108637, 9.63281923,
                            9.89486727, 10.15444536])
    return np.interp(zred, ztal, mcomp_prosp)