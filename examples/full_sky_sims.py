import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'stix'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (6.5, 4)
plt.rcParams['figure.dpi'] = 150

import healpy as hp
import cosmoboost as cb
from tqdm import trange
from tqdm import tqdm

# read the default parameters from cosmoboost
pars = cb.DEFAULT_PARS

#-------------------
#     parameters
#-------------------

# cosmoboost parameters
lmax = pars['lmax'] = 500
delta_ell = pars['delta_ell'] = 8
pars['d'] = 1
beta = pars['beta']
T_0 = pars["T_0"]

# simulation parameters
n_sims = 10
show_plot = True

# calculate the kernel
kernel = cb.Kernel(pars)
ell = np.arange(lmax + 1)

# load the sample power spectrum
lib_dir = os.path.join(cb.COSMOBOOST_DIR, "lib")
Cl_camb = np.load(os.path.join(lib_dir, "sample_Cl.npz"))

Cl_TT = Cl_camb["TT"][:lmax + 1]



if __name__ == "__main__":

    # simulate the rest frame alms
    alm_T_r_arr = [hp.synalm(Cl_TT, lmax=lmax, new=True, verbose=True)
        for _ in trange(n_sims)]

    # boost the alms
    alm_T_b_arr = [cb.boost_alm(alm_T_r, kernel)
                   for alm_T_r in tqdm(alm_T_r_arr, total=n_sims)]


    #alms = np.array(alm_T_r_arr, alm_T_b_arr)

    np.savez(f"alms_r_b_lmax={lmax}", r=alm_T_r_arr, b=alm_T_b_arr)

    if show_plot:

        Cl_r = hp.alm2cl(alm_T_r_arr[0])
        Cl_b = hp.alm2cl(alm_T_b_arr[0])

        plt.plot(ell, 100*(Cl_b/Cl_r-1))
        plt.xlabel("$\ell$")
        plt.ylabel("$C_\ell^b/C_\ell^r$-1 (\%)")
        plt.show()

