"""
library containing recursive equations for the Doppler and aberration kernel elements
"""

__author__ = " Siavash Yasini"
__email__ = "yasini@usc.edu"

import numpy as np
from . import FileHandler as fh
from . import MatrixHandler as mh
from scipy.misc import derivative
from scipy.special import factorial, comb

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)

sign_pref = {0: "0",
             1: "+",
             -1: "-",
            }


# ------------------------------
#      recursive relation
# ------------------------------

def get_K_d(K, d, s):
    """Calculate the Kernel of Doppler weight d, from the Doppler weight 1, using the recursive
    expressions in Yasini & Pierpaoli 2017 (http://arxiv.org/abs/1709.08298) Eq 15 & 16 and
    Dai & Chluba 2014 (http://arxiv.org/abs/1403.6117) Eqs 8 & 9


    Parameters
    ----------
    K:  object
        Kernel class instance
    d:  scalar
        Desired Doppler weight of the kernel
    s:  scalar
        Spin weight of the kernel

    Returns
    -------
        ndarray((lmax+1)*(lmax+2)/2,2*delta_ell+1): K_mLl matrix with Doppler weight d"""

    if d >= 1:
        # call the recursive function directly
        K_d_mLl = _K_d_lift(K, d, s)
        return K_d_mLl

    elif d < 1:
        # use the symmetry prooprty of the Kernel to save calculation time
        # convert d to positive number and use transpose of the Kernel
        K_d_mLl = _K_d_lift(K, 2 - d, -s)
        K_d_mlL = mh.transpose(K_d_mLl, K.delta_ell)

        return mh.minus_one_LplusLp(K.delta_ell, K.lmax) * K_d_mlL


def _K_d_lift(K, d, s):
    """
    Lift the Doppler weight of the kernel by d>1 using the recursive formula in
    Yasini & Pierpaoli 2017 (http://arxiv.org/abs/1709.08298) Eq 15 & 16

    Parameters
    ----------
    K:  object
        Kernel class instance
    d:  scalar
        Desired Doppler weight of the kernel
    s:  scalar
        Spin weight of the kernel

    Returns
    -------
        K_mLl matrix with Doppler weight d

    """

    assert(d >= 1)

    # no need to do anything if d=1
    if d == 1:
        return K._mLl_d1

    elif d > 1:
        key = sign_pref[np.sign(s)]+"d{}".format(d)
        logger.info("loading key {}".format(key))
        try:
            logger.info("loading key {}".format(key))
            K_d = fh.load_matrix(K.kernel_filename, key=key)
        except:
            logger.info("calculating key {}".format(key))

            # start the recursive calculation
            K_d_minusone = _K_d_lift(K, d-1, s)

            # calculate K_{ell', ell+1} and C_{ell+1,m} for Kernel weight d-1
            K_l_plusone_d_minusone = mh.shift_left(K_d_minusone)
            C_l_plusone = mh.shift_left(K.Cmatrix)

            # calculate K_{ell', ell-1} for Kernel weight d-1
            K_l_minusone_d_minusone = mh.shift_right(K_d_minusone)

            K_d = K.gamma*K_d_minusone + K.gamma*K.beta*(C_l_plusone * K_l_plusone_d_minusone
                                                         + np.sign(s) * K.Smatrix*K_d_minusone
                                                         + K.Cmatrix * K_l_minusone_d_minusone)

            logger.info("adding key {} to kernel file".format(key))
            fh.append_kernel(K.kernel_filename, K_d, key=key)

        return K_d


# ------------------------------
#     Generalized Kernel
# ------------------------------

def calc_K_d_arr(K, d, s):
    """
    Construct a kernel array where axis=0 corresponds to the weights d to d-beta_exp_order

    Parameters
    ----------
    K:
        Kernel class object
    d:
        Desired Doppler weight of the kernel
    s:
        Spin weight of the kernel

    Returns
    -------
        ndarray(beta_exp_order,(lmax+1)*(lmax+2)/2,2*delta_ell+1)
    """
    height, width = ((K.lmax+1)*(K.lmax+2)//2,2*K.delta_ell+1)
    K_d_arr = np.zeros((K.beta_exp_order+1,height,width))
    for i in range(d,d-K.beta_exp_order-1,-1):
        logger.info("d, i = {},{}".format(d,i))
        K_d_arr[d2indx(d,i)] = get_K_d(K,i,s)
    
    return K_d_arr


# FIXME: change derivative_dnu to dnu
def get_K_nu_d(K_d_arr, nu, pars, freq_func=None, return_normalize=True):
    """
    Calculate the frequency dependent generalized Kernel by adding the K_d array with the
    appropriate weights

    Parameters
    ----------
    K_d_arr: matrix
        output of the calc_K_d_arr function
    nu:     scalar
        frequency at which the kernel is calculated
    pars: dict
        parameter dictionary
    freq_func:  function
        frequency function of the observed radiation
        library of functions can be found in FreqyencyFunctions.py
    return_normalize: boolean
        if True, normalizes the output to temperature units

    Returns
    -------
    Generalized kernel elements of Doppler weight d at frequency nu

    """
    # extract some parameters
    beta_exp_order = pars['beta_exp_order']
    T = pars['T_0']
    dx = pars['derivative_dnu']

    # calculate the weights and add the kernel matrices together
    Kernel = 0.0
    for n in range(beta_exp_order+1):
        kfactor = 0.0
        for k in range(n+1):
            KLlm = K_d_arr[k]
            kfactor += KLlm * (-1.0)**(n+k) * comb(n, k)
        
        Kernel = Kernel + np.true_divide(kfactor, factorial(n))*nu**n * \
                            derivative(freq_func, nu, dx=dx, n=n, args=(T,), order=13)

    if return_normalize:
        return np.true_divide(Kernel, freq_func(nu, pars['T_0']))

    else:
        return Kernel


def d2indx(d,i):
    """convert the Doppler weight of the kernel to the index of the array"""
    return d-i

