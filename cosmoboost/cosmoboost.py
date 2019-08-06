"""
Library containing the Kernel class and some boosting functions
"""
__author__ = "Siavash Yasini"
__email__ = "yasini@usc.edu"

import sys
import os
import numpy as np
import warnings
import pdb

from cosmoboost import COSMOBOOST_DIR
sys.path.insert(0, COSMOBOOST_DIR)

from lib import FileHandler as fh
from lib import FrequencyFunctions as ff
from lib import MatrixHandler as mh
from lib import KernelODE
from lib import KernelRecursive as kr

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_PARS = {
    'd' : 1, # Doppler weight
    's' : 0, # spin weight
    'beta' : 0.00123, #dimensionless velocity
    'lmin' : 0, # minimum ell mode
    'lmax' : 1000, # maximum ell mode
    'delta_ell' : 6, #number of ell neighbors on each side
    'T_0': 2.72548 ,# Kelvins
    'beta_expansion_order': 4, # expansion order of beta
    'derivative_dnu': 1.0, # resolution of frequency derivative in GHz
    'normalize': True, # normalize to temperature units
    'frequency_function': "CMB"
                }

# TODO: add custom frequency function
FREQ_DICT = {
    "CMB" : ff.F_nu,
    "tSZ" : ff.F_tSZ,
    }


##################################################
#           Kernel Class
##################################################

class Kernel(object):
    """Generalized Doppler and aberration Kernel class"""
    def __init__(self,
                 pars=DEFAULT_PARS,
                 overwrite=False,
                 save_kernel=True):

        self.d = pars['d']
        self.s = pars['s']
        self.beta = pars['beta']
        self.gamma = 1.0/np.sqrt(1-self.beta**2)
        self.T_0 = pars['T_0']
        self.derivative_dnu = pars['derivative_dnu']
        self.beta_exp_order = pars['beta_expansion_order']
        self.normalize = pars['normalize']
        self.overwrite = overwrite
        self.save_kernel = save_kernel
        self.freq_func = FREQ_DICT[pars["frequency_function"]]

        # set ell limits
        self.lmin = pars['lmin']
        assert self.lmin == 0, "lmin should be set to zero\n" \
                             "it is only included in the parameters for clarity"

        self.lmax = pars['lmax'] #+pars['delta_ell']  #TODO: add padding to this
                                                      #TODO: add private safe ell_max parameter

        # set delta_ell
        safe_delta_ell = np.min((4, np.round(self.beta * (2 * self.lmax))))
        try:
            self.delta_ell = pars['delta_ell']
            assert int(self.delta_ell) == self.delta_ell
            if self.delta_ell < safe_delta_ell:
                warnings.warn("The minimum suggested delta_ell is : {:d}".format(int(
                    safe_delta_ell)))
        except KeyError:
            self.delta_ell = safe_delta_ell

        # initialize other attributes
        self.pars = None  # dictionary of parameters
        self.kernel_filename = None
        self.matrices_filename = None

        # update the parameters
        self.update()
        
    def update(self):
        """update the parameters and evaluate the kernel elements"""

        self.pars = {
            'd' : self.d,
            's' : self.s,
            'beta' : self.beta,
            'lmin' : self.lmin,
            'lmax' : self.lmax,
            'delta_ell' : self.delta_ell,
            'T_0':self.T_0, # Kelvins
            'beta_exp_order': self.beta_exp_order,
            'derivative_dnu': self.derivative_dnu,
            'normalize': self.normalize
        }

        # determine file names based on parameters
        self.kernel_filename = fh.get_kernel_filename(self.pars)
        self.matrices_filename = fh.get_matrices_filename(self.pars)

        # initialize the matrices and coefficient matrix
        self._init_matrices()
        self._init_mLl()
        
        self.mLl = []

    # ------------------------------
    #     Matrix initialization
    # ------------------------------
    
    def _init_matrices(self):
        """initialize the kernel matrices"""

        # check to see if the file exists
        if fh.file_exists(self.matrices_filename) and self.overwrite is False:

            print("\nMatrices loaded from file:\n{}\n".format( self.matrices_filename))
            self._load_matrices()

        else:
            print("Calculating the index matrices...\n")
            self.Mmatrix, self.Lmatrix = mh.get_ML_matrix(self.delta_ell, self.lmax)
            _, Clms = mh.get_Blm_Clm(delta_ell=self.delta_ell, lmax=self.lmax, s=self.s)
            self.Cmatrix = Clms[self.Lmatrix, self.Mmatrix]
            self.Smatrix = mh.get_S_matrix(self.Lmatrix, self.Mmatrix, self.s)
        
            # save Mmatrix and Lmatrix for future use
            # matrices_file_name = fh.matrices_filename(pars)
            dir_name = fh.dirname(lmax=self.lmax, beta=self.beta)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                fh.save_matrices(self.matrices_filename, self.Mmatrix, 'M')
                fh.save_matrices(self.matrices_filename, self.Lmatrix, 'L')
                # generating Clm is relatively fast so there's no need to save to file
                
                print("Matrices saved in file:\n{}\n".format(self.matrices_filename))
                print("Done!\n")
    
    def _init_mLl(self):
    
        # initialize kernel with d=1    
        self._mLl_d1 = self._get_mLl_d1()
        # initialize (call setter) the mLl coefficients for d=1
        self._mLl = self._mLl_d1
        self._Ll = self._get_Ll()

    def _load_matrices(self):
        
        print("Loading the index matrices...\n")
        self.Mmatrix = fh.load_matrix(self.matrices_filename, key="M")
        self.Lmatrix = fh.load_matrix(self.matrices_filename, key="L")
        #self.Lmatrix = fh.load_matrix(self.matrices_filename,key="L")
        
        _, Clms = mh.get_Blm_Clm(delta_ell=self.delta_ell, lmax=self.lmax, s=self.s)
        self.Cmatrix = Clms[self.Lmatrix, self.Mmatrix]
        self.Smatrix = mh.get_S_matrix(self.Lmatrix, self.Mmatrix, self.s)
        print("Done!\n")

    # ------------------------------
    #     m, ell', ell index (mLl)
    # ------------------------------
    
    @property # mLl getter
    def mLl(self):
        # get values for mLl"
        return self._mLl
        
            
    @mLl.setter # mLl setter
    def mLl(self, value):
        if self.d == 1:
            # set d=1 values for mLl"
            self._mLl = self._mLl_d1
        else:
            # set d>1 values for mLl"
            self._mLl = self._get_mLl()

            
    @property #mLl getter
    def Ll(self):
        
        # get values for mLl"
        return self._get_Ll()
        
            
    @Ll.setter #mLl setter
    def Ll(self, value):
        if self.d == 1:
            # set d=1 values for mLl"
            self._Ll = self._get_Ll()
        else:
            # set d>1 values for mLl"
            self._Ll = self._get_Ll()


    def _get_mLl_d1(self):
        """return the DC aberration kernel elements K^m_{\ell' \ell} for d=1
        if the kernel has been calculated before, it will be loaded
        otherwise it will be calculated using the ODE"""

        # load the aberration kernel if it exists,
        # otherwise calculate it by solving the kernel_ODE
        if fh.file_exists(self.kernel_filename) and self.overwrite is False:
            print("Kernel loaded from file:\n {}\n".format(self.kernel_filename))
            K_mLl = fh.load_kernel(self.kernel_filename, key='D1')
        else: 
            print("Solving kernel ODE for d=1")
            K_mLl = KernelODE.solve_K_T_ODE(self.pars, save_kernel=self.save_kernel)

        return K_mLl


    def _get_mLl(self):
        """return the DC aberration kernel elements K^m_{\ell' \ell} for d!=1
        if the kernel has been calculated before, it will be loaded
        otherwise it will be calculated using the ODE"""
        if fh.file_exists(self.kernel_filename) and self.overwrite is False:
            print("Using Kernel for d=1 from file")
        else:
            print("Solving kernel ODE for d=1")
            self._get_mLl_d1()
        
        return kr.get_K_d(self, self.d, self.s)


    def nu_mLl(self, nu):
        """return the Doppler and aberration kernel elements K^m_{ell' ell} at frequency nu [GHz]"""
        K_d_arr = kr.calc_K_d_arr(self, self.d, self.s)

        if self.pars['normalize'] is True:
            print("Kernel elements are normalized\n")
        else:
            print("Kernel elements are not normalized\n")

        return kr.get_K_nu_d(K_d_arr, nu, self.pars, freq_func=self.freq_func,
                             return_normalize=self.pars['normalize'])


    def d_arrary(self):
        """calculate the generalized Doppler and aberration kernel elements for the relevant
        Doppler weights at frequency nu [GHz]"""

        # calculate the number of rows and columns required for the Kernel array
        height, width = ((self.lmax + 1) * (self.lmax + 2) / 2, 2 * self.delta_ell + 1)

        # initialize the Kernel array
        # elements of the first dimension holds terms of the beta expansion
        K_d_arr = np.zeros((self.beta_exp_order + 1, height, width))

        # calculate the Kernel array
        for i in range(self.d, self.d - self.beta_exp_order - 1, -1):
            K_d_arr[kr.d2indx(self.d, i)] = kr.get_K_d(self, i, self.s)

        return K_d_arr
    

    def _get_Ll(self):
        """returns the Boost Power Transfer Matrix (BPTM) K_{L,l} defined in Yasini &
        Pierpeoli 2017"""
        K_mLl = self.mLl
        K_Ll = np.zeros((self.lmax+1, 2*self.delta_ell+1))
        
        for L in range(self.lmax+1):
            # find all the m mode relevant for each ell'
            M = np.arange(self.lmin, L+1)

            # sum over all the m modes for each ell'
            # m=0 has to be removed because it's counted twice
            K_Ll[L, :] = 2*np.sum(K_mLl[mh.mL2indx(M, L, self.lmax), :]**2, axis=0)\
                                  - K_mLl[mh.mL2indx(0, L, self.lmax), :]**2

            K_Ll[L, :] /= 2*L+1

        return K_Ll

    # TODO: add Ll_nu function for boosting Cl in intensity
    def nu_Ll(self, nu):
        """returns the Boost Power Transfer Matrix (BPTM) K_{L,l} at frequency nu [GHz] defined in
        Yasini & Pierpeoli 2017"""
        K_nu_mLl = self.nu_mLl(nu)
        K_nu_Ll = np.zeros((self.lmax + 1, 2 * self.delta_ell + 1))

        for L in range(self.lmax + 1):
            # find all the m mode relevant for each ell'
            M = np.arange(self.lmin, L + 1)

            # sum over all the m modes for each ell'
            # m=0 has to be removed because it's counted twice
            K_nu_Ll[L, :] = 2 * np.sum(K_nu_mLl[mh.mL2indx(M, L, self.lmax), :] ** 2, axis=0) \
                           - K_nu_mLl[mh.mL2indx(0, L, self.lmax), :] ** 2

            K_nu_Ll[L, :] /= 2 * L + 1

        return K_nu_Ll


##################################################
#           boosting functions
##################################################

# ------------------------------
#           a_{ell, m}
# ------------------------------
def boost_alm(alm, kernel, *nu):
    """
    boost alm using the provided Doppler & aberration kernel

    Parameters
    ----------
    alm: array with shape (n,(lmax+1)*(lmax+2)/2)
        spherical harmonic multipole coeffient of the background radiation
    kernel: object
        an instance of the Doppler and aberration kernel
    nu [GHz]: scalar
        if provided the generalized Doppler and aberration kernel will be used at this frequency

    Returns
    -------
    The boosted a_lms

    """
    # TODO: implement parameter "beta_hat" arbitrary direction for the boost
    #   rotate with healpy.rotate_alm method

    if np.ndim(alm) != 1 and (alm.shape[0] not in (1, 3)):
        raise ValueError("alm should be either 1 dimensional (T) or 3 dimentional (T, E, B)")

    # add dimension to alm for consistency
    if np.ndim(alm) == 1:
        print("adding new axis to the input alm...\n")
        alm = alm[None, :]

    # slice the temperature alm
    almT = alm[0]
    
    # initialize the boosted_alm array (1 or 3 dimensional)
    boosted_alm = np.zeros(alm.shape, dtype=np.complex)
    
    # set the first column to boosted almT
    if nu:
        print("boosting T with nu [GHz] = {}".format(nu))
        boosted_alm[0] = _boost_almT(almT, kernel, nu[0])
    else:
        print("boosting T")
        boosted_alm[0] = _boost_almT(almT, kernel)

    # return boosted T if alm is 1 dim
    if alm.shape[0] == 1:
        print("Done!")
        return boosted_alm[0]

    # return boosted E and B as well, if alm is 3 dim
    if alm.shape[0] == 3:

        almE = alm[1]
        almB = alm[2]

        if nu:
            print("boosting E & B with nu = {}".format(nu))
            boosted_alm[1:3] = _boost_almEB(almE, almB, kernel, nu[0])
        else:
            print("boosting E & B")
            boosted_alm[1:3] = _boost_almEB(almE, almB, kernel)

    print("Done!")
    return boosted_alm


def _boost_almT(almT, kernel, *nu):
    """boost temperature multipoles almT (s=0)"""

    logger.info("boosting T_lm")

    if kernel.s != 0:
        kernel.s = 0
        kernel.update()

    lmax = kernel.lmax

    extention = kernel.delta_ell
    # pad the alm with zero
    almT = np.append(almT, np.zeros(extention))
    
    Mmatrix = kernel.Mmatrix
    Lmatrix = kernel.Lmatrix

    # if nu is provided calculate the kernel for intensity
    if nu:
        assert len(nu) == 1, "only one frequency should be provide "
        kernel.d = 3
        kernel.update()
        print("\nboosting intensity at nu = {} [GHz]\n\n".format(nu[0]))
        alm_boosted = np.sum(kernel.nu_mLl(nu[0]) * almT[mh.mL2indx(Mmatrix, Lmatrix, lmax)],
                             axis=1)
    # otherwise do it for temperature
    else:
        alm_boosted = np.sum(kernel.mLl * almT[mh.mL2indx(Mmatrix, Lmatrix, lmax)], axis=1)
    
    return alm_boosted


def _boost_almEB(almE, almB, kernel, *nu):
    """boost polarization multipoles almE and almB (s=2)"""

    logger.info("boosting E_lm and B_lm")

    if kernel.s != 2:
        kernel.s = 2
        kernel.update()
    if nu:
        kernel_plus = kernel.nu_mLl(nu[0])
    else:
        kernel_plus = kernel.mLl
    
    kernel.s = -2
    kernel.update()

    if nu:
        kernel_minus = kernel.nu_mLl(nu[0])
    else:
        kernel_minus = kernel.mLl

    kernelEE_mLl = 0.5 * (kernel_plus + kernel_minus)
    kernelEB_mLl = 0.5j * (kernel_plus - kernel_minus)

    lmax=kernel.lmax

    extention = kernel.delta_ell
    # pad the alm with zero
    almE = np.append(almE, np.zeros(extention))
    almB = np.append(almB, np.zeros(extention))

    Mmatrix = kernel.Mmatrix
    Lmatrix = kernel.Lmatrix
    
    # sort and prepare alms for direct multiplication with kernel.mLl
    alm_indx = mh.mL2indx(Mmatrix, Lmatrix, lmax)
    
    almE_boosted = np.sum(kernelEE_mLl*almE[alm_indx], axis=1) + np.sum(kernelEB_mLl*almB[
        alm_indx], axis=1)
    almB_boosted = np.sum(kernelEE_mLl*almB[alm_indx], axis=1) - np.sum(kernelEB_mLl*almE[
        alm_indx], axis=1)

    return np.vstack((almE_boosted, almB_boosted))


# ------------------------------
#           C_ell
# ------------------------------
def boost_Cl(Cl, kernel, *nu):
    """

    boost alm using the provided Doppler & aberration kernel

    Parameters
    ----------
    Cl: array with shape (n, lmax+1)
        spherical harmonic power spectrum of the background radiation
    kernel: object
        an instance of the Doppler and aberration kernel
    nu [GHz]: scalar
        if provided the generalized Doppler and aberration kernel will be used at this frequency

    Returns
    -------
    The boosted Cl
    """
    lmax = kernel.lmax
    delta_ell = kernel.delta_ell
    extention = (delta_ell*(2*lmax+1)+delta_ell**2)//2
    
    Cl_ext = np.append(Cl, np.zeros(extention))
    
    L = np.arange(lmax+1, dtype=int)
    ell = np.tensordot(L, np.ones(2 * delta_ell + 1, dtype=int), axes=0)\
                    + np.arange(-delta_ell, delta_ell+1, dtype=int)

    if nu:
        assert len(nu) == 1, "only one frequency (nu) can be provided"
        Cl_boosted = np.sum(kernel.nu_Ll(nu[0]) * Cl_ext[ell], axis=1)
    else:
        Cl_boosted = np.sum(kernel.Ll * Cl_ext[ell], axis=1)
    
    return Cl_boosted


