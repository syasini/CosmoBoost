# coding: utf-8

__author__ = "Siavash Yasini"
__email__ = "yasini@usc.edu"

import sys
import os
import numpy as np
import pdb

np.set_printoptions(precision=4)

test_par = 21

from cosmoboost import COSMOBOOST_DIR
#COSMOBOOST_DIR = os.path.dirname(os.path.realpath(__file__)) #os.getcwd()

sys.path.insert(0,COSMOBOOST_DIR)

from lib import FileHandler as fh
from lib import FrequencyFunctions as ff
from lib import MatrixHandler as mh
from lib import KernelODE
from lib import KernelRecursive as kr


DEFAULT_PARS = {
    'd' : 1,
    's' : 0,
    'beta' : 0.00123,
    'lmin' : 0,
    'lmax' : 1000,
    'delta_ell' : 3,
    'T_0':2.72548 ,#Kelvins
    'beta_exp_order':4, 
    'derivative_dnu':1.0,
    'normalize': True,
    'frequency_function': "CMB"
}

FREQ_DICT = {
    "CMB" : ff.F_nu,
    "tSZ" : ff.F_tSZ,
    }

class Kernel(object):
    """Generalized Doppler and aberration Kernel class"""
    def __init__(self, pars=DEFAULT_PARS,
                 overwrite=False,save_kernel=True):
        
        # self._pars = pars
        
        self.d = pars['d']
        
        self.s = pars['s']
        self.beta = pars['beta']
        self.gamma = 1.0/np.sqrt(1-self.beta**2)
        self.lmin = pars['lmin']
        self.lmax = pars['lmax'] #+pars['delta_ell']
        self.T_0 = pars['T_0']
        self.derivative_dnu = pars['derivative_dnu']
        self.normalize = pars['normalize']
        #self._lmax_safe = pars['lmax']
        self.delta_ell = pars['delta_ell']
        self.beta_exp_order = pars['beta_exp_order']
        
        self.overwrite= overwrite
        self.save_kernel=save_kernel

        self.freq_func = FREQ_DICT[pars["frequency_function"]]
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
            'T_0':self.T_0 ,#Kelvins
            'beta_exp_order':self.beta_exp_order,
            'derivative_dnu':self.derivative_dnu,
            'normalize': self.normalize
        }
        self.gamma = 1.0/np.sqrt(1-self.beta**2)
        
        self.kernel_filename = fh.kernel_filename(self.pars)
        self.matrices_filename = fh.matrices_filename(self.pars)

        self._init_matrices()
        self._init_mlpl()
        
        
        self.mlpl =[]

    def _init_matrices(self):
        """initialize the kernel matrices"""

        # check to see if the file exists
        if fh.file_exists(self.matrices_filename) and self.overwrite==False:

            print ("\nMatrices loaded from file: {} \n".format( self.matrices_filename))
            self._load_matrices()

        else:

            print ("Calculating the index matrices...")
            self.Mmatrix, self.Lmatrix = mh.ML_matrix(self.delta_ell,self.lmax)
            _,Clms = mh.Blm_Clm(delta_ell=self.delta_ell, lmax=self.lmax,s=self.s)
            self.Cmatrix = Clms[self.Lmatrix,self.Mmatrix]
            self.Smatrix = mh.S_matrix(self.Lmatrix,self.Mmatrix,self.s)
        
            #save Mmatrix and Lmatrix for future use
            #matrices_file_name = fh.matrices_filename(pars)
            dir_name = fh.dirname(lmax=self.lmax,beta=self.beta)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                fh.save_matrices(self.matrices_filename,self.Mmatrix,'M')
                fh.save_matrices(self.matrices_filename,self.Lmatrix,'L')
                print ("\nMatrices saved in: {} \n".format(self.matrices_filename))

                print ("Done!\n")
    
    def _init_mlpl(self):
    
        #initialize kernel with d=1    
        self._mlpl_d1 = self._get_mlpl_d1()
        #initialize (call setter) the mlpl coefficients for d=1
        self._mlpl  = self._mlpl_d1 
        #self._mlpl = self.mlpl
        self._lpl    = self._get_lpl()

    def _load_matrices(self):
        
        print ("Loading the index matrices...")
        self.Mmatrix = fh.load_matrix(self.matrices_filename,key="M")
        self.Lmatrix = fh.load_matrix(self.matrices_filename,key="L")
        #self.Lpmatrix = fh.load_matrix(self.matrices_filename,key="Lp")
        _,Clms = mh.Blm_Clm(delta_ell=self.delta_ell, lmax=self.lmax,s=self.s)
        self.Cmatrix = Clms[self.Lmatrix,self.Mmatrix]
        self.Smatrix = mh.S_matrix(self.Lmatrix,self.Mmatrix,self.s)
        print("Done!\n")

    
    @property #mlpl getter
    def mlpl(self):
        
        #print "get values for mlpl"
        return self._mlpl
        
            
    @mlpl.setter #mlpl setter
    def mlpl(self, value):
        if self.d == 1:
            # set d=1 values for mlpl"
            self._mlpl = self._mlpl_d1
        else:
            # set d>1 values for mlpl"
            self._mlpl = self._get_mlpl()

            
    @property #mlpl getter
    def lpl(self):
        
        # get values for mlpl"
        return self._get_lpl()
        
            
    @lpl.setter #mlpl setter
    def lpl(self, value):
        if self.d == 1:
            #set d=1 values for mlpl"
            self._lpl = self._get_lpl()
        else:
            #set d>1 values for mlpl"
            self._lpl = self._get_lpl()



    def _get_mlpl_d1(self):
        '''return the DC aberration kernel elements K^m_{\ell' \ell} for d=1
        if the kernel has been calculated before, it will be loaded
        otherwise it will be calculated using the ODE'''
        
        #directory for saving/loading the aberration kernel elements 
        
        
        #load the aberration kernel if it exists,
        #otherwise calculate it by solving the kernel_ODE
        if fh.file_exists(self.kernel_filename) and self.overwrite==False:
            print("Kernel loaded from file: {}".format(self.kernel_filename))
            K_mlpl = fh.load_kernel(self.kernel_filename,key='D1')
        else: 
            print ("Solving kernel ODE for d=1")
            K_mlpl = KernelODE.solve_K_T_ODE(self.pars,save_kernel=self.save_kernel)
        
        
        return K_mlpl
        
    def _get_mlpl(self):
        '''return the DC aberration kernel elements K^m_{\ell' \ell} for d!=1
        if the kernel has been calculated before, it will be loaded
        otherwise it will be calculated using the ODE'''
        if fh.file_exists(self.kernel_filename) and self.overwrite==False:
            print ("Kernel loaded from file: "+ self.kernel_filename)
        else:
            print ("Solving kernel ODE for d=1")
            self._get_mlpl_d1()
        
        return kr.K_d(self,self.d,self.s)
    
    def nu_mlpl(self,nu):
        print ("d={}".format(self.d))
        K_d_arr = kr.calc_K_d_arr(self,self.d,self.s)

        if self.pars['normalize']==True:
            print ("normalized\n")
        else:
            print ("not normalized\n")

        return kr.K_nu_d(K_d_arr,nu,self.pars,freq_func=self.freq_func,
                             return_normalize=self.pars['normalize'])

    def d_arrary(self):
        #return kr.get_K_d_arr(self,self.d,self.s)
        height, width = ((self.lmax + 1) * (self.lmax + 2) / 2, 2 * self.delta_ell + 1)
        K_d_arr = np.zeros((self.beta_exp_order + 1, height, width))
        for i in range(self.d, self.d - self.beta_exp_order - 1, -1):
            print ("d, i = {},{}".format(self.d, i))
            K_d_arr[kr.d2indx(self.d, i)] = kr.K_d(self, i, self.s)

        return K_d_arr
    
    
        
    def _get_lpl(self):
        '''returns the Boost Power Transfer Matrix (BPTM) K_{lp,l} defined in Yasini &
        Pierpeoli 2017'''
        K_mlpl = self.mlpl
        K_lpl = np.zeros((self.lmax+1,2*self.delta_ell+1))
        
        for lp in range(self.lmax+1):
            M = np.arange(self.lmin,lp+1)
        #K_delta[lp]=2*np.sum(K_mlp[fh.linenumb_mlp_vec(M,lp,lmax=lmax)])-K_mlp[fh.linenumb_mlp_vec(0,lp,lmax=lmax)]
            K_lpl[lp,:]=2*np.sum(K_mlpl[mh.mlp2indx(M,lp,self.lmax),:]**2,axis=0)-K_mlpl[mh.mlp2indx(0,lp,self.lmax),:]**2
            K_lpl[lp,:] /= 2*lp+1
        return K_lpl



        
def boost_alm(alm,kernel,*nu):
    ''' boost alm with the shape (n,(lmax+1)*(lmax+2)/2)
    where n = 1 for T only
    and   n = 3 for (T,E,B)
    if frequency nu is provided, the generalized aberation kernel coefficients will be used'''

    if (np.ndim(alm)!=1  and (alm.shape[0] not in (1,3)) ):
        raise ValueError("alm should be either 1 dimensional (T) or 3 dimentional (T, E, B)")

    if np.ndim(alm) == 1:
        print ("adding new axis to alm...\n")
        alm = alm[None,:]
    #slice the temperature alm
    almT = alm[0]
    
    #initialize the boosted_alm array (1 or 3 dimensional)
    boosted_alm = np.zeros(alm.shape,dtype=np.complex)
    
    #set the first column to boosted almT
    if nu:
        print ("boosting T with nu = {}".format(nu))
        boosted_alm[0] = _boost_almT(almT,kernel,nu[0])
    else:
        boosted_alm[0] = _boost_almT(almT, kernel)
    #return boosted T if alm is 1 dim
    if alm.shape[0] == 1:
        return boosted_alm[0]
    
    #return boosted E and B as well, if alm is 3 dim
    if alm.shape[0] == 3:

        almE = alm[1]
        almB = alm[2]
        
        #boosted_almE = np.zeros(almE.shape)
        #boosted_almB = np.zeros(almB.shape)

        if nu:
            print ("boosting EB with nu = {}".format(nu))
            boosted_alm[1:3] = _boost_almEB(almE,almB,kernel,nu[0])
        else:
            boosted_alm[1:3] = _boost_almEB(almE, almB, kernel)
        
    

    return boosted_alm




def _boost_almT(almT,kernel,*nu):
    '''boost temperature multipoles almT (s=0)'''
    

    print ("boosting almT\n")
    #if (almT.shape[0]!=1) : raise ValueError('almT.shape!=1')
    if (kernel.s !=0):
        kernel.s=0
        kernel.update()
    
    #delta_ell=kernel.delta_ell
    lmax=kernel.lmax
    
    #extention = (kernel.delta_ell*(2*kernel.lmax+1)+kernel.delta_ell**2)/2
    extention = kernel.delta_ell
    #extend the alm
    almT = np.append(almT,np.zeros(extention))
    
    Mmatrix = kernel.Mmatrix
    Lmatrix = kernel.Lmatrix
    
    #print almT.shape
    #print almT[mh.mlp2indx(Mmatrix,Lmatrix,lmax)].shape

    if nu:
        kernel.d = 3
        kernel.update()
        print ("\n boosting with nu={}\n\n".format(nu[0]))
        alm_boosted = np.sum(kernel.nu_mlpl(nu[0])*almT[mh.mlp2indx(Mmatrix,Lmatrix,lmax)],axis=1 )
    else:
        alm_boosted = np.sum(kernel.mlpl * almT[mh.mlp2indx(Mmatrix, Lmatrix, lmax)], axis=1)
    
    return alm_boosted


def _boost_almEB(almE,almB,kernel,*nu):
    '''boost polarization multipoles almE and almB (s=2)'''

    print ("boosting almEB\n")
    #if (almE.shape[0]!=1) : raise ValueError('almT.shape!=1')
    if (kernel.s !=2):
        kernel.s=2
        kernel.update()
    if nu:
        kernel_plus = kernel.nu_mlpl(nu[0])
    else:
        kernel_plus = kernel.mlpl
    
    kernel.s = -2
    kernel.update()

    if nu:
        kernel_minus= kernel.nu_mlpl(nu[0])
    else:
        kernel_minus = kernel.mlpl
    kernelEE_mlpl = 0.5 *(kernel_plus + kernel_minus)
    kernelEB_mlpl = 0.5j*(kernel_plus - kernel_minus)
    
    #delta_ell=kernel.delta_ell
    lmax=kernel.lmax
    
    #extention = (kernel.delta_ell*(2*kernel.lmax+1)+kernel.delta_ell**2)/2
    extention = kernel.delta_ell
    #extend the alm
    almE = np.append(almE,np.zeros(extention))
    almB = np.append(almB,np.zeros(extention))

    
    Mmatrix = kernel.Mmatrix
    Lmatrix = kernel.Lmatrix
    
    #sort and prepare alms for direct multiplication with kernel.mlpl
    alm_indx = mh.mlp2indx(Mmatrix,Lmatrix,lmax)
    
    almE_boosted = np.sum(kernelEE_mlpl*almE[alm_indx],axis=1 ) + np.sum(kernelEB_mlpl*almB[alm_indx],axis=1 )    
    almB_boosted = np.sum(kernelEE_mlpl*almB[alm_indx],axis=1 ) - np.sum(kernelEB_mlpl*almE[alm_indx],axis=1 )
    
    
    return np.vstack((almE_boosted,almB_boosted))


def boost_Cl(Cl,kernel):
    lmax = kernel.lmax
    delta_ell = kernel.delta_ell
    extention = (delta_ell*(2*lmax+1)+delta_ell**2)//2
    
    Cl_ext = np.append(Cl,np.zeros(extention))
    
    Lp = np.arange(lmax+1,dtype=int)
    L = np.tensordot(Lp,np.ones(2*delta_ell+1,dtype=int),axes=0) + np.arange(-delta_ell,delta_ell+1,dtype=int)
    
    Cl_boosted = np.sum(kernel.lpl()*Cl[L],axis=1)
    
    return Cl_boosted

