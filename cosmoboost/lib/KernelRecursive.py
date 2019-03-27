"""
library including recursive equations for the Doppler and aberration kernel elements
"""

__author__ = " Siavash Yasini"
__email__ = "yasini@usc.edu"

import numpy as np
from . import FileHandler as fh
from . import FrequencyFunctions as ff
from . import MatrixHandler as mh
from scipy.misc import derivative, factorial, comb
from scipy.special import sph_harm as Ylm
from scipy.integrate import quad

sign_pref={0:"0",1:"+",-1:"-"}

def _K_d_lift(K,d,s):
    '''
    Lift the Doppler weight of the kernel by d>1 using the recursive formula in
    Yasini & Pierpaoli 2017 (http://arxiv.org/abs/1709.08298) Eq 15 & 16

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
        K_mlpl matrix with Doppler weight d

    '''

    assert(d>=1)

    if d == 1:
        return K._mlpl_d1

    elif (d>1):
        key = sign_pref[np.sign(s)]+"d"+str(d)
        print("key is {}".format(key))
        try:
            print("loading key {}".format(key))
            K_d = fh.load_matrix(K.kernel_filename,key=key)
        except:
            print("calculating key {}".format(key))
            K_d_minusone = _K_d_lift(K,d-1,s)
            
            K_l_plusone_d_minusone = mh.shift_left(K_d_minusone)
            C_l_plusone = mh.shift_left(K.Cmatrix)
            
            K_l_minusone_d_minusone = mh.shift_right(K_d_minusone)
            
            
            K_d= K.gamma*K_d_minusone + K.gamma*K.beta*(  C_l_plusone* K_l_plusone_d_minusone\
                                                      + np.sign(s)*K.Smatrix*K_d_minusone + K.Cmatrix*K_l_minusone_d_minusone)
                                                        
            fh.append_kernel(K.kernel_filename,K_d,key=key)

        return K_d


def K_d(K,d,s):
    '''Calculate the Kernel of Doppler weight d, from the Doppler weight 1, using the recursive
    expressions in Yasini & Pierpaoli 2017 (http://arxiv.org/abs/1709.08298) Eq 15 & 16 and
    Dai & Chluba 2014 (http://arxiv.org/abs/1403.6117) Eqs 8 & 9


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
        ndarray((lmax+1)*(lmax+2)/2,2*delta_ell+1): K_mlpl matrix with Doppler weight d'''

    if d>=1:
        K_d_mlpl = _K_d_lift(K,d,s)
        return K_d_mlpl
    
    elif d<1:
        K_d_mlpl = _K_d_lift(K,2-d,-s)
        print("d={} ".format(d))
        K_d_mllp = mh.transpose(K_d_mlpl,K.delta_ell)
    
        return mh.minus_one_LplusLp(K.delta_ell,K.lmax)*K_d_mllp


def d2indx(d,i):
    """convert the Doppler weight of the kernel to the index of the array"""
    return d-i


def calc_K_d_arr(K,d,s):
    '''
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
    '''
    height, width = ((K.lmax+1)*(K.lmax+2)//2,2*K.delta_ell+1)
    K_d_arr = np.zeros((K.beta_exp_order+1,height,width))
    for i in range(d,d-K.beta_exp_order-1,-1):
        print("d, i = {},{}".format(d,i))
        K_d_arr[d2indx(d,i)] = K_d(K,i,s)
    
    return K_d_arr

#FIXME: change derivative_dnu to dnu
def K_nu_d(K_d_arr,nu,pars,freq_func=ff.F_nu,dmod="sp",return_normalize=True):
    
    #d = pars['d']
    beta_exp_order = pars['beta_exp_order']
    T = pars['T_0']
    dx = pars['derivative_dnu']
    #N = betaorder
    
    #if (N > 3): print "the expansion order must be < 3"
    
    Kernel = 0.0
    for n in range(beta_exp_order+1):
        kfactor = 0.0
        for k in range(n+1):
            Klplm=K_d_arr[k]
            kfactor +=  Klplm *(-1.0)**(n+k) * comb(n,k)
        
        Kernel += kfactor/factorial(n)*nu**n * derivative(freq_func,nu,dx =dx,
                                                                   n=n,args=(T,),order=13) #
        # calculate the derivative with scipy
    
    if return_normalize:
        return np.true_divide(Kernel, freq_func(nu, pars['T_0']))
    #Kernel[abs(Kernel)<1.0E-12]=0
    else:
        return Kernel

#
# def K_nu_d_norm(K_d_arr,nu,pars,freq_func=ff.F_nu,dmod="sp"):
#
#     d = pars['d']
#     beta_exp_order = pars['beta_exp_order']
#     T = pars['T_0']
#     dx = pars['derivative_dnu']
#     #N = betaorder
#
#     #if (N > 3): print "the expansion order must be < 3"
#
#     Kernel = 0.0
#     for beta_order in xrange(beta_exp_order+1):
#         kfactor = 0.0
#         for k in xrange(beta_order+1):
#             Klplm=K_d_arr[k]
#             kfactor +=  Klplm *(-1.0)**(beta_order+k) * comb(beta_order,k)
#
#         Kernel += kfactor/factorial(beta_order)*nu**beta_order * derivative(freq_func,nu,dx =dx,n=beta_order,args=(T,),order=21) # calculate the derivative with scipy
#
#
#     #Kernel[abs(Kernel)<1.0E-12]=0
#     return np.true_divide(Kernel,ff.F_nu(nu,pars['T_0']))
#


#============test functions==========

def Kerd_intgrl_method(lp,l,m,d,beta,gamma):
    gamma = 1.0/np.sqrt(1-beta**2)
    func = lambda mup: np.conj(Ylm(m,lp,0,np.arccos(mup)))*\
        Ylm(m,l,0,np.arccos((mup-beta)/(1-beta*mup)))/(gamma*(1-beta*mup))**d
    return 2*np.pi*quad(func,-1,1)[0]



def Ker_nu_d_intgrl_method(nup,lp,l,m,d):
    beta = 0.001
    gamma = 1.0/np.sqrt(1-beta**2)
    func = lambda mup: np.conj(Ylm(m,lp,0,np.arccos(mup)))*\
        Ylm(m,l,0,np.arccos((mup-beta)/(1-beta*mup)))/(gamma*(1-beta*mup))**d * F_nu(gamma*(1-beta*mup)*nup,T_0)
    return 2*np.pi*quad(func,-1,1)[0]

