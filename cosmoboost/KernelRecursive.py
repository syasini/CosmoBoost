#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Sun Jul  2 17:03:33 2017
    
    @author: siavashyasini
    """


import numpy as np
import FileHandler as fh
import FrequencyFunctions as ff
import MatrixHandler as mh
from scipy.misc import derivative, factorial, comb
from scipy.special import sph_harm as Ylm
from scipy.integrate import quad

sign_pref={0:"0",1:"+",-1:"-"}

def _K_d_lift(K,d,s):
    
    if d == 1:
        return K._mlpl_d1

    elif (d>1):
        key = sign_pref[np.sign(s)]+"d"+str(d)
        print key
        try:
            K_d = fh.load_matrix(K.kernel_filename,key=key)
        except:
            K_d_minusone = _K_d_lift(K,d-1,s)
            
            K_l_plusone_d_minusone = mh.shift_left(K_d_minusone)
            C_l_plusone = mh.shift_left(K.Cmatrix)
            
            K_l_minusone_d_minusone = mh.shift_right(K_d_minusone)
            
            
            K_d= K.gamma*K_d_minusone + K.gamma*K.beta*(  C_l_plusone* K_l_plusone_d_minusone\
                                                      + np.sign(s)*K.Smatrix*K_d_minusone + K.Cmatrix*K_l_minusone_d_minusone)
                                                        
            fh.append_kernel(K.kernel_filename,K_d,key=key)

        return K_d


def K_d(K,d,s):
    '''calculate the Kernel of doppler weight d, from the doppler weight 1, using the recursive expressions '''
    if d>=1:
        K_d_mlpl = _K_d_lift(K,d,s)
        return K_d_mlpl
    
    elif d<1:
        K_d_mlpl = _K_d_lift(K,2-d,-s)
        print d
        K_d_mllp = mh.transpose(K_d_mlpl,K.delta_ell)
    
    return mh.minus_one_LplusLp(K.delta_ell,K.lmax)*K_d_mlpl


def d2indx(d,i):
    return d-i


def get_K_d_arr(K,d,s):
    height, width = ((K.lmax+1)*(K.lmax+2)/2,2*K.delta_ell+1)
    K_d_arr = np.zeros((K.beta_exp_order+1,height,width))
    for i in range(d,d-K.beta_exp_order-1,-1):
        K_d_arr[d2indx(d,i)] = K_d(K,i,s)
    
    return K_d_arr


def K_nu_d(K_d_arr,nu,pars,freq_func=ff.F_nu,dmod="sp"):
    
    d = pars['d']
    beta_exp_order = pars['beta_exp_order']
    T = pars['T_0']
    dx = pars['derivative_dnu']
    #N = betaorder
    
    #if (N > 3): print "the expansion order must be < 3"
    
    Kernel = 0.0
    for beta_order in xrange(beta_exp_order+1):
        kfactor = 0.0
        for k in xrange(beta_order+1):
            Klplm=K_d_arr[k]
            kfactor +=  Klplm *(-1.0)**(beta_order+k) * comb(beta_order,k)
        
        Kernel += kfactor/factorial(beta_order)*nu**beta_order * derivative(freq_func,nu,dx =dx,n=beta_order,args=(T,),order=21) # calculate the derivative with scipy
    
    
    #Kernel[abs(Kernel)<1.0E-12]=0
    return Kernel


def K_nu_d_norm(K_d_arr,nu,pars,freq_func=ff.F_nu,dmod="sp"):
    
    d = pars['d']
    beta_exp_order = pars['beta_exp_order']
    T = pars['T_0']
    dx = pars['derivative_dnu']
    #N = betaorder
    
    #if (N > 3): print "the expansion order must be < 3"
    
    Kernel = 0.0
    for beta_order in xrange(beta_exp_order+1):
        kfactor = 0.0
        for k in xrange(beta_order+1):
            Klplm=K_d_arr[k]
            kfactor +=  Klplm *(-1.0)**(beta_order+k) * comb(beta_order,k)
        
        Kernel += kfactor/factorial(beta_order)*nu**beta_order * derivative(freq_func,nu,dx =dx,n=beta_order,args=(T,),order=21) # calculate the derivative with scipy
    
    
    #Kernel[abs(Kernel)<1.0E-12]=0
    return np.true_divide(Kernel,ff.F_nu(nu,pars['T_0']))



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

