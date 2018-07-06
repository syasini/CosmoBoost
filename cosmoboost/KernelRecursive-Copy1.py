#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:03:33 2017

@author: siavashyasini
"""

import constants
import numpy as np
import FileHandler as fh
import MatrixHandler as mh
from scipy.misc import derivative, factorial, comb
from scipy.special import sph_harm as Ylm
from scipy.integrate import quad 



def K_d(K,Cl,d,delta_ell,lmax,beta):
    '''calculate the Kernel of doppler weight d, from the doppler weight 1, using the recursive expressions '''
    gamma = 1.0/np.sqrt(1-beta**2)
    
    
    if (d==1):
        #L, K_lpm = find_kernel_lpm(K,lp,m,lmax=lmax)
        return K
    
    elif (d>1):
        K_d_minusone = K_d(K,Cl,d-1,delta_ell,lmax,beta)
        
        K_l_plusone_d_minusone = shift_left(K_d_minusone)
        C_l_plusone = shift_left(Cl)
        
        K_l_minusone_d_minusone = shift_right(K_d_minusone)
        
        
        return gamma*K_d_minusone + gamma*beta*(  C_l_plusone* K_l_plusone_d_minusone\
                        + Cl*K_l_minusone_d_minusone  )
    elif (d<1):
        K_d_plusone = K_d(K,C,d+1)
        
        K_lp_plusone_d_plusone = shift_right(shift_up(K_d_plusone))
        C_lp_plusone = shift_right(shift_up(C))
        
        K_lp_minusone_d_plusone = shift_left(shift_down(K_d_plusone))
        
        
        return gamma*K_d_plusone - gamma*beta*(  C_lp_plusone* K_lp_plusone_d_plusone\
                        + C*K_lp_minusone_d_plusone  )
 

def K_ds(K,C,SLmatrix,SLpmatrix, d,s,lmax):
    '''calculate the Kernel of doppler weight d, from the doppler weight 1, using the recursive expressions '''
    
    
    
    if (d==1):
        #L, K_lpm = find_kernel_lpm(K,lp,m,lmax=lmax)
        return K
    
    elif (d>1): 
        K_d_minusone = K_ds(K,C,SLmatrix,SLpmatrix,d-1,s)
        
        K_l_plusone_d_minusone = shift_left(K_d_minusone)
        C_l_plusone = shift_left(C)
        
        K_l_minusone_d_minusone = shift_right(K_d_minusone)
        
        
        return gamma*K_d_minusone + gamma*beta*(  C_l_plusone* K_l_plusone_d_minusone\
                    + SLmatrix*K_d_minusone  + C*K_l_minusone_d_minusone  )
    elif (d<1):
        K_d_plusone = K_ds(K,C,SLmatrix,SLpmatrix,d+1,s)
        
        K_lp_plusone_d_plusone = shift_right(shift_up(K_d_plusone))
        C_lp_plusone = shift_right(shift_up(C))
        
        K_lp_minusone_d_plusone = shift_left(shift_down(K_d_plusone))
        
        
        return gamma*K_d_plusone - gamma*beta*(  C_lp_plusone* K_lp_plusone_d_plusone\
                     + SLpmatrix*K_d_plusone     + C*K_lp_minusone_d_plusone  )
 
def get_K_ds(K_T,C_T,SLmatrix,SLpmatrix,d,s,beta_exp_order,lmax):
    height, width = K_T.shape
    Kd = np.zeros((beta_exp_order+1,height,width))
    if s == 0 :
        for i in range(d,d-beta_exp_order-1,-1):
            Kd[d2indx(i)] = K_d(K_T,C_T,i,lmax=lmax)
    if s!= 0 :
        for i in range(d,d-beta_exp_order-1,-1):
            Kd[d2indx(i)] = K_ds(K_T,C_T,SLmatrix,SLpmatrix, i,lmax=lmax,s=s)
    return Kd
       
#==========frequency functions        
def d2indx(d):
    return 3-d
       
        #define the blackbody frequencies
def B_nu(nu,T=T_0):
    '''calculate the BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz)'''
    return 0.0014745 * nu**3 /(np.exp(0.0479924*nu/(T))-1)

#differential blackbody spectrum 
def F_nu(nu,T=T_0):
    '''calculate the differential BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz) '''
    x = 0.0479924*nu/T
    f = x*np.exp(x)/(np.exp(x)-1)
    F = B_nu(nu,T)*f
    return F

def F11_nu(nu,T=T_0):
    '''calculate the differential BB spectrum (11) (MJy/sr) with a temperature T (K) at frequency nu (GHz) '''
    x = 0.0479924*nu/T
    f = x*np.exp(x)/(np.exp(x)-1)
    F = B_nu(nu,T)*f
    F11 = -3*F + nu*derivative(F_nu,nu,args=(T,))
    return F11

def K_nu_d(Kd,nu,d,beta_exp_order,T=T_0,dmod="sp",dx=1.0):
    #N = betaorder
    
    #if (N > 3): print "the expansion order must be < 3"
    
    Kernel = 0.0
    for beta_order in xrange(beta_exp_order+1):
        kfactor = 0.0
        for k in xrange(beta_order+1):
            Klplm=Kd[d2indx(d-k)]
            kfactor +=  Klplm *(-1.0)**(beta_order+k) * comb(beta_order,k)
            
        Kernel += kfactor/factorial(beta_order)*nu**beta_order * derivative(F_nu,nu,dx =dx,n=beta_order,args=(T,),order=21) # calculate the derivative with scipy
        
            
    #Kernel[abs(Kernel)<1.0E-12]=0             
    return Kernel


def K_nu_d_normalized(Kd,nu,d,beta_exp_order,T=T_0,dmod="sp",dx=1.0):
    #N = betaorder
    
    #if (N > 3): print "the expansion order must be < 3"
    
    Kernel = 0.0
    for beta_order in xrange(beta_exp_order+1):
        kfactor = 0.0
        for k in xrange(beta_order+1):
            Klplm=Kd[d2indx(d-k)]
            kfactor +=  Klplm *(-1.0)**(beta_order+k) * comb(beta_order,k)
            
        Kernel += kfactor/factorial(beta_order)*nu**beta_order * derivative(F_nu,nu,dx =dx,n=beta_order,args=(T,),order=21) # calculate the derivative with scipy
        
            
    #Kernel[abs(Kernel)<1.0E-12]=0             
    return np.true_divide(Kernel,F_nu(nu,T_0))

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
