#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:03:33 2017

@author: siavashyasini
"""
import numpy as np

############################################
#           frequency functions
############################################
       
#blackbody frequency spectrum
def B_nu(nu,T):
    '''calculate the BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz)'''
    return 0.0014745 * nu**3 /(np.exp(0.0479924*nu/(T))-1)

#differential blackbody spectrum 
def F_nu(nu,T):
    '''calculate the differential BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz) '''
    x = 0.0479924*nu/T
    f = x*np.exp(x)/(np.exp(x)-1)
    F = B_nu(nu,T)*f
    return F

#Doppler derivative of the differential blackbody spectrum (defined in arXiv 1610.00015)
def F11_nu(nu,T):
    '''calculate the differential BB spectrum (11) (MJy/sr) with a temperature T (K) at frequency nu (GHz) '''
    x = 0.0479924*nu/T
    f = x*np.exp(x)/(np.exp(x)-1)
    F = B_nu(nu,T)*f
    F11 = -3*F + nu*derivative(F_nu,nu,args=(T,))
    return F11


@np.vectorize
def F_tSZ(nu, T, normalized=False):
    '''calculate the frequency function of tSZ Intensity (tau*theta effect) at frequency nu (GHz)
    Default setting returns the frequency function normalized by the differential blackbody
    spectrum '''
    x = 0.0479924 * nu / T
    f = x / np.tanh(x / 2.) - 4.

    if normalized:
        return f
    else:
        F = F_nu(nu, T ) * f
        return F
