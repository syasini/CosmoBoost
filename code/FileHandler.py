
# coding: utf-8

# In[150]:

import numpy as np
from astropy.io import fits
import os
from cosmoboost import DEFAULT_PARS
import warnings


# In[249]:

def dirname(beta,lmax):
    '''returns the local directory address where the fits file should be saved'''
    return  "../data/beta_"+str(beta)[2:]+"/lmax_"+str(lmax) 

def filename(X,s,lmax,delta_ell,beta):
    '''returns the name of the fits file based on params'''
    return dirname(beta,lmax)+"/"+str(X)+"(d=1_s="+str(s)+")_lmax"+str(lmax)+"_delta"+str(delta_ell)+"_beta"+str(beta)[2:]+".fits"

def init_fits(file_name):
    '''initialize a fits file with 6 HDUs in the following order:
    PRIMARY (K): the aberration kernel matrix
    Blm : the Blm matrix, used in the ODE 
    M : Mmatrix used in doppler weight lifting of the kernel matrix
    L : Lmatrix used in doppler weight lifting of the kernel matrix
    Lp: Lpmatrix used in doppler weight lifting of the kernel matrix
    S: Smatrix used in doppler weight lifting of the kernel matrix with s!=0
    '''
    k_hdu = fits.PrimaryHDU()
    b_hdu = fits.ImageHDU(name="B")
    m_hdu = fits.ImageHDU(name="M")
    l_hdu = fits.ImageHDU(name="L")
    lp_hdu = fits.ImageHDU(name="Lp")
    s_hdu = fits.ImageHDU(name="S")
    
    hdus = [k_hdu,b_hdu,m_hdu,l_hdu,lp_hdu,s_hdu]
    
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(str(file_name),overwrite=True)
    
def save_matrix(file_name, matrix, key='k',overwrite=False):
    '''saves the chosen matrix to the fits file
    initializes the fits file if it doesn't already exist'''
    
    if key=='k': key='primary'
    
    file_exists = os.path.isfile(str(file_name))
    
    if (not file_exists or overwrite==True):  
        warnings.warn("initializing fits file...")
        init_fits(file_name)
    
    kernel_hdul = fits.open(str(file_name),mode='update')
    kernel_hdul[key].data = matrix 
    kernel_hdul.close()

