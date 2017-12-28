


import numpy as np
from astropy.io import fits
import os
from cosmoboost import DEFAULT_PARS
import warnings



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
    
    #setup all 6 HDUs with their respective keywords
    #*note that the keyword for the aberration kernel matrix is "primary"
    k_hdu = fits.PrimaryHDU()
    b_hdu = fits.ImageHDU(name="B")
    m_hdu = fits.ImageHDU(name="M")
    l_hdu = fits.ImageHDU(name="L")
    lp_hdu = fits.ImageHDU(name="Lp")
    s_hdu = fits.ImageHDU(name="S")
    
    hdus = [k_hdu,b_hdu,m_hdu,l_hdu,lp_hdu,s_hdu]
    
    #concatenate the HDUs into an HDUList and write to fits file
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(str(file_name),overwrite=True)


def save_matrix(file_name, matrix, key='k',overwrite=False):
    '''saves the matrix chosen by 'key' to the fits file
    initializes the fits file if it doesn't already exist'''
    
    #change the key from 'k' to 'primary for the aberration kernel matrix
    if (key=='k' or key=='K'): key='primary'
    
    #check to see if the file exists
    file_exists = os.path.isfile(str(file_name))
    if (not file_exists or overwrite==True):
        #initialize the fits file if it doesn't already exist
        warnings.warn("initializing fits file...")
        init_fits(file_name)
    
    #open the file in update mode and write the matrix in the appropriate HDU, then close it
    kernel_hdul = fits.open(str(file_name),mode='update')
    kernel_hdul[key].data = matrix 
    kernel_hdul.close()


def load_matrix(file_name, key='k'):
    '''loads the matrix chosen by 'key' from fits file'''
    
    #for the aberration kernel matrix change the key 'k' to primary
    if (key=='k' or key=='K'): key='primary'
    
    # if the file exists, open it and read the HDU chosen by 'key'
    try:
        kernel_hdul = fits.open(str(file_name),mode='readonly')
        matrix = kernel_hdul[key].data
        kernel_hdul.close()

        return matrix

    #raise error if the file does not exist
    except IOError:
        print str(file_name)+" does not exist."




