"""
library containing functions for handling (saving/loading) fits files
"""
__author__ = " Siavash Yasini"
__email__ = "yasini@usc.edu"

from astropy.io import fits
import os

from cosmoboost import COSMOBOOST_DIR

#######################################################
#             file and directory names
#######################################################

def dirname(beta, lmax):
    """returns the address and directory name where the fits file should be saved"""
    dirname = os.path.join(COSMOBOOST_DIR, "kernel", "beta_{}".format(beta),"lmax_{}".format(lmax))
    return dirname


def get_kernel_filename(pars):
    """returns the name and address of the fits file based on values in pars dictionary"""
    pars['s'] = abs(pars['s'])
    
    kernel_fname = os.path.join(
        dirname(pars['beta'], pars['lmax']),
        "K_T_s_{s}_delta_{delta_ell}_lmax_{lmax}_beta_{beta}.fits".format(**pars)
        )
    return kernel_fname
    

def get_matrices_filename(pars):
    """returns the name and address of the fits file based on params"""

    pars['s'] = abs(pars['s'])

    matrices_fname = os.path.join(
        dirname(pars['beta'], pars['lmax']),
        "K_T_s_{s}_delta_{delta_ell}_lmax_{lmax}_beta_{beta}.fits".format(**pars)
        )
    return matrices_fname
     

def file_exists(file_name):
    """check to see if the fits file exists in the given address"""
    
    return os.path.isfile(str(file_name))


#######################################################
#             fits file initialization
#######################################################

def init_kernel_fits_temp(kernel_file_name):
    """initialize a fits file with 5 HDUs in the following order:
        PRIMARY (T): the aberration kernel matrix for T
        E : the aberration kernel matrix for polarization E mode
        B : the aberration kernel matrix for polarization B mode
        SP2 : the aberration kernel matrix for polarization s = +2
        SM2 : the aberration kernel matrix for polarization s = -2
        """
    
    # setup all 6 HDUs with their respective keywords
    # NOTE: the keyword for the kernel matrix is "primary"
    T_hdu = fits.PrimaryHDU()
    T_hdu.name = "T"
    E_hdu = fits.ImageHDU(name="E")
    B_hdu = fits.ImageHDU(name="B")
    SP2_hdu = fits.ImageHDU(name="SP2")
    SM2_hdu = fits.ImageHDU(name="SM2")
    
    hdus = [T_hdu, E_hdu, B_hdu, SP2_hdu, SM2_hdu]
    
    # concatenate the HDUs into an HDUList and write to fits file
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(str(kernel_file_name), overwrite=True)

def init_kernel_fits(kernel_file_name):
    """initialize a fits file with 5 HDUs in the following order:
        PRIMARY (T): the aberration kernel matrix for T
        E : the aberration kernel matrix for polarization E mode
        B : the aberration kernel matrix for polarization B mode
        SP2 : the aberration kernel matrix for polarization s = +2
        SM2 : the aberration kernel matrix for polarization s = -2
        """

    # TODO: Clean this up
    # FIXME: check polarization kernel
    # setup all 6 HDUs with their respective keywords
    # NOTE: The keyword for the aberration kernel matrix is "primary"
    T_hdu = fits.PrimaryHDU()
    T_hdu.name = "D1"
    #E_hdu = fits.ImageHDU(name="E")
    #B_hdu = fits.ImageHDU(name="B")
    #SP2_hdu = fits.ImageHDU(name="SP2")
    #SM2_hdu = fits.ImageHDU(name="SM2")
    
    #hdus = [T_hdu,E_hdu,B_hdu,SP2_hdu,SM2_hdu]
    hdus = [T_hdu]
    #concatenate the HDUs into an HDUList and write to fits file
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(str(kernel_file_name), overwrite=True)

def init_matrices_fits(matrices_file_name):
    """initialize a fits file with 6 HDUs in the following order:
        PRIMARY (M): Mmatrix used in doppler weight lifting of the kernel matrix
        Lp: Lpmatrix used in Doppler weight lifting of the kernel matrix
        L : Lmatrix used in Doppler weight lifting of the kernel matrix
        CS0 : Clm matrix for s=0.
        CS2: Clm matrix for s=2.
        """
    # TODO: Clean this up
    # setup all 6 HDUs with their respective keywords
    # NOTE: The keyword for the aberration kernel matrix is "primary"
    M_hdu = fits.PrimaryHDU()
    M_hdu.name = "M"
    #LP_hdu = fits.ImageHDU(name="LP")
    L_hdu = fits.ImageHDU(name="L")
    #CS0_hdu = fits.ImageHDU(name="CS0")
    #CS2_hdu = fits.ImageHDU(name="CS2")
    #S_hdu = fits.ImageHDU(name="S")
    
    #hdus = [M_hdu,LP_hdu,L_hdu,CS0_hdu,CS2_hdu,S_hdu]
    hdus = [M_hdu,L_hdu]
    
    #concatenate the HDUs into an HDUList and write to fits file
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(str(matrices_file_name),overwrite=True)

#######################################################
#              file saving / loading
#######################################################

def save_kernel(kernel_file_name, kernel, key='D1', overwrite=False):
    """saves the kernel chosen by 'key' to the fits file
        initializes the fits file if it doesn't already exist"""

    # check to see if the file exists
    file_exists = os.path.isfile(str(kernel_file_name))
    if not file_exists or overwrite is True:
        # initialize the fits file if it doesn't already exist
        print("initializing fits file for the kernel...\n")
        init_kernel_fits(kernel_file_name)
    
    # open the file in update mode and write the kernel in the appropriate HDU, then close it
    kernel_hdul = fits.open(str(kernel_file_name), mode='update')
    kernel_hdul[key].data = kernel
    kernel_hdul.close()


def save_matrices(matrices_file_name, matrix, key='M',overwrite=False):
    """saves the matrix chosen by 'key' to the fits file
        initializes the fits file if it doesn't already exist"""

    # check to see if the file exists
    file_exists = os.path.isfile(str(matrices_file_name))
    if not file_exists or overwrite is True:
        # initialize the fits file if it doesn't already exist
        print("initializing fits file for the matrices...\n")
        init_matrices_fits(matrices_file_name)
    
    # open the file in update mode and write the matrix in the appropriate HDU, then close it
    kernel_hdul = fits.open(str(matrices_file_name), mode='update')
    kernel_hdul[key].data = matrix
    kernel_hdul.close()


def append_kernel(kernel_file_name, kernel, key):
    kernel_hdul = fits.open(str(kernel_file_name), mode='append')
    
    hdu_d = fits.ImageHDU(name=key)
    kernel_hdul.append(hdu_d)
    kernel_hdul[key].data = kernel
    kernel_hdul.close()
        

def load_kernel(kernel_file_name, key='D1'):
    """loads the matrix chosen by 'key' from fits file"""

    # if the file exists, open it and read the HDU chosen by 'key'
    try:
        kernel_hdul = fits.open(str(kernel_file_name), mode='readonly')
        matrix = kernel_hdul[key].data
        kernel_hdul.close()
        
        return matrix
    
    # raise error if the file does not exist
    except IOError:
        print("File does not exist:\n{}".format(kernel_file_name))


def load_matrix(matrices_file_name, key='M'):
    """loads the matrix chosen by 'key' from fits file"""

    # if the file exists, open it and read the HDU chosen by 'key'
    try:
        kernel_hdul = fits.open(str(matrices_file_name), mode='readonly')
        matrix = kernel_hdul[key].data
        kernel_hdul.close()
        
        return matrix
    
    # raise error if the file does not exist
    except IOError:
        print("File does not exist:\n{}".format(matrices_file_name))




