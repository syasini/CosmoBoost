import os

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.integrate import odeint
from timeit import default_timer as timer
from contextlib import contextmanager
import FileHandler as fh
import MatrixHandler as mh

N = 2

@contextmanager
def timeit():
    '''Time the code in mins'''
    t0 = timer()

    yield

    t1 = timer()

    t = t1-t0

    print("Done in {:.1f} min.".format(t/60))


def dK_deta(Kstore,eta,Bmatrix):
    '''The derivative of the Kernel for index m and lp'''
    #set index m to the predefined value M
    #m = M
    
    #L = np.arange(ellp-delta_ell,ellp+delta_ell+1)
    #L = L[L>=0]
    #L = L[L<=l_max-1]
    #print L
    #define the derivative according to the ODE (eq. 44 in Dai, Chluba 2014)
    
    #for l in L:
    #Kp_ellp[l] = + (0.0,Blm[l+1,m]*K_ellp[l+1])[l<l_max] - (0.0,Blm[l,m]*K_ellp[l-1])[l>0]
    #    if l == 0: Kp_ellp[0] = Blm[1,m]*K_ellp[1]
    #    elif l == l_max-1: Kp_ellp[l_max-1] = -Blm[l_max-1,m]*K_ellp[l_max-2]
    #    else: Kp_ellp[l] = + Blm[l+1,m]*K_ellp[l+1] - Blm[l,m]*K_ellp[l-1]
    K_return = mh.shift_left(Bmatrix*Kstore) - Bmatrix*mh.shift_right(Kstore)
    
    return K_return

def dK_deta_loop(K_ellp,eta,m,ellp):
    '''The derivative of the Kernel for index m and lp'''
    #set index m to the predefined value M
    #m = M
    
    #determine the length of the input kernel array K
    l_max = len(K_ellp)
    
    #initialize the output kernel array Kp with the same size as the input array K
    Kp_ellp = np.zeros(l_max)
    
    #pick the range \pm delta_ell around lp to loop over
    #the other elements of the kernel are essentially zero
    #so the loop goes only over the non-zero values to speed up the code
    
    
    L = np.arange(ellp-delta_ell,ellp+delta_ell+1)
    L = L[L>=0]
    L = L[L<=l_max-1]
    #print L
    #define the derivative according to the ODE (eq. 44 in Dai, Chluba 2014)
    
    for l in L:
        #Kp_ellp[l] = + (0.0,Blm[l+1,m]*K_ellp[l+1])[l<l_max] - (0.0,Blm[l,m]*K_ellp[l-1])[l>0]
        if l == 0: Kp_ellp[0] = Blm[1,m]*K_ellp[1]
        elif l == l_max-1: Kp_ellp[l_max-1] = -Blm[l_max-1,m]*K_ellp[l_max-2]
        else: Kp_ellp[l] = + Blm[l+1,m]*K_ellp[l+1] - Blm[l,m]*K_ellp[l-1]
    
    
    return Kp_ellp


def solve_K_T_ODE(pars,save_kernel=True,rtol=1.e-6,atol=1.e-6,mxstep=0):
    '''solves the ODE to find the temperature aberration kernel elements
    uses Eq. 44 in Dai, Chluba 2014 arXiv:1403.6117v2
    
    
    Parameters
    ----------
    s : int 
        The spin weight of the kernel elements
    delta_ell : int
        The neighborhood around ell' for ewhich the ODE will be solved 
    lmax : int 
        Maximum ell' for which the ODE will be solved
    save_kernel : bool, optional
        If True, the kernel elements will be saved to a file for later use
        
    Returns
    -------
    K_T : 2D numpy array
        Each row corresponds to the (m,ell') index calculated with the getindx
        scheme in the file_handler  . The rows correspond to different values of 
        ell for a neighborhood of delta_ell around ell'.
        
    Example
    -------
    ????
    '''
        
    with timeit():
        print ("calculating the Doppler and aberration Kernel elements...")

        beta = pars['beta']
        s = pars['s']
        delta_ell = pars['delta_ell']
        lmax = pars['lmax']
        lmin= pars['lmin']

        #set the height and width of the aberration kernel storage matrix.
        #the storage matrix is set around each value of ell' for a neighborhood of delta_ell
        #on each side. The middle value of each row corresponds to ell'=ell or delta_ell=0
        #the number of columns corresponds to different values of ell' for each m mode.
        height, width = ((lmax+1)*(lmax+2)//2,2*delta_ell+1)


        #initialize the K0=dirac_delta(ell,ell') (initial condition for the ODE)
        K0 = np.zeros(width)
        K0[delta_ell]=1
        K0 = np.tensordot(np.ones(height),K0,axes=0)


        #initialze K_T to store the temperature aberration kernel matrix
        #after solving the ODE
        #K_T = np.empty((height,width))
        #K_T.fill(np.nan)

        #evaluate the Blm coefficients
        #Blm = get_Blm(delta_ell=delta_ell,lmax=lmax,lmin=lmin,s=s)

        #fetch the Lmatrix and Mmatrix from file_handler.py
        #the elements of these matrices return the ell and m indices stored in that spot


        matrices_file_name = fh.matrices_filename(pars)
        if fh.file_exists(matrices_file_name):
            Mmatrix = fh.load_matrix(matrices_file_name,key='M')
            Lmatrix = fh.load_matrix(matrices_file_name,key='L')
        else:
            Mmatrix,Lmatrix = mh.ML_matrix(delta_ell=delta_ell,lmax=lmax)


        #construct the Bmatrix corresponding to the elements of K0


        Blms,_ = mh.Blm_Clm(delta_ell,lmax,s=s)
        #Clms = np.true_divide(Blms,1)
        Bmatrix = Blms[Lmatrix,Mmatrix]
        Bmatrix[np.isnan(Bmatrix)]=0


        #pad the K and B matrices to avoid leakage from the edges
        K0 = np.insert(K0,[2*delta_ell+1,2*delta_ell+1],0,axis=1)
        Bmatrix = np.insert(Bmatrix,[2*delta_ell+1,2*delta_ell+1],0,axis=1)

        #reshape all the 2D matrices to 1D arrays so that the ODE can be solved in vectorized mode
        K0 = K0.reshape((width+2)*height)
        #K_T = K_T.reshape((width+2)*height)
        #Bmatrix = np.ones((width,height))
        Bmatrix = Bmatrix.reshape((width+2)*height)




        # initialize the eta = np.arctanh(beta) array for ODE iterations
        # the index (N-1) will give the final result
        eta = np.linspace(0,np.arctanh(beta),N)

        print ("eta values : ", eta)




        #solve the ODE for a range of ell'  between lmin and lmax
        #the derivative of the aberration kernel with respect to eta is defined in Kernel_recursive.py
        #start_ode = timer()
        sol = odeint(dK_deta,K0,eta,args=(Bmatrix,),rtol=rtol,atol=atol,mxstep=mxstep,printmessg=True)

        #end_ode = timer()
        #print "ode: ", end_ode-start_ode
        #store the results in the K_T matrix
        K_T = sol[N-1].reshape(height,width+2)

        K_T = np.delete(K_T,[2*delta_ell+1,2*delta_ell+2],axis=1)


    if save_kernel:
        
        kernel_file_name = fh.kernel_filename(pars)
        #matrices_file_name = fh.matrices_filename(pars)
        dir_name = fh.dirname(lmax=lmax,beta=beta)
        if not os.path.exists(dir_name):
            "directory created."
            os.makedirs(dir_name)
        
        #save the kernel
        fh.save_kernel(kernel_file_name,K_T,'D1')
        
        #save the other matrices
        #fh.save_matrices(matrices_file_name,Mmatrix,'M')
        #fh.save_matrices(matrices_file_name,Lpmatrix,'LP')
        #fh.save_matrices(matrices_file_name,Lmatrix,'L')
        #fh.save_matrices(matrices_file_name,Clms,'CS'+str(s))
        #np.save(C_filename,Blms)
        

        print ("Kernel saved in: " + kernel_file_name)
        #print "Matrices saved in: " + kernel_file_name
    
    
    return K_T

