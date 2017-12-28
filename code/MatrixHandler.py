
# coding: utf-8

# In[6]:

import os
COSMOBOOST_DIR = '/Users/siavashyasini/Dropbox/Cosmology/Projects/2018/cosmoboost'
sys.path.insert(0,COSMOBOOST_DIR+'/code')

import FileHandler as fh


# In[53]:

#######################################################
#      calculate the B,C,M,Lp,L,S martices
#######################################################

def Blm_Clm(delta_ell, lmax,s):
    '''calculate the Blm coefficients for the aberration kernel ODE
    uses #Eq. 23 in Dai, Chluba 2014 arXiv:1403.6117v2'''
    
    ell = np.arange(0,lmax+1+delta_ell) #arange the ell array from lmin to lmax
                                        #the Blm coefficients will be calcualted for these values
    
    L,M = np.meshgrid(ell,ell,indexing="ij")                             
                                 
    #initialize an zero matrix Blm of size (lmax,mmax=lmax) 
    #undefined values (for example ell < m will return 0.)
    Blm = np.zeros((lmax+1+delta_ell,lmax+1+delta_ell))
    Clm = np.zeros((lmax+1+delta_ell,lmax+1+delta_ell))
    
    #eq. 23 in Dai, Chluba 2014 arXiv:1403.6117v2
    Blm = np.sqrt((L**2-s**2)*np.true_divide(L**2-M**2,4.0*L**2-1))
    
    
    #set the NaN values from division by zero and sqrt(-1) to 0.
    Blm[np.isnan(Blm)]=0
    
    #L,_=np.meshgrid(np.arange(len(Blm)),np.ones(len(Blm)),indexing="ij")
    Clm = np.true_divide(Blm,L)
    
    #set the NaN and inf values from division by zero and sqrt(-1) to 0.
    Clm[np.isnan(Clm)]=0
    Clm[np.isinf(Clm)]=0
    return Blm,Clm



# In[54]:

def MLpL_matrix(delta_ell,lmax,lmin=0):
    '''finds the Mmatrix, Lpmatrix and Lmatrix:
    the Xmatrix returns the x index values for each element of the kernel matrix'''
    
    #height, width = ((lmax+1)*(lmax+2)/2,2*delta_ell+1)
    width = 2*delta_ell+1
    
    Mmatrix= np.array([])
    Lpmatrix = np.array([])
    
    
    Mmatrix = np.concatenate([m*np.ones(lmax+1-max(lmin,m)) for m in  xrange(0,lmax+1)])
    Lpmatrix = np.concatenate([np.arange(max(lmin,m),lmax+1) for m in  xrange(0,lmax+1)])
    
        
    Mmatrix = np.tensordot(Mmatrix,np.ones(width),axes=0)
    Lpmatrix = np.tensordot(Lpmatrix,np.ones(width),axes=0)    
    Lmatrix = Lpmatrix + np.arange(-delta_ell,delta_ell+1)

    return Mmatrix.astype(int),Lpmatrix.astype(int),Lmatrix.astype(int)


# In[ ]:

def S_matrix(Lmatrix,Mmatrix,s):
    '''finds the Smatrix:
    the Smatrix returns the s index values for each element of the kernel matrix for polarized observables'''
    
    Smatrix = s* np.true_divide(Mmatrix,Lmatrix*(Lmatrix+1))
    Smatrix[~np.isfinite(Smatrix)]=0
    return Smatrix




# In[ ]:

#######################################################
#         functions for shifting the rows 
#           and columns of the matrices
#######################################################



def shift_right(arr):
    arr = arr.T
    arr = np.roll(arr,1,axis=0)
    arr[0]=0
    arr = arr.T
    
    return arr 

def shift_left(arr):
    arr = arr.T
    arr = np.roll(arr,-1,axis=0)
    arr[-1]=0
    arr = arr.T
    
    return arr 

def shift_up(arr):
    #arr = arr.T
    arr = np.roll(arr,-1,axis=0)
    arr[-1]=0
    #arr = arr.T
    
    return arr


def shift_down(arr):
    #arr = arr.T
    arr = np.roll(arr,1,axis=0)
    arr[0]=0
    #arr = arr.T
    
    return arr


