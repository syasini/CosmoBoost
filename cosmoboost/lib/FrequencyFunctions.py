"""
library containing frequency functions of the observed radiation
"""

__author__ = " Siavash Yasini"
__email__ = "yasini@usc.edu"

import numpy as np

############################################
#           frequency functions
############################################
      
# ------------------------------
#           black body
# ------------------------------

def B_nu(nu, T):
    """The BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz)"""
    return 0.0014745 * nu**3 /(np.exp(0.0479924*nu/(T))-1)


def F_nu(nu, T, normalized=False):
    """The differential BB spectrum (MJy/sr) with a temperature T (K) at frequency nu (GHz) """

    x = 0.0479924*nu/T
    f = x*np.exp(x)/(np.exp(x)-1)
    F = B_nu(nu, T)*f

    if normalized:
        return f
    else:
        return F

# ------------------------------
#        Sunyaev-Zeldovich
# ------------------------------

@np.vectorize
def F_tSZ(nu, T, normalized=False):
    """The frequency function of tSZ Intensity (tau*theta effect) at frequency nu (GHz)
    Default setting returns the frequency function normalized by the differential black body
    spectrum"""

    x = 0.0479924 * nu / T
    g = x / np.tanh(x / 2.) - 4.

    if normalized:
        return g
    else:
        F = F_nu(nu, T) * g
        return F


@np.vectorize
def F_kSZ(nu, T, normalized=False):
    """The frequency function of kSZ Intensity (tau*beta effect) at frequency nu (GHz)
    Default setting returns the frequency function normalized by the differential black body
    spectrum"""

    return F_nu(nu, T, normalized=normalized)
