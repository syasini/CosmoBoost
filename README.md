# CosmoBoost

[![author](https://img.shields.io/badge/author-syasini-red)](https://github.com/syasini)
[![MIT license](http://img.shields.io/badge/license-MIT-blue.svg)](http://opensource.org/licenses/MIT)
[![stable](https://img.shields.io/badge/stable-v0.3-green)](https://github.com/syasini/CosmoBoost)
[![email](https://img.shields.io/badge/email-yasini%40usc.edu-lightgrey)](mailto:yasini@usc.edu)




CosmoBoost is a python package for Lorentz boosting anisotropic temperature and polarization maps in healpix format. The calculations are done in spherical harmonic space based on the *relativistic Doppler and aberration kernel formalism* developed in [Yasini & Pierpaoli (2017)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.103502) and [Dai & Chluba (2014)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.89.123504), following up on the original idea by [Challinor & van Leeuwen (2002](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.65.103001).


Currently the supported radiation types are:

- Cosmic Microwave Background (CMB)
- Kinetic Sunyaev Zeldovich (kSZ)
- Thermal Sunyaev Zeldovich (tSZ)

See the `tutorial.ipynb` notebook for an overview of the features through a set of examples.  

# Dependencies

- python 3 [![stable](https://img.shields.io/badge/tested%20on-v3.6-brightgreen)](https://www.python.org/downloads/release/python-360/)
- numpy [![stable](https://img.shields.io/badge/tested%20on-v1.16.4-brightgreen)](https://pypi.org/project/numpy/1.16.4/)
- scipy [![stable](https://img.shields.io/badge/tested%20on-v1.2.1-brightgreen)](https://pypi.org/project/scipy/1.2.1/)
- healpy[![stable](https://img.shields.io/badge/tested%20on-v1.12.9-brightgreen)](https://pypi.org/project/healpy/)(needed for running the tutorial)

# Installation

First, clone the repository by running 

`git clone https://github.com/syasini/CosmoBoost.git` 

then move to the CosmoBoost directory 

` cd CosmoBoost` 

and run 

`python setup.py install --user` 

or use pip

`pip install [-e] .`

the `-e` argument will install the package in editable mode which is suitable for developement. If you want to modify the code use this option. 


# Example Session

CosmoBoost has a simple and user friendly interface. Simply import the package using

`import cosmoboost as cb`

Then load the default boosting parameters dictionary (`beta = 0.00123`, `d=1`, `s=0`, `lmax= 1000`, etc.)

`pars = cb.DEFAULT_PARS`

Instantiate the kernel object

`kernel = cb.Kernel(pars)`

Now simply boost a set of alm's (recommended) using

`alm_boosted = cb.boost_alm(alm_rest, kernel)`

or boost the power spectrum Cl directly

`Cl_boosted = cb.boost_Cl(Cl_rest, kernel)`

See the tutorial for a comprehensive example. 

# Acknowledgement

If you find the contents of this repository useful for your research, please consider citing the following papers:
 ```
  @article{Yasini:2017jqg,
      author         = "Yasini, Siavash and Pierpaoli, Elena",
      title          = "{Generalized Doppler and aberration kernel for
                        frequency-dependent cosmological observables}",
      journal        = "Phys. Rev.",
      volume         = "D96",
      year           = "2017",
      number         = "10",
      pages          = "103502",
      doi            = "10.1103/PhysRevD.96.103502",
      eprint         = "1709.08298",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      SLACcitation   = "%%CITATION = ARXIV:1709.08298;%%"
  }
```

```
@article{Dai:2014swa,
      author         = "Dai, Liang and Chluba, Jens",
      title          = "{New operator approach to the CMB aberration kernels in
                        harmonic space}",
      journal        = "Phys. Rev.",
      volume         = "D89",
      year           = "2014",
      number         = "12",
      pages          = "123504",
      doi            = "10.1103/PhysRevD.89.123504",
      eprint         = "1403.6117",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      SLACcitation   = "%%CITATION = ARXIV:1403.6117;%%"
}
```
The bibtex entries are copied from `inspirehep.net`.

