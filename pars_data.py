"""
Compilation of sexual behavior data and assumptions for three countries
with high HPV burden, for use in HPVsim.
"""


#%% Initialization

import numpy as np

use_dhs = True # Whether to use DHS data on age of debut directly
# use_dhs = True # Whether to use DHS data on age of debut directly

# Initialize objects with per-country results
debut                = dict()
layer_probs          = dict()
mixing               = dict()
dur_pship            = dict()
screening_coverage   = dict()
screening_start      = dict()
vaccination_coverage = dict()
vaccination_start    = dict()
condoms              = dict()



if use_dhs:
    debut['south africa'] = dict(
        f=dict(dist='normal', par1=17.7, par2=2.),# DHS 2013
        m=dict(dist='normal', par1=18.2, par2=2.)) # No data for males, assumption
else:
    debut['south africa'] = dict(
        f=dict(dist='normal', par1=14.8, par2=2.),# No data, assumption
        m=dict(dist='normal', par1=17.0, par2=2.)) # No data, assumption


dur_pship['south africa'] = dict(m=dict(dist='normal_pos', par1=15, par2=3),
                            c=dict(dist='normal_pos', par1=1, par2=1))

layer_probs['south africa'] = dict(
    m=np.array([
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        [0, 0, 0, 0.05, 0.25, 0.35, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.005, 0.001, 0.001, 0.001], # Share of females of each age who are married
        [0, 0, 0, 0.01, 0.25, 0.35, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.005, 0.001, 0.001, 0.001]] # Share of males of each age who are married
    ),
    c=np.array([
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        [0, 0, 0.10, 0.7, 0.8, 0.6, 0.6, 0.5, 0.2, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # Share of females of each age having casual relationships
        [0, 0, 0.05, 0.7, 0.8, 0.6, 0.6, 0.5, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01, 0.01, 0.01]], # Share of males of each age having casual relationships
    ),
)

mixing['south africa'] = dict(
    m=np.array([
        #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [15, 0, 0, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [20, 0, 0, .1, .1, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [25, 0, 0, .5, .1, .5, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 1, .5, .5, .5, .5, .1, 0, 0, 0, 0, 0, 0, 0, 0],
        [35, 0, 0, .5, 1, 1, .5, 1, 1, .5, 0, 0, 0, 0, 0, 0, 0],
        [40, 0, 0, 0, .5, 1, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0],
        [45, 0, 0, 0, 0, .1, 1, 1, 2, 1, 1, .5, 0, 0, 0, 0, 0],
        [50, 0, 0, 0, 0, 0, .1, 1, 1, 1, 1, 2, .5, 0, 0, 0, 0],
        [55, 0, 0, 0, 0, 0, 0, .1, 1, 1, 1, 1, 2, .5, 0, 0, 0],
        [60, 0, 0, 0, 0, 0, 0, 0, .1, .5, 1, 1, 1, 2, .5, 0, 0],
        [65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, .5, 0],
        [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, .5],
        [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ]),
    c=np.array([
        #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [15, 0, 0, 1, 1, 1, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [20, 0, 0, .5, 1, 1, 1, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [25, 0, 0, 0, 1, 1, 1, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 0, .5, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0, 0, 0],
        [35, 0, 0, 0, .5, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0, 0],
        [40, 0, 0, 0, 0, .5, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0],
        [45, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0],
        [50, 0, 0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1, .5, 0, 0, 0, 0],
        [55, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, .5, 0, 0, 0],
        [60, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, .5, 0, 0],
        [65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, .5, 0],
        [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, .5],
        [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ]),
)

m_partners = dict(
    m=dict(dist='poisson1', par1=0.01),
    c=dict(dist='poisson1', par1=0.2),
)
f_partners = dict(
    m=dict(dist='poisson1', par1=0.01),
    c=dict(dist='poisson1', par1=0.2),
)

# Intervention coverage
screening_start['south africa'] = '2019'
screening_coverage['south africa'] = 0
vaccination_start['south africa'] = 2018
vaccination_coverage['south africa'] = 0
condoms['south africa'] = 0.0375 # 2018 estimate