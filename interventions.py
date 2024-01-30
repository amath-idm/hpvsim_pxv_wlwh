'''
Define custom intervention
'''

import hpvsim as hpv
import numpy as np
import sciris as sc


class adjust_hiv_death(hpv.Intervention):
    def __init__(self, hiv_mort_adj=1, years=2010, **kwargs):
        super().__init__(**kwargs)
        self.hiv_mort_adj = sc.promotetolist(hiv_mort_adj)
        self.years = sc.promotetolist(years)
        self.label = 'hiv_mort_adj'

    def apply(self, sim):
        year = int(sim.yearvec[sim.t])
        if year in self.years:
            year_ind = sc.findinds(self.years, year)[0]
            hiv_mort = self.hiv_mort_adj[year_ind]
            sim['hiv_pars']['hiv_death_adj'] = hiv_mort
            sim.hivsim['hiv_pars']['hiv_death_adj'] = hiv_mort
        return