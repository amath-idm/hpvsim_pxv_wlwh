'''
Define custom intervention
'''

import hpvsim as hpv

class adjust_hiv_death(hpv.Intervention):
    def __init__(self, hiv_mort_adj=0, **kwargs):
        super().__init__(**kwargs)
        self.hiv_mort_adj = hiv_mort_adj
        self.label = 'hiv_mort_adj'

    def apply(self, sim):
        year = int(sim.yearvec[sim.t])
        if year == 2010:
            sim['hiv_pars']['hiv_death_adj'] = self.hiv_mort_adj
            sim.hivsim['hiv_pars']['hiv_death_adj'] = self.hiv_mort_adj
        return