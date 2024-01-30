'''
Define custom analyzers for HPVsim for GHlab analyses
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv


class econ_analyzer(hpv.Analyzer):
    '''
    Analyzer for feeding into costing/health economic analysis.

    Produces a dataframe by year storing:

        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    '''

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        return

    def initialize(self, sim):
        super().initialize(sim)
        columns = ['routine_vaccinations', 'plwh_vaccinations', 'new_cancers', 'new_cancer_deaths', 'new_other_deaths',
                   'av_age_cancers', 'av_age_cancer_deaths', 'av_age_other_deaths']
        self.si = sc.findinds(sim.res_yearvec, self.start)[0]
        self.df = pd.DataFrame(0.0, index=pd.Index(sim.res_yearvec[self.si:], name='year'), columns=columns)
        return

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start:
            ppl = sim.people

            def av_age(arr):
                if len(hpv.true(arr)):
                    return np.mean(sim.people.age[hpv.true(arr)])
                else:
                    return np.nan

            li = np.floor(sim.yearvec[sim.t])
            ltt = int((sim.t - 1) * sim[
                'dt'])  # this is the timestep number vs year of sim, needed to retrieve outcomes from interventions
            lt = (sim.t - 1)

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            pxv = simvals.plwh
            # Resources
            self.df.loc[li].routine_vaccinations += sim.get_intervention('Routine vx').n_products_used.values[ltt]
            self.df.loc[li].routine_vaccinations += sim.get_intervention('Catchup vx').n_products_used.values[ltt]

            if pxv:
                self.df.loc[li].plwh_vaccinations += sim.get_intervention('PxV for PLWH').n_products_used.values[
                        ltt]

            # Age outputs
            self.df.loc[li].av_age_other_deaths = av_age(ppl.date_dead_other == lt)
            self.df.loc[li].av_age_cancer_deaths = av_age(ppl.date_dead_cancer == lt)
            self.df.loc[li].av_age_cancers = av_age(ppl.date_cancerous == lt)
        return

    def finalize(self, sim):
        # Add in results that are already generated (NB, these have all been scaled already)
        self.df['new_cancers'] = sim.results['cancers'][self.si:]
        self.df['new_cancer_deaths'] = sim.results['cancer_deaths'][self.si:]
        self.df['new_other_deaths'] = sim.results['other_deaths'][self.si:]
        return


class prop_exposed(hpv.Analyzer):
    ''' Store proportion of agents exposed '''
    def __init__(self, years=None):
        super().__init__()
        self.years = years
        self.timepoints = []

    def initialize(self, sim):
        super().initialize(sim)
        for y in self.years:
            try:    tp = sc.findinds(sim.yearvec, y)[0]
            except: raise ValueError('Year not found')
            self.timepoints.append(tp)
        self.prop_exposed = dict()
        for gtype in range(sim['n_genotypes']):
            self.prop_exposed[gtype] = dict()
            for hiv_status in [True, False]:
                self.prop_exposed[gtype][hiv_status] = dict()
                for y in self.years: self.prop_exposed[gtype][hiv_status][y] = []

    def apply(self, sim):
        if sim.t in self.timepoints:
            tpi = self.timepoints.index(sim.t)
            year = self.years[tpi]
            for gtype in range(sim['n_genotypes']):
                for hiv_status in [True, False]:
                    prop_exposed = sc.autolist()
                    for a in range(10,30):
                        ainds = hpv.true((sim.people.age >= a) & (sim.people.age < a+1) & (sim.people.sex==0) & (sim.people.hiv==hiv_status))
                        prop_exposed += sc.safedivide(sum((~np.isnan(sim.people.date_exposed[gtype, ainds]))), len(ainds))
                    self.prop_exposed[gtype][hiv_status][year] = np.array(prop_exposed)
        return
    #
    # @staticmethod
    # def reduce(analyzers, quantiles=None):
    #     if quantiles is None: quantiles = {'low': 0.1, 'high': 0.9}
    #     base_az = analyzers[0]
    #     reduced_az = sc.dcp(base_az)
    #     reduced_az.prop_exposed = dict()
    #     for year in base_az.years:
    #         reduced_az.prop_exposed[year] = sc.objdict()
    #         allres = np.empty([len(analyzers), len(base_az.prop_exposed[year])])
    #         for ai,az in enumerate(analyzers):
    #             allres[ai,:] = az.prop_exposed[year][:]
    #         reduced_az.prop_exposed[year].best  = np.quantile(allres, 0.5, axis=0)
    #         reduced_az.prop_exposed[year].low   = np.quantile(allres, quantiles['low'], axis=0)
    #         reduced_az.prop_exposed[year].high  = np.quantile(allres, quantiles['high'], axis=0)
    #
    #     return reduced_az