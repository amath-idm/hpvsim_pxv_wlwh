"""
Define a custom intervention to adjust the HIV mortality rate
"""

import hpvsim as hpv
import sciris as sc
import numpy as np


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


class Vaccination(hpv.Intervention):

    def __init__(self, vx_coverage=None, plwh=False, rel_imm_lt200=1,
                 routine_start_year=2023, catch_up_year=2023, age_range=None, **kwargs):
        super().__init__(**kwargs)
        self.vx_coverage = vx_coverage
        self.plwh = plwh
        self.rel_imm_lt200 = rel_imm_lt200

        # Make interventions
        self.interventions = None
        self.make_interventions(age_range, routine_start_year, catch_up_year)

    def make_interventions(self, age_range, routine_start_year, catch_up_year):
        intvs = sc.autolist()

        # Deal with age
        if age_range is None: age_range = (9, 14)
        catchup_age = (age_range[0]+1, age_range[1])
        routine_age = (age_range[0], age_range[0]+1)

        # Add routine vaccination
        intvs += hpv.routine_vx(
                    prob=self.vx_coverage,
                    start_year=routine_start_year,
                    product='nonavalent',
                    age_range=routine_age,
                    label='Routine vx'
                )

        # Add campaign vaccination
        intvs += hpv.campaign_vx(
                    prob=self.vx_coverage,
                    years=catch_up_year,
                    product='nonavalent',
                    age_range=catchup_age,
                    label='Catchup vx'
                )

        # Optionally add PLHIV vaccination
        if self.plwh:
            uptake = 0.8
            px_eligible = lambda sim: (sim.people.hiv == True) & (sim.people.doses < 2)
            plwh_prod = hpv.default_vx(prod_name='nonavalent')
            plwh_prod.imm_init['par1'] *= self.rel_imm_lt200

            age_range = [16, 30]
            len_age_range = (age_range[1] - age_range[0]) / 2

            model_annual_vx_prob = 1 - (1 - uptake) ** (1 / len_age_range)

            paired_vx = hpv.routine_vx(
                prob=model_annual_vx_prob,
                start_year=routine_start_year,
                eligibility=px_eligible,
                age_range=age_range,
                product=plwh_prod,
                label='PxV for PLWH'
            )
            intvs += paired_vx

        self.interventions = intvs

        return

    def apply(self, sim):
        for intv in self.interventions:
            intv.apply(sim)
        return

    def finalize(self, sim=None):
        self.n_products_used = hpv.Result(name='Doses', npts=sim.res_npts, scale=True)
        for intv in self.interventions:
            self.n_products_used += intv.n_products_used


class ScreenTreat(hpv.Intervention):
    """
    Make screening and treatment interventions
    """

    def __init__(self, primary='hpv', screen_coverage=0.15, tx_coverage=0.9, start_year=2020):

        age_range = [30, 50]
        len_age_range = (age_range[1]-age_range[0])/2
        model_annual_screen_prob = 1 - (1 - screen_coverage)**(1/len_age_range)

        # Routine screening
        screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                      (sim.t > (sim.people.date_screened + 10 / sim['dt']))
        screening = hpv.routine_screening(
            product=primary,
            prob=model_annual_screen_prob,
            eligibility=screen_eligible,
            age_range=[30, 50],
            start_year=start_year,
            label='screening'
        )

        # Routine screening for WLWH
        screen_eligible = lambda sim: sim.people.art & np.isnan(sim.people.date_screened) | \
                                      (sim.t > (sim.people.date_screened + 3 / sim['dt']))
        screening_hiv = hpv.routine_screening(
            product=primary,
            prob=model_annual_screen_prob,
            eligibility=screen_eligible,
            age_range=[20, 50],
            start_year=start_year,
            label='screening PLWH'
        )


        # Assign treatment
        screen_positive = lambda sim: list(set(sim.get_intervention('screening').outcomes['positive'].tolist() +
                                               sim.get_intervention('screening PLWH').outcomes['positive'].tolist()))
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product='tx_assigner',
            eligibility=screen_positive,
            label='tx assigner'
        )

        ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
        ablation = hpv.treat_num(
            prob=tx_coverage,
            annual_prob=False,
            product='ablation',
            eligibility=ablation_eligible,
            label='ablation'
        )

        excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                                 sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
        excision = hpv.treat_num(
            prob=tx_coverage,
            annual_prob=False,
            product='excision',
            eligibility=excision_eligible,
            label='excision'
        )

        radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
        radiation = hpv.treat_num(
            prob=tx_coverage,
            annual_prob=False,
            product=hpv.radiation(),
            eligibility=radiation_eligible,
            label='radiation'
        )

        self.interventions = [screening, screening_hiv, triage_screening, ablation, excision, radiation]

        return

    def apply(self, sim):
        for intv in self.interventions:
            intv.apply(sim)
        return

