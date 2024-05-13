'''
Run HPVsim scenarios for each location. 

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)


# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
from interventions import Vaccination

# Comment out to not run
debug = 0
n_seeds = [3, 1][debug]  # How many seeds to use for stochasticity in projections


# %% Functions
def make_vx_scenarios(coverage_arr=None, rel_imm_arr=None):

    vx_scenarios = dict()

    # Baseline
    vx_scenarios['Baseline'] = []

    # Construct the scenarios
    for cov_val in coverage_arr:
        label = f'Coverage: {cov_val}'
        vx_scenarios[label] = Vaccination(vx_coverage=cov_val, plwh=False)

        # Construct the PLHIV scenarios
        if rel_imm_arr is not None:
            for imm_val in rel_imm_arr:
                label = f'Coverage: {cov_val} with PLHIV ({imm_val})'
                vx_scenarios[label] = Vaccination(vx_coverage=cov_val, plwh=True, rel_imm_lt200=imm_val)
        else:
            label = f'Coverage: {cov_val} with PLHIV'
            vx_scenarios[label] = Vaccination(vx_coverage=cov_val, plwh=True)

    return vx_scenarios


def make_hiv_scenarios():
    hiv_scens = dict()
    hiv_scens['Baseline HIV'] = dict(hiv_death_adj=1, hiv_inc_filename='hiv_incidence_south_africa_sens')
    hiv_scens['Lower mortality'] = dict(hiv_death_adj=1.5, hiv_inc_filename='hiv_incidence_south_africa_sens')
    hiv_scens['Higher incidene'] = dict(hiv_death_adj=1, hiv_inc_filename='hiv_incidence_south_africa_sens_2')
    return hiv_scens


def make_sims(calib_pars=None, vx_scenarios=None, hiv_scens=None):
    """ Set up scenarios """

    all_msims = sc.autolist()
    for vx_name, vx_intv in vx_scenarios.items():
        for hiv_name, hiv_scen in hiv_scens.items():
            sims = sc.autolist()
            for seed in range(n_seeds):
                sim = rs.make_sim(calib_pars=calib_pars, debug=debug, vx_intv=vx_intv, end=2100, seed=seed, **hiv_scen)
                sim.label = f'{hiv_name}, {vx_name}'
                sims += sim
            all_msims += hpv.MultiSim(sims)

    msim = hpv.MultiSim.merge(all_msims, base=False)

    return msim


def run_sims(calib_pars=None, vx_scenarios=None, hiv_scens=None, verbose=0.2):
    """ Run the simulations """
    msim = make_sims(calib_pars=calib_pars, vx_scenarios=vx_scenarios, hiv_scens=hiv_scens)
    msim.run(verbose=verbose)
    return msim


# %% Run as a script
if __name__ == '__main__':

    do_run = True
    do_save = True
    do_process = False

    coverage_arr = [0.2, 0.4, 0.8]
    rel_imm_arr = None  #[1]
    calib_pars = sc.loadobj(f'results/south_africa_pars_feb21_artsens.obj')

    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    if do_run:

        vx_scenarios = make_vx_scenarios(coverage_arr=coverage_arr, rel_imm_arr=rel_imm_arr)
        hiv_scens = make_hiv_scenarios()
        msim = run_sims(calib_pars=calib_pars, vx_scenarios=vx_scenarios, hiv_scens=hiv_scens)

        if do_save: msim.save('results/vs.msim')


    # # Plot results of scenarios
    # if 'plot_scenarios' in to_run:
    #     location = 'south africa'
    #
    #
    #     for sens in ['', '_1.5xmortredux', '_incredux']:
    #
    #         ut.plot_impact(
    #             location=location,
    #             routine_coverage=[0.2, 0.4, 0.8],
    #             rel_imm=[1],#, 0.75, 0.5],
    #             filestem=f'_jan28{sens}'
    #         )
    #
    #         ut.plot_ts(
    #             location=location,
    #             routine_coverage=[0.2, 0.4, 0.8],
    #             plwh=[True, False],
    #             filestem=f'_jan28{sens}'
    #         )
    #
    #         ut.plot_hiv_ts(
    #             location=location,
    #             routine_coverage=0,
    #             plwh=False,
    #             filestem=f'_jan28{sens}'
    #         )
    #
    #
    #     ut.plot_hiv_ts_combined(
    #         location=location,
    #         routine_coverage=0,
    #         plwh=False,
    #         calib_filestem='_jan28',
    #         filestems=['_1.5xmortredux', '',  '_incredux']#v2']
    #     )
    #
    #     ut.plot_impact_combined(
    #         location=location,
    #         routine_coverage=[0.2, 0.4, 0.8],
    #         calib_filestem='_jan28',
    #         filestems=['_1.5xmortredux', '', '_incredux']#v2']
    #     )
    #

