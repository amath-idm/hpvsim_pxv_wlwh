'''
Run HPVsim scenarios for each location. 

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp

# Comment out to not run
to_run = [
    'run_scenarios',
    # 'plot_scenarios',

]


debug = 0
n_seeds = [3, 1][debug] # How many seeds to use for stochasticity in projections

#%% Functions

def make_msims(sims, use_mean=True, save_msims=False):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_r, i_pl, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_r == sim.meta.inds[0]
        assert i_pl == sim.meta.inds[1]
        assert (s == 0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_r, i_pl]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims:  # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{ut.resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def run_scens(location=None, vx_coverage=None, plwh=None, # Input data
              debug=0, n_seeds=2, verbose=-1# Sim settings
              ):
    '''
    Run all screening/triage product scenarios for a given location
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(vx_coverage) * len(plwh) * n_seeds

    for i_r, routine_cov in enumerate(vx_coverage):
        for i_pl, plwh_scen in enumerate(plwh):
            for i_s in range(n_seeds):  # n seeds
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [i_r, i_pl, i_s]
                vx_scen_dict = dict(
                    vx_coverage=routine_cov,
                    plwh=plwh_scen
                )
                meta.vals = sc.objdict(sc.mergedicts(vx_scen_dict, dict(seed=i_s, vx_coverage=routine_cov,
                                                                        plwh=plwh_scen)))
                ikw.append(sc.objdict(vx_intv=vx_scen_dict, seed=i_s))
                ikw[-1].meta = meta                

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(verbose=verbose, debug=debug, location=location)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(vx_coverage), len(plwh), n_seeds), dtype=object)

    for sim in all_sims:  # Unflatten array
        i_r,i_pl, i_s = sim.meta.inds
        sims[i_r,i_pl, i_s] = sim

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_r in range(len(vx_coverage)):
        for i_pl in range(len(plwh)):
            sim_seeds = sims[i_r, i_pl, :].tolist()
            all_sims_for_multi.append(sim_seeds)
                

    # Convert sims to msims
    msims = np.empty((len(vx_coverage), len(plwh)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_r, i_pl = msim.meta.inds
        msims[i_r, i_pl] = msim
        df = pd.DataFrame()
        df['year']                      = msim.results['year']
        df['cancers']                   = msim.results['cancers'][:] # TODO: process in a loop
        df['cancers_low']               = msim.results['cancers'].low
        df['cancers_high']              = msim.results['cancers'].high
        df['cancer_incidence']          = msim.results['cancer_incidence'][:]
        df['cancer_incidence_high']     = msim.results['cancer_incidence'].high
        df['cancer_incidence_low']      = msim.results['cancer_incidence'].low
        df['asr_cancer_incidence']      = msim.results['asr_cancer_incidence'][:]
        df['asr_cancer_incidence_low']  = msim.results['asr_cancer_incidence'].low
        df['asr_cancer_incidence_high'] = msim.results['asr_cancer_incidence'].high
        df['cancer_deaths']             = msim.results['cancer_deaths'][:]
        df['cancer_deaths_low']         = msim.results['cancer_deaths'].low
        df['cancer_deaths_high']        = msim.results['cancer_deaths'].high
        df['n_vaccinated']              = msim.results['n_vaccinated'][:]
        df['n_vaccinated_low']          = msim.results['n_vaccinated'].low
        df['n_vaccinated_high']         = msim.results['n_vaccinated'].high
        # df['n_doses'] = msim.results['n_doses'][:]
        # df['n_doses_low'] = msim.results['n_doses_low'].low
        # df['n_doses_high'] = msim.results['n_doses_high'].high
        df['cum_doses'] = msim.results['cum_doses'][:]
        df['cum_doses_low'] = msim.results['cum_doses_low'].low
        df['cum_doses_high'] = msim.results['cum_doses_high'].high
        df['location'] = location

        # Store metadata about run #TODO: fix this
        df['vx_coverage'] = msim.meta.vals['vx_coverage']
        df['plwh'] = msim.meta.vals['plwh']
        dfs += df

    alldf = pd.concat(dfs)
    location = location.replace(' ', '_')
    sc.saveobj(f'{ut.resfolder}/{location}_results.obj', alldf)

    return alldf, msims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)


    if 'run_scenarios' in to_run:

        # Construct the scenarios
        location = 'south africa'

        vx_coverage = [0,0.4, 0.8, 1]
        plwh = [True, False]

        alldf, msims = run_scens(vx_coverage=vx_coverage, plwh=plwh, n_seeds=n_seeds, location=location, 
                                 debug=debug)


    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        location = 'south africa'
        # ut.plot_residual_burden(
        #     location=location,
        #     vx_scens=['No vaccine',
        #               'Vx, 50% cov, 9-10 routine, 10-18 catchup',
        #               'Vx, 50% cov, 9-10 routine, 15-18 catchup',
        #               'Vx, 70% cov, 9-10 routine, 10-18 catchup',
        #               'Vx, 70% cov, 9-10 routine, 15-18 catchup'],
        # )

        ut.plot_impact(
            location=location,
            routine_coverage=[0, 0.4, 0.8],#, 1],
            plwh = [True, False]
        )

        ut.plot_ts(
            location=location,
            routine_coverage=[0, 0.4, 0.8],#, 1],
            plwh=[True, False]
        )


