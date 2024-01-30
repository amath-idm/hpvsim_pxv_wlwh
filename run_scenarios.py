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
import analyzers as an

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
    i_r, i_pl, i_ri, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_r == sim.meta.inds[0]
        assert i_pl == sim.meta.inds[1]
        assert i_ri == sim.meta.inds[2]
        assert (s == 0) or i_s != sim.meta.inds[3]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_r, i_pl, i_ri]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims:  # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{ut.resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def run_scens(location=None, vx_coverage=None, plwh=None, rel_imm=None, hiv_death_adj=1, calib_filestem='', filestem='', # Input data
              debug=0, n_seeds=2, verbose=-1# Sim settings
              ):
    '''
    Run all screening/triage product scenarios for a given location
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(vx_coverage) * len(plwh) * len(rel_imm) * n_seeds

    for i_r, routine_cov in enumerate(vx_coverage):
        for i_pl, plwh_scen in enumerate(plwh):
            for i_ri, rel_imm_scen in enumerate(rel_imm):
                for i_s in range(n_seeds):  # n seeds
                    count += 1
                    meta = sc.objdict()
                    meta.count = count
                    meta.n_sims = n_sims
                    meta.inds = [i_r, i_pl, i_ri, i_s]
                    vx_scen_dict = dict(
                        vx_coverage=routine_cov,
                        plwh=plwh_scen,
                        rel_imm_lt200=rel_imm_scen,
                    )
                    meta.vals = sc.objdict(sc.mergedicts(vx_scen_dict, dict(seed=i_s, vx_coverage=routine_cov,
                                                                            plwh=plwh_scen, rel_imm=rel_imm_scen)))
                    ikw.append(sc.objdict(vx_intv=vx_scen_dict, seed=i_s))
                    ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    dflocation = location.replace(' ', '_')
    calib_pars = sc.loadobj(f'results/{dflocation}_pars{calib_filestem}.obj')
    kwargs = dict(calib_pars=calib_pars, verbose=verbose, debug=debug, location=location,
                  econ_analyzer=True, n_agents=50e3, hiv_death_adj=hiv_death_adj)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(vx_coverage), len(plwh), len(rel_imm), n_seeds), dtype=object)
    econdfs = sc.autolist()
    for sim in all_sims:  # Unflatten array
        i_r, i_pl, i_ri, i_s = sim.meta.inds
        sims[i_r, i_pl, i_ri, i_s] = sim
        if i_s == 0:
            econdf = sim.get_analyzer(an.econ_analyzer).df
            econdf['location'] = location
            econdf['seed'] = i_s
            econdf['vx_coverage'] = sim.meta.vals['vx_coverage']
            econdf['plwh'] = sim.meta.vals['plwh']
            econdf['rel_imm'] = sim.meta.vals['rel_imm']
            econdfs += econdf
        sim['analyzers'] = []  # Remove the analyzer so we don't need to reduce it
    econ_df = pd.concat(econdfs)
    sc.saveobj(f'{ut.resfolder}/{dflocation}_econ{calib_filestem}{filestem}.obj', econ_df)

    for sim in all_sims:  # Unflatten array
        i_r, i_pl, i_ri, i_s = sim.meta.inds
        sims[i_r, i_pl, i_ri, i_s] = sim

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_r in range(len(vx_coverage)):
        for i_pl in range(len(plwh)):
            for i_ri in range(len(rel_imm)):
                sim_seeds = sims[i_r, i_pl, i_ri, :].tolist()
                all_sims_for_multi.append(sim_seeds)
                

    # Convert sims to msims
    msims = np.empty((len(vx_coverage), len(plwh), len(rel_imm)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_r, i_pl, i_ri = msim.meta.inds
        msims[i_r, i_pl, i_ri] = msim
        df = pd.DataFrame()
        df['year']                      = msim.results['year']
        df['cancers']                   = msim.results['cancers'][:] # TODO: process in a loop
        df['cancers_low']               = msim.results['cancers'].low
        df['cancers_high']              = msim.results['cancers'].high
        df['cancers_with_hiv']          = msim.results['cancers_with_hiv'][:] # TODO: process in a loop
        df['cancers_with_hiv_low']      = msim.results['cancers_with_hiv'].low
        df['cancers_with_hiv_high']     = msim.results['cancers_with_hiv'].high
        df['female_hiv_prevalence']     = msim.results['female_hiv_prevalence'][:] # TODO: process in a loop
        df['female_hiv_prevalence_low'] = msim.results['female_hiv_prevalence'].low
        df['female_hiv_prevalence_high']= msim.results['female_hiv_prevalence'].high
        df['hiv_mortality']             = msim.results['hiv_mortality'][:]
        df['hiv_mortality_low']         = msim.results['hiv_mortality'].low
        df['hiv_mortality_high']        = msim.results['hiv_mortality'].high
        df['hiv_incidence']             = msim.results['hiv_incidence'][:] # TODO: process in a loop
        df['hiv_incidence_low']         = msim.results['hiv_incidence'].low
        df['hiv_incidence_high']        = msim.results['hiv_incidence'].high
        df['art_coverage']              = msim.results['art_coverage'][:]  # TODO: process in a loop
        df['art_coverage_low']          = msim.results['art_coverage'].low
        df['art_coverage_high']         = msim.results['art_coverage'].high
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
        df['cum_doses_low'] = msim.results['cum_doses'].low
        df['cum_doses_high'] = msim.results['cum_doses'].high
        df['location'] = location

        # Store metadata about run #TODO: fix this
        df['vx_coverage'] = msim.meta.vals['vx_coverage']
        df['plwh'] = msim.meta.vals['plwh']
        df['rel_imm'] = msim.meta.vals['rel_imm']
        dfs += df

    alldf = pd.concat(dfs)

    sc.saveobj(f'{ut.resfolder}/{dflocation}_results{calib_filestem}{filestem}.obj', alldf)

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

        for hiv_death_adj, label in zip([1, 1.5], ['_nomortredux_artcov', '_1.5xmortredux_artcov']):

            vx_coverage = [0]#, 0.4, 0.8]
            plwh = [False]#, True]
            rel_imm = [1]#, 0.75, 0.5]

            alldf, msims = run_scens(vx_coverage=vx_coverage, plwh=plwh, rel_imm=rel_imm, hiv_death_adj=hiv_death_adj, n_seeds=n_seeds, location=location,
                                     debug=debug, calib_filestem='_jan28', filestem=label)


    # Plot results of scenarios
    if 'plot_scenarios' in to_run:
        location = 'south africa'


        for sens in ['nomortredux_artcov', '1.5xmortredux_artcov']:

            # ut.plot_impact(
            #     location=location,
            #     routine_coverage=[0.4, 0.8],
            #     rel_imm=[1],#, 0.75, 0.5],
            #     filestem=f'_jan28_{sens}'
            # )
            #
            # ut.plot_ts(
            #     location=location,
            #     routine_coverage=[0.4, 0.8],
            #     plwh=[True, False],
            #     filestem=f'_jan28_{sens}'
            # )

            ut.plot_hiv_ts(
                location=location,
                routine_coverage=0,
                plwh=False,
                filestem=f'_jan28_{sens}'
            )


        ut.plot_hiv_ts_combined(
            location=location,
            routine_coverage=0,
            plwh=False,
            calib_filestem='_jan28',
            filestems=['nomortredux_artcov', '1.5xmortredux_artcov']#, '2xmortredux']
        )

        # ut.plot_impact_combined(
        #     location=location,
        #     routine_coverage=[0.4, 0.8],
        #     calib_filestem='_jan28',
        #     filestems=['nomortredux_artcov', '1.5xmortredux_artcov']#, '2xmortredux']
        # )


