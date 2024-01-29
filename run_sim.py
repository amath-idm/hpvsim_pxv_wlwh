'''
Define the HPVsim simulations for South Africa that are used as
the basis for the calibration, scenarios, and sweeps.

By default, all locations are run. To not run a location, comment out the line
below. For all three locations, this script should take 1-5 minutes to run.
'''

# Additions to handle numpy multithreading
import os

import pandas as pd

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)
# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pandas as pd

# Imports from this repository
import pars_data as dp
import pars_scenarios as sp
import utils as ut
import analyzers as an
import matplotlib.pylab as pl



#%% Settings and filepaths

# Locations -- comment out a line to not run
locations = [
    'south africa'
]

# Debug switch
debug = 0 # Run with smaller population sizes and in serial
do_shrink = True # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


#%% Simulation creation functions
def make_sim(location=None, calib=False, debug=0, datafile=None, hiv_datafile=None, calib_pars=None, n_agents=10e3,
        art_datafile=None, vx_intv=None, econ_analyzer=False, analyzer=None, end=None, seed=1):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''
    if end is None:
        end = 2100
    if calib:
        end = 2020

    # Parameters
    pars = dict(
        n_agents       = [n_agents,1e3][debug],
        dt             = [0.25,1.0][debug],
        start          = [1950,1980][debug],
        end            = end,
        network        = 'default',
        location       = location,
        genotypes      = [16, 18, 'hi5', 'ohr'],
        f_partners     = dp.f_partners,
        m_partners     = dp.m_partners,
        debut          = dp.debut[location],
        mixing         = dp.mixing[location],
        layer_probs    = dp.layer_probs[location],
        init_hpv_dist  = dp.init_genotype_dist[location],
        init_hpv_prev  = {
            'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
            'm'             : np.array([ 0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f'             : np.array([ 0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        condoms        = dict(m=0.01, c=0.1),
        eff_condoms    = 0.5,
        ms_agent_ratio = 100,
        verbose        = 0.0,
        model_hiv      = True,
        hiv_pars       = dict(rel_imm=dict(lt200=1,gt200=1),
                              hiv_death_adj=4)
    )


    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)
    # Analyzers
    analyzers = sc.autolist()
    interventions = sc.autolist()
    if not calib:
        if len(vx_intv):
            interventions += sp.get_vx_intvs(**vx_intv)

        interventions += sp.get_screen_intvs(primary='hpv', screen_coverage=0.15, start_year=2020)

        if econ_analyzer:
            analyzers += an.econ_analyzer()

        if analyzer is not None:
            analyzers += analyzer

    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions,
                  datafile=datafile, hiv_datafile=hiv_datafile, art_datafile=art_datafile, rand_seed=seed)
    return sim




#%% Simulation running functions

def run_sim(location=None, vx_intv=None, n_agents=50e3, calib_pars=None, econ_analyzer=False,
            debug=0, seed=0, label=None, meta=None, verbose=0.1, end=None,
            do_save=False, die=False):
    ''' Assemble the parts into a complete sim and run it '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {location}'
    else:
        msg = f'Making sim for {location}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)
    dflocation = location.replace(' ', '_')
    # Make arguments
    if location == 'south africa':
        hiv_datafile = ['data/hiv_incidence_south_africa_sens.csv',
                        'data/south_africa_female_hiv_mortality.csv',
                        'data/south_africa_male_hiv_mortality.csv']
        art_datafile = ['data/south_africa_art_coverage_by_age_males.csv',
                        'data/south_africa_art_coverage_by_age_females.csv']
    else:
        hiv_datafile = None
        art_datafile = None
    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        vx_intv=vx_intv,
        n_agents=n_agents,
        calib_pars=calib_pars,
        econ_analyzer=econ_analyzer,
        end=end,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile,
    )
    sim['rand_seed'] = seed
    sim.label = f'{location}--{seed}'

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location # Store location in an easy-to-access place

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    if do_save:
        sim.save(f'{ut.resfolder}/{dflocation}.sim')

    return sim


def run_sims(locations=None, *args, **kwargs):
    ''' Run multiple simulations in parallel '''

    kwargs = sc.mergedicts(dict(debug=debug), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location:sim for location,sim in zip(locations, simlist)}) # Convert from a list to a dict

    return sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    location='south africa'
    calib_filestem='_jan28'
    dflocation = location.replace(' ', '_')
    calib_pars = sc.loadobj(f'results/{dflocation}_pars{calib_filestem}.obj')
    analyzer=an.prop_exposed(years=[2020])
    hiv_datafile = ['data/hiv_incidence_south_africa.csv',
                    'data/south_africa_female_hiv_mortality.csv',
                    'data/south_africa_male_hiv_mortality.csv']
    art_datafile = ['data/south_africa_art_coverage_by_age_males.csv',
                    'data/south_africa_art_coverage_by_age_females.csv']
    # Make sim


    sim = make_sim(location=location,
                   calib_pars=calib_pars,
                   analyzer=analyzer,
                   vx_intv=[],
                   hiv_datafile=hiv_datafile,
                   art_datafile=art_datafile,
                   end=2030,
                   n_agents=10e3
                   )

    sim.run()

    sim.plot(to_plot=['asr_cancer_incidence', 'cancers'])
    sim.plot(to_plot=['cancer_incidence_with_hiv', 'cancer_incidence_no_hiv'])
    sim.plot(to_plot=['hiv_prevalence', 'art_coverage'])

    # to_plot = {
    #     'HIV prevalence': [
    #         'hiv_prevalence',
    #         'female_hiv_prevalence',
    #         'male_hiv_prevalence'
    #     ],
    #     'HIV infections': [
    #         'hiv_infections'
    #     ],
    #     'Total pop': [
    #         'n_alive'
    #     ]
    #     # 'HPV prevalence by HIV status': [
    #     #     'hpv_prevalence_by_age_with_hiv',
    #     #     'hpv_prevalence_by_age_no_hiv'
    #     # ],
    #     # 'Age standardized cancer incidence (per 100,000 women)': [
    #     #     'asr_cancer_incidence',
    #     #     'cancer_incidence_with_hiv',
    #     #     'cancer_incidence_no_hiv',
    #     # ],
    #     # 'Cancers by age and HIV status': [
    #     #     'cancers_by_age_with_hiv',
    #     #     'cancers_by_age_no_hiv'
    #     # ]
    # }
    # sim.plot(to_plot=to_plot)


    T.toc('Done')

