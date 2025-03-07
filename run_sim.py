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
import seaborn as sns

# Imports from this repository
import pars_data as dp
import pars_scenarios as sp
from interventions import adjust_hiv_death
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
             art_datafile=None, vx_intv=None, hiv_death_adj=1, econ_analyzer=False, analyzer=None, end=None, seed=1,
             art_sens=False):
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
        hiv_pars       = dict(rel_imm=dict(lt200=1,gt200=1))
    )


    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    pars['hiv_pars']['art_failure_prob'] = 0.1
    if art_sens:
        pars['hiv_pars']['cd4_lb'] = [0, 5e3] # Lower bound for CD4 states
        pars['hiv_pars']['cd4_ub'] = [5e6, 5e6] # Upper bound for CD4 states
    # Analyzers
    analyzers = sc.autolist()
    interventions = sc.autolist()
    # interventions += adjust_hiv_death(hiv_death_adj)

    interventions += adjust_hiv_death(years=[1985,2000,2010], hiv_mort_adj=[1,1,1.5*hiv_death_adj])

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

def run_sim(location=None, vx_intv=None, n_agents=50e3, hiv_death_adj=1, calib_pars=None, econ_analyzer=False,
            debug=0, seed=0, label=None, meta=None, verbose=0.1, end=None, hivinc_datafile=None,
            do_save=False, die=False, art_sens=False):
    ''' Assemble the parts into a complete sim and run it '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {location}'
    else:
        msg = f'Making sim for {location}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)
    dflocation = location.replace(' ', '_')

    if hivinc_datafile is None:
        hivinc_datafile = 'hiv_incidence_south_africa'
    # Make arguments
    if location == 'south africa':
        hiv_datafile = [f'data/{hivinc_datafile}.csv',
                        'data/south_africa_female_hiv_mortality.csv',
                        'data/south_africa_male_hiv_mortality.csv']
        # art_datafile = ['data/south_africa_art_coverage_by_age_males.csv',
        #                 'data/south_africa_art_coverage_by_age_females.csv']
        art_datafile = ['data/art_coverage_south_africa.csv']
    else:
        hiv_datafile = None
        art_datafile = None
    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        vx_intv=vx_intv,
        n_agents=n_agents,
        hiv_death_adj=hiv_death_adj,
        calib_pars=calib_pars,
        econ_analyzer=econ_analyzer,
        end=end,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile,
        art_sens=art_sens
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
    hiv_datafile = ['data/hiv_incidence_south_africa_sens.csv',
                    'data/south_africa_female_hiv_mortality.csv',
                    'data/south_africa_male_hiv_mortality.csv']
    # art_datafile = ['data/south_africa_art_coverage_by_age_males.csv',
    #                 'data/south_africa_art_coverage_by_age_females.csv']
    art_datafile = ['data/art_coverage_south_africa.csv']
    # Make sim


    sim = make_sim(location=location,
                   calib_pars=calib_pars,
                   analyzer=analyzer,
                   vx_intv=[],
                   hiv_datafile=hiv_datafile,
                   art_datafile=art_datafile,
                   end=2020,
                   n_agents=50e3,
                   hiv_death_adj=1
                   )

    sim.run()

    # New infections by age and sex
    simres = sc.dcp(sim.results)
    years = simres['year']
    year_ind = sc.findinds(years, 1985)[0]
    rsa_df = pd.read_csv('data/RSA_data.csv').set_index('Unnamed: 0').T
    title_dict = dict(
        female_hiv_prevalence='HIV prevalence (%), females 15+',
        hiv_incidence='HIV incidence',
        art_coverage='ART coverage (%)',
    )
    years = years[year_ind:]
    fig, axes = pl.subplots(1, 3, figsize=(10, 4))
    to_plot = ['female_hiv_prevalence', 'art_coverage', ['cancer_incidence_with_hiv', 'cancer_incidence_no_hiv']]
    for iv, ax in enumerate(axes.flatten()):
        val = to_plot[iv]
        if isinstance(val, list):
            for val_to_plot in val:
                label = 'HIV+' if 'with_hiv' in val_to_plot else 'HIV-'
                result = simres[val_to_plot][year_ind:]
                result = np.convolve(list(result), np.ones(5), "valid") / 5
                ax.plot(years[4:], result, label=label)
            ax.legend()
            ax.set_title('Cancer incidence (per 100k)')
        else:

            result = simres[val][year_ind:]
            if iv == 0:
                ax.plot(years, 100 * result, label='HPVsim')
            else:
                ax.plot(years, 100 * result)
            thembisa_val_lb = f'{val}_lb'
            thembisa_val_ub = f'{val}_ub'
            if iv == 0:
                ax.scatter(years, 100 * rsa_df[thembisa_val_lb][:-10], marker='_', label='Thembisa,\n95% uncertainty',
                           color='grey')
                ax.scatter(years, 100 * rsa_df[thembisa_val_ub][:-10], marker='_', color='grey')
            else:
                ax.scatter(years, 100 * rsa_df[thembisa_val_lb][:-10], marker='_', color='grey')
                ax.scatter(years, 100 * rsa_df[thembisa_val_ub][:-10], marker='_', color='grey')
            ax.set_title(title_dict[val])
            if iv == 0:
                ax.legend(title='Source')
        ax.set_ylim(bottom=0)
        sc.SIticks(ax)
    fig.tight_layout()
    fig_name = f'figures/hiv_fit_{location}.png'
    sc.savefig(fig_name, dpi=100)

    from scipy.stats import weibull_min
    n_sample = 1000
    fig, axes = pl.subplots(1,2, figsize=(8,8))
    for i, adjust in enumerate([1, 2]):
        ax=axes[i]
        for age in [15, 20, 30, 40]:
            shape = sim['hiv_pars']['time_to_hiv_death_shape']
            scale = sim['hiv_pars']['time_to_hiv_death_scale'](age)
            scale = np.maximum(scale, 0)
            time_to_hiv_death = adjust * weibull_min.rvs(c=shape, scale=scale, size=n_sample)
            ax.hist(time_to_hiv_death, alpha=0.4, label=f'age {age}')

        title = ', 2x longer' if i == 1 else ''
        ax.set_title(f'HIV progression (not on ART){title}')
        ax.set_xlabel('Time to HIV mortality')
    axes[0].legend(title='Age')
    fig.tight_layout()
    fig_name = f'figures/hiv_progression.png'
    sc.savefig(fig_name, dpi=100)

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

