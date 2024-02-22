"""
This file is used to run calibrations for TxV 10-country analysis.

Instructions: Go to the CONFIGURATIONS section on lines 29-36 to set up the script before running it.
"""

# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs


# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    # 'run_calibration',  # Make sure this is uncommented if you want to _run_ the calibrations (usually on VMs)
    'plot_calibration',  # Make sure this is uncommented if you want to _plot_ the calibrations (usually locally)
]
debug = False  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

# Run settings for calibration (dependent on debug)
n_trials = [15000, 10][debug]  # How many trials to run for calibration
n_workers = [40, 1][debug]  # How many cores to use
storage = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations


########################################################################
# Run calibration
########################################################################
def make_priors():
    default = dict(
        rel_beta=[0.9, 0.8, 1.2, 0.05],
        cancer_fn=dict(ld50=[25, 20, 40, 1]),
        dur_cin=dict(par1=[14, 5, 25, 0.5],
                     par2=[20, 10, 25, 0.5])
    )

    genotype_pars = dict(
        hpv18=sc.dcp(default),
        hi5=sc.dcp(default),
        ohr=sc.dcp(default),
        hpv16=dict(
            cancer_fn=dict(ld50=[20, 15, 30, 1]),
            dur_cin=dict(par1=[17, 10, 25, 0.5],
                         par2=[20, 10, 25, 0.5])
        ),
    )

    return genotype_pars


def run_calib(location=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):
    dflocation=location.replace(" ", "_")
    if location == 'south africa':
        hiv_datafile = ['data/hiv_incidence_south_africa.csv',
                        'data/south_africa_female_hiv_mortality.csv',
                        'data/south_africa_male_hiv_mortality.csv']
        art_datafile = ['data/south_africa_art_coverage_by_age_males.csv',
                        'data/south_africa_art_coverage_by_age_females.csv']

    else:
        hiv_datafile = None
        art_datafile = None

    sim = rs.make_sim(location, hiv_datafile=hiv_datafile, art_datafile=art_datafile, calib=True, art_sens=True)
    datafiles = [
        f'data/{dflocation}_cancer_cases.csv', #Globocan
        # f'data/{dflocation}_cancer_incidence_by_age_no_hiv.csv', #https://onlinelibrary.wiley.com/doi/10.1002/ijc.34707
        # f'data/{dflocation}_cancer_incidence_by_age_with_hiv.csv', #https://onlinelibrary.wiley.com/doi/10.1002/ijc.34707
        f'data/{dflocation}_asr_cancer_incidence.csv',
        f'data/{dflocation}_cin_types.csv',
        f'data/{dflocation}_cancer_types.csv',
    ]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.05, 0.02, 0.5, 0.02],
        own_imm_hr=[0.5, 0.25, 1, 0.05],
        age_risk=dict(risk=[3.2, 1, 4, 0.1],
                      age=[38, 30, 45, 1]),
        # sev_dist=dict(par1=[1, 1, 2, 0.1]),
        cell_imm_init=dict(par1=[0.2, 0.2, 0.8, 0.05]),
        # hpv_control_prob=[0,0,1, 0.25],
        # hpv_reactivation=[0.025, 0, 0.1, 0.025]
    )

    if location is None:
        sexual_behavior_pars = dict(
            m_cross_layer=[0.9, 0.5, 0.95, 0.05],
            m_partners=dict(
                c=dict(par1=[10, 5, 12, 1])
            ),
            f_cross_layer=[0.1, 0.05, 0.5, 0.05],
            f_partners=dict(
                c=dict(par1=[1, .5, 2, .1], par2=[.2, .1, 1, .05])
            )
        )
    else:
        sexual_behavior_pars = dict(
            m_cross_layer=[0.3, 0.1, 0.7, 0.05],
            m_partners=dict(
                c=dict(par1=[0.5, 0.1, 0.6, 0.05])
            ),
            f_cross_layer=[0.4, 0.05, 0.7, 0.05],
            f_partners=dict(
                c=dict(par1=[0.2, 0.1, 0.6, 0.05])
            )
        )
    calib_pars = sc.mergedicts(calib_pars, sexual_behavior_pars)

    genotype_pars = make_priors()

    hiv_pars = dict(
        rel_sus=dict(
            lt200=[2.25, 2, 5, 0.25],
            # gt200=[2.25, 2, 4, 0.25]
        ),
        rel_sev=dict(
            lt200=[1.5, 1.25, 5, 0.25],
            # gt200=[1.5, 1.25, 3, 0.25]
        ),
        # rel_reactivation_prob=[3, 2, 5, 0.5]
    )

    # Save some extra sim results
    extra_sim_result_keys = ['cancers', 'cancers_with_hiv', 'cancers_no_hiv',
                             'cancers_by_age_with_hiv', 'cancers_by_age_no_hiv',
                             'asr_cancer_incidence', 'cancer_incidence_by_age_with_hiv',
                             'cancer_incidence_by_age_no_hiv']

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            hiv_pars=hiv_pars,
                            name=f'{location}_calib_final',
                            datafiles=datafiles,
                            extra_sim_result_keys=extra_sim_result_keys,
                            total_trials=n_trials, n_workers=n_workers,
                            storage=storage
                            )
    calib.calibrate()
    filename = f'{location}_calib{filestem}'
    if do_plot:
        calib.plot(do_save=True, fig_path=f'figures/{filename}.png')
    if do_save:
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib


########################################################################
# Load pre-run calibration
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, filestem=''):
    fnlocation = location.replace(' ', '_')
    filename = f'{fnlocation}_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')
    if do_plot:
        sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
        sc.options(font='Libertinus Sans')
        fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'figures/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        sc.save(f'results/{location}_pars{filestem}.obj', calib_pars)

    return calib


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    locations = [
        'south africa'
    ]


    # Run calibration - usually on VMs
    if 'run_calibration' in to_run:
        filestem = '_feb21_artsens'
        for location in locations:
            sim, calib = run_calib(location=location, n_trials=n_trials, n_workers=n_workers,
                                   do_save=do_save, do_plot=False, filestem=filestem)

    # Load the calibration, plot it, and save the best parameters -- usually locally
    if 'plot_calibration' in to_run:

        for location in locations:
            filestem = '_feb21_artsens'
            calib = load_calib(location=location, do_plot=True, save_pars=True, filestem=filestem)

            best_par_ind = calib.df.index[0]
            extra_sim_results = calib.extra_sim_results[best_par_ind]
            years = calib.sim.results['year']
            year_ind = sc.findinds(years, 1985)[0]
            import matplotlib.pylab as pl

            fig, axes = pl.subplots(3, 1)
            axes[0].plot(years[year_ind:], extra_sim_results['cancers_with_hiv'][year_ind:], label='HIV+')
            axes[0].plot(years[year_ind:], extra_sim_results['cancers_no_hiv'][year_ind:], label='HIV-')
            axes[0].plot(years[year_ind:], extra_sim_results['cancers'][year_ind:], label='Total')
            axes[0].set_title(f'Cancers over time')
            axes[0].legend()
            axes[1].plot(calib.sim.pars['age_bin_edges'][:-1],
                         extra_sim_results['cancer_incidence_by_age_with_hiv'][:, -2], label='HIV+')
            axes[1].plot(calib.sim.pars['age_bin_edges'][:-1],
                         extra_sim_results['cancer_incidence_by_age_no_hiv'][:, -2],
                         label='HIV-')
            axes[1].legend()

            axes[2].plot(years[year_ind:], extra_sim_results['asr_cancer_incidence'][year_ind:])

            fig.show()

    T.toc('Done')
