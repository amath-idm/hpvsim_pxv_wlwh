'''
Define the HPVsim simulations for South Africa that are used as
the basis for the calibration, scenarios, and sweeps.

By default, all locations are run. To not run a location, comment out the line
below. For all three locations, this script should take 1-5 minutes to run.
'''

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
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import pars_data as dp
import pars_scenarios as sp
import utils as ut



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
def make_sim(location=None, calib=False, debug=0, datafile=None, hiv_datafile=None, calib_pars=None,
        art_datafile=None, vx_intv=None, end=None, seed=1):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''
    if end is None:
        end = 2100
    if calib:
        end = 2020

    # Parameters
    pars = dict(
        n_agents       = [50e3,1e3][debug],
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
        hiv_pars       = dict(rel_imm=dict( lt200=1,gt200=1))
    )


    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)
    # Analyzers
    analyzers = sc.autolist()
    interventions = sc.autolist()
    if not calib:
        if len(vx_intv):
            interventions += sp.get_vx_intvs(**vx_intv)

    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions,
                  datafile=datafile, hiv_datafile=hiv_datafile, art_datafile=art_datafile, rand_seed=seed)
    return sim




#%% Simulation running functions

def run_sim(location=None, vx_intv=None, calib_pars=None,
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

    # Make arguments
    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        vx_intv=vx_intv,
        calib_pars=calib_pars,
        end=end,
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
    dflocation = location.replace(' ', '_')
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

    T.toc('Done')

