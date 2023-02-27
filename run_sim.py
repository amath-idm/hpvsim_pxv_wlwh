'''
Define the HPVsim simulations for South Africa that are used as
the basis for the calibration, scenarios, and sweeps.

By default, all locations are run. To not run a location, comment out the line
below. For all three locations, this script should take 1-5 minutes to run.
'''

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
def make_sim_parts(location=None, calib=False, debug=0,
                   screen_intv=None, vx_intv=None,
                   end=None):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''
    if end is None:
        end = 2060
    if calib:
        end = 2020

    # Parameters
    pars = dict(
        n_agents       = [50e3,1e3][debug],
        dt             = [0.25,1.0][debug],
        start          = [1950,1980][debug],
        end            = end,
        location       = location,
        partners       = dp.partners[location],
        debut          = dp.debut[location],
        mixing         = dp.mixing[location],
        layer_probs    = dp.layer_probs[location],
        condoms        =dp.condoms[location],
        genotypes      = [16, 18, 'hrhpv'],
        ms_agent_ratio = 100,
        verbose        = 0.0,
        model_hiv      = True,
    )

    # Analyzers
    analyzers = sc.autolist()
    interventions = sc.autolist()
    if not calib:
        if len(vx_intv):
            interventions += sp.get_vx_intvs(**vx_intv)

        if len(screen_intv):
            interventions += sp.get_screen_intvs(**screen_intv)

    return pars, analyzers, interventions


def make_sim(pars=None, analyzers=None, interventions=None, datafile=None, hiv_datafile=None, art_datafile=None, seed=1):
    ''' Actually create the sim '''
    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions,
                  hiv_datafile=hiv_datafile, art_datafile=art_datafile, datafile=datafile, rand_seed=seed)
    return sim



#%% Simulation running functions

def run_sim(location=None,
            screen_intv=None, vx_intv=None,
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

    location_file = location.replace(' ', '_')

    hiv_datafile = f'data/hiv_incidence_{location_file}.csv'
    art_datafile = f'data/art_coverage_{location_file}.csv'

    # Make arguments
    args = make_sim_parts(location=location,
                          vx_intv=vx_intv, screen_intv=screen_intv,
                          end=end, debug=debug)
    sim = make_sim(hiv_datafile=hiv_datafile, art_datafile=art_datafile, *args)

    # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location # Store location in an easy-to-access place
    sim['rand_seed'] = seed # Set seed
    sim.label = f'{label}--{location}' # Set label

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()
        
    if do_save:
        sim.save(f'{ut.resfolder}/{location}.sim')
    
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
    
    # Run a single sim per location -- usually locally, can be used for sanity checking and debugging

    vx_scen = dict(
        routine_start_year=2020,
        catch_up_year=2025,
        vx_coverage=0.9,
        age_range=(9, 14)
    )
    screen_scen = dict(
        primary=dict(
            precin=0.3,
            cin1=0.3,
            cin2=0.93,
            cin3=0.93,
            cancerous=0.93
        ),
        screen_coverage=0.35,
        tx_coverage=0.5
    )  # Not varying S&T
    sim0 = run_sim(location='south africa', end=2050, vx_intv=vx_scen, screen_intv=screen_scen)
    sim0.plot()
    T.toc('Done')

