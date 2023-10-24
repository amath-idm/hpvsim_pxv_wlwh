'''
Utilities for HPVsim TxVx analyses, mostly related to plotting
'''

import sciris as sc
import numpy as np
import pandas as pd

import pylab as pl
import seaborn as sns
import hpvsim.plotting as hppl

import hpvsim as hpv


resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'


########################################################################
#%% Plotting utils
########################################################################

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def process_country_files(locations, top_results=100, do_save=True):
    '''
    Read in all country files and create a master dataframe
    '''

    dfs = []
    for i, location in enumerate(locations):
        file = f'{resfolder}/{location}_calib.obj'
        calib = sc.loadobj(file)
        thisdf = calib.df.sort_values(by=['mismatch'])[:top_results]
        thisdf['location'] = f'{location.capitalize()}'
        dfs.append(thisdf)

    all_calib_pars = pd.concat(dfs)
    if do_save:
        sc.save(f'{resfolder}/all_calib_pars.obj', all_calib_pars)

    return all_calib_pars


def plot_impact(location=None, routine_coverage=None, plwh=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    bigdf = sc.loadobj(f'{resfolder}/{location}_results.obj')

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 2020)[0]
    ye = sc.findinds(years, 2100)[0]
    yev  = sc.findinds(years, 2050)[0]

    dfs = sc.autolist()
    for routine_cov in routine_coverage:
        summary_df = pd.DataFrame()
        plwh_df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.plwh == True)]
        df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.plwh == False)]
        summary_df['cancers_averted'] = [np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])]
        summary_df['cancer_deaths_averted'] = [np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye])]
        summary_df['n_vaccinated'] = [plwh_df['n_vaccinated'][yev] - df['n_vaccinated'][yev]]
        summary_df['vx_coverage'] = routine_cov
        dfs += summary_df

    final_df = pd.concat(dfs)

    final_df['deaths_averted_FVP'] = 1000*final_df['cancer_deaths_averted']/final_df['n_vaccinated']

    fig, axes = pl.subplots(3, 1, figsize=(12, 12))
    for iv, val in enumerate(['cancers_averted', 'cancer_deaths_averted', 'deaths_averted_FVP']):

        df_pivot = pd.pivot_table(
            final_df,
            values=val,
            index="vx_coverage",
            # columns="plwh"
        )
        df_pivot.plot(kind="bar", ax=axes[iv])
        axes[iv].set_xlabel('Vaccine Coverage')
        axes[iv].set_ylabel(val)
        axes[iv].set_xticklabels(['0%', '40%', '80%'], rotation=0)
        sc.SIticks(axes[iv])

    fig.tight_layout()
    fig_name = f'{figfolder}/summary_{location}.png'
    sc.savefig(fig_name, dpi=100)

    return


########################################################################
#%% Other utils
########################################################################
def make_msims(sims, use_mean=True, save_msims=False):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_sc == sim.meta.inds[0]
        assert (s == 0) or i_s != sim.meta.inds[1]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_sc]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims:  # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def make_msims_sweeps(sims, use_mean=True, save_msims=False):
    ''' Take a slice of sims and turn it into a multisim '''
    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_txs, draw, i_s = sims[0].meta.inds
    for s,sim in enumerate(sims): # Check that everything except seed matches
        assert i_txs == sim.meta.inds[0]
        assert draw == sim.meta.inds[1]
        assert (s==0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_txs, draw]
    msim.meta.eff_vals = sc.dcp(sims[0].meta.eff_vals)
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')
    print(f'Processing multisim {msim.meta.vals.values()}...')

    if save_msims: # Generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def plot_ts(location=None, routine_coverage=None, plwh=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    bigdf = sc.loadobj(f'{resfolder}/{location}_results.obj')

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 2010)[0]
    ye = sc.findinds(years, 2100)[0]

    colors = sc.gridcolors(20)
    ls = ['solid', 'dotted']


    fig, axes = pl.subplots(2, 1, figsize=(12, 12))
    for iv, val in enumerate(['cancers', 'asr_cancer_incidence']):
        for ir, routine_cov in enumerate(routine_coverage):
            for ip, plwh_scen in enumerate(plwh):
                df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.plwh == plwh_scen)]
                years = np.array(df['year'])[ys:ye]
                result = np.array(df[val])[ys:ye]
                low = np.array(df[f'{val}_low'])[ys:ye]
                high = np.array(df[f'{val}_high'])[ys:ye]
                if iv + ip == 0:
                    axes[iv].plot(years, result, color=colors[ir], linestyle=ls[ip], label=routine_cov)
                    axes[iv].fill_between(years, low, high, color=colors[ir], alpha=0.3)
                elif iv == 1 and ir ==1:
                    axes[iv].plot(years, result, color=colors[ir], linestyle=ls[ip], label=plwh_scen)
                    axes[iv].fill_between(years, low, high, color=colors[ir], alpha=0.3)
                else:
                    axes[iv].plot(years, result, color=colors[ir], linestyle=ls[ip])
                    axes[iv].fill_between(years, low, high, color=colors[ir], alpha=0.3)


        axes[iv].set_title(val)
    axes[0].legend(title='Vaccine coverage')
    axes[1].legend(title='Vaccinate WLWH')
    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    fig.tight_layout()
    fig_name = f'{figfolder}/time_series_{location}.png'
    sc.savefig(fig_name, dpi=100)

    return
