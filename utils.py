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


def plot_impact(location=None, routine_coverage=None, rel_imm=None, discounting=False, filestem=''):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    bigdf = sc.loadobj(f'{resfolder}/{location}_results{filestem}.obj')
    econdf = sc.loadobj(f'{resfolder}/{location}_econ{filestem}.obj')

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 2020)[0]
    ye = sc.findinds(years, 2100)[0]
    yev  = sc.findinds(years, 2090)[0]
    standard_le = 88.8

    dfs = sc.autolist()
    for routine_cov in routine_coverage:
        for rel_imm_scen in rel_imm:
            summary_df = pd.DataFrame()
            plwh_df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.rel_imm == rel_imm_scen) & (bigdf.plwh == True)]
            df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.rel_imm == rel_imm_scen) & (bigdf.plwh == False)]

            econdf_cancers = econdf[(econdf.vx_coverage == routine_cov) & (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == False)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths']].sum()

            econdf_ages = econdf[(econdf.vx_coverage == routine_cov) & (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == False)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()
            econdf_plwh_cancers = econdf[(econdf.vx_coverage == routine_cov) & (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == True)].groupby('year')[
                    ['new_cancers', 'new_cancer_deaths', 'plwh_vaccinations']].sum()

            econdf_plwh_ages = econdf[(econdf.vx_coverage == routine_cov) & (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == True)].groupby('year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            cancers = econdf_cancers['new_cancers'].values
            cancer_deaths = econdf_cancers['new_cancer_deaths'].values
            cancers_plwh = econdf_plwh_cancers['new_cancers'].values
            cancer_deaths_plwh = econdf_plwh_cancers['new_cancer_deaths'].values
            additional_vaccinations = np.sum(econdf_plwh_cancers['plwh_vaccinations'].values)


            avg_age_ca_death = np.mean(econdf_ages['av_age_cancer_deaths'])
            avg_age_ca = np.mean(econdf_ages['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            dalys = yll + yld

            avg_age_ca_death_plwh = np.mean(econdf_plwh_ages['av_age_cancer_deaths'])
            avg_age_ca_plwh = np.mean(econdf_plwh_ages['av_age_cancers'])
            ca_years_plwh = avg_age_ca_death_plwh - avg_age_ca_plwh
            yld_plwh = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years_plwh * cancers_plwh)
            yll_plwh = np.sum((standard_le - avg_age_ca_death_plwh) * cancer_deaths_plwh)
            dalys_plwh = yll_plwh + yld_plwh

            summary_df['cancers_averted'] = [np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])]
            summary_df['cancer_deaths_averted'] = [np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye])]
            summary_df['dalys_averted'] = [dalys - dalys_plwh]
            summary_df['perc_cancers_averted'] = [
                100*(np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye]))/
                np.sum(np.array(df['cancers'])[ys:ye])]
            summary_df['perc_cancer_deaths_averted'] = [
                100*(np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye]))/
                np.sum(np.array(df['cancer_deaths'])[ys:ye])]
            summary_df['perc_dalys_averted'] = [100*(dalys - dalys_plwh)/dalys]
            summary_df['additional_vaccinations'] = [additional_vaccinations]
            summary_df['dalys_averted/dose'] = [1000*(dalys - dalys_plwh)/additional_vaccinations]
            summary_df['cancers_averted/dose'] = [1000 * (np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])) / additional_vaccinations]
            summary_df['cancer_deaths_averted/dose'] = [1000 * (np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye])) / additional_vaccinations]

            summary_df['vx_coverage'] = routine_cov
            summary_df['rel_imm'] = rel_imm_scen
            dfs += summary_df

    final_df = pd.concat(dfs)

    label_dict = {
        'cancers_averted': 'Cancers averted',
        'cancer_deaths_averted': 'Cancer deaths averted',
        'dalys_averted': 'DALYs averted',
        'perc_cancers_averted': 'Percent cancers averted',
        'perc_cancer_deaths_averted': 'Percent cancer deaths averted',
        'perc_dalys_averted': 'Percent DALYs averted',
        'cancers_averted/dose': 'Cancers averted \nper 1,000 doses',
        'cancer_deaths_averted/dose': 'Cancer deaths averted \nper 1,000 doses',
        'dalys_averted/dose': 'DALYs averted \nper 1,000 doses',

    }
    fig, axes = pl.subplots(3, 3, figsize=(12, 12), sharex=True)
    to_plot = ['cancers_averted', 'perc_cancers_averted', 'cancers_averted/dose',
               'cancer_deaths_averted', 'perc_cancer_deaths_averted', 'cancer_deaths_averted/dose',
               'dalys_averted', 'perc_dalys_averted', 'dalys_averted/dose']

    for i, ax in enumerate(axes.flatten()):
        val = to_plot[i]
        df_pivot = pd.pivot_table(
            final_df,
            values=val,
            index="vx_coverage",
            # columns="rel_imm"
        )
        df_pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(label_dict[val])
        ax.get_legend().remove()
        sc.SIticks(ax)

    axes[2,0].set_xlabel('Routine Vaccine Coverage')
    axes[2, 1].set_xlabel('Routine Vaccine Coverage')
    # axes[2,0].set_xticklabels(['20%', '40%', '80%'], rotation=0)
    # axes[2, 1].set_xticklabels(['20%', '40%', '80%'], rotation=0)

    fig.tight_layout()
    fig_name = f'{figfolder}/summary_{location}{filestem}.png'
    sc.savefig(fig_name, dpi=100)

    return


def make_datafiles(locations):
    ''' Get the relevant datafiles for the selected locations '''
    datafiles = dict()
    asr_locs            = ['drc', 'ethiopia', 'kenya', 'nigeria', 'tanzania', 'uganda']
    cancer_type_locs    = ['ethiopia', 'kenya', 'nigeria', 'tanzania', 'india', 'uganda']
    cin_type_locs       = ['nigeria', 'tanzania', 'india']

    for location in locations:
        dflocation = location.replace(' ','_')
        datafiles[location] = [
            f'data/{dflocation}_cancer_cases.csv',
        ]

        if location in asr_locs:
            datafiles[location] += [f'data/{dflocation}_asr_cancer_incidence.csv']

        if location in cancer_type_locs:
            datafiles[location] += [f'data/{dflocation}_cancer_types.csv']

        if location in cin_type_locs:
            datafiles[location] += [f'data/{dflocation}_cin_types.csv']

    return datafiles


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

def plot_ts(location=None, routine_coverage=None, plwh=None, filestem=''):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    bigdf = sc.loadobj(f'{resfolder}/{location}_results{filestem}.obj')

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 2010)[0]
    ye = sc.findinds(years, 2100)[0]

    colors = sc.gridcolors(20)
    ls = ['solid', 'dotted']


    fig, axes = pl.subplots(3, 1, figsize=(12, 12))
    for iv, val in enumerate(['cancers_with_hiv', 'cancers', 'cancer_deaths']):
        for ir, routine_cov in enumerate(routine_coverage):
            for ip, plwh_scen in enumerate(plwh):
                df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.plwh == plwh_scen) & bigdf.rel_imm == 1]
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
    axes[0].legend(title='Routine vaccine coverage')
    axes[1].legend(title='Vaccinate WLWH')
    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    fig.tight_layout()
    fig_name = f'{figfolder}/time_series_{location}{filestem}.png'
    sc.savefig(fig_name, dpi=100)

    return


def plot_hiv_ts_combined(location, routine_coverage, plwh, calib_filestem, filestems):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    dfs = sc.autolist()
    for filestem in filestems:
        df = sc.loadobj(f'{resfolder}/{location}_results{calib_filestem}_{filestem}.obj')
        df['mort_redux'] = filestem
        dfs.append(df)
    bigdf = pd.concat(dfs)

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 1985)[0]
    ye = sc.findinds(years, 2100)[0]

    colors = sc.gridcolors(5)

    rsa_df = pd.read_csv('data/RSA_data.csv').set_index('Unnamed: 0').T

    fig, axes = pl.subplots(3, 2, figsize=(12, 12))
    to_plot = ['female_hiv_prevalence', 'hiv_incidence', 'art_coverage', 'hiv_mortality', 'cancers', 'cancers_with_hiv']
    for iv, ax in enumerate(axes.flatten()):
        val = to_plot[iv]
        for i_redux, mort_redux in enumerate(filestems):
            mort_label = mort_redux.replace('mortredux', '')
            df = bigdf[(bigdf.vx_coverage == routine_coverage) & (bigdf.plwh == plwh) & (bigdf.mort_redux==mort_redux)
                       & bigdf.rel_imm == 1]
            years = np.array(df['year'])[ys:ye]
            result = np.array(df[val])[ys:ye]
            low = np.array(df[f'{val}_low'])[ys:ye]
            high = np.array(df[f'{val}_high'])[ys:ye]
            if val in ['female_hiv_prevalence', 'hiv_incidence', 'art_coverage']:
                result *= 100
                low *= 100
                high *= 100
            if iv == 0:
                ax.plot(years, result, color=colors[i_redux], label=mort_label)
            elif iv == 1:
                if i_redux == 0:
                    ax.plot(years, result, label='HPVsim', color=colors[i_redux])
                else:
                    ax.plot(years, result, color=colors[i_redux])
            else:
                ax.plot(years, result, color=colors[i_redux])
            ax.fill_between(years, low, high, alpha=0.3, color=colors[i_redux])
        if val in rsa_df.columns:
            if iv == 1:
                ax.scatter(years[:-69], 100 * rsa_df[val], label='Thembisa', color='grey')
            else:
                ax.scatter(years[:-69], 100 * rsa_df[val], color='grey')

        ax.set_title(val)
        if iv == 0:
            ax.legend(title='HIV mort redux')

        if iv == 1:
            ax.legend(title='Source')

        ax.set_ylim(bottom=0)
        sc.SIticks(ax)

    axes[2,1].sharey(axes[2,0])
    fig.tight_layout()
    fig_name = f'{figfolder}/hiv_time_series_{location}_compare.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_impact_combined(location, routine_coverage, calib_filestem, filestems):
    '''
        Plot the residual burden of HPV under different scenarios
        '''

    set_font(size=20)
    location = location.replace(' ', '_')
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for filestem in filestems:
        df = sc.loadobj(f'{resfolder}/{location}_results{calib_filestem}_{filestem}.obj')
        df['mort_redux'] = filestem
        dfs.append(df)

        econdf = sc.loadobj(f'{resfolder}/{location}_econ{calib_filestem}_{filestem}.obj')
        econdf['mort_redux'] = filestem
        econdfs.append(econdf)
    bigdf = pd.concat(dfs)
    econdf = pd.concat(econdfs)

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 2020)[0]
    ye = sc.findinds(years, 2100)[0]
    yev = sc.findinds(years, 2090)[0]
    standard_le = 88.8
    rel_imm_scen = 1

    dfs = sc.autolist()
    for routine_cov in routine_coverage:
        mortlabels=[]
        for i_redux, mort_redux in enumerate(filestems):
            mort_label = mort_redux.replace('mortredux', '')
            mortlabels.append(mort_label)
            summary_df = pd.DataFrame()
            plwh_df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.mort_redux==mort_redux) & (bigdf.rel_imm == rel_imm_scen) & (bigdf.plwh == True)]
            df = bigdf[(bigdf.vx_coverage == routine_cov) & (bigdf.mort_redux==mort_redux) & (bigdf.rel_imm == rel_imm_scen) & (bigdf.plwh == False)]

            econdf_cancers = econdf[(econdf.vx_coverage == routine_cov) & (econdf.mort_redux==mort_redux)& (econdf.rel_imm == rel_imm_scen) & (
                        econdf.plwh == False)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths', 'new_vaccinations']].sum()

            econdf_ages = econdf[(econdf.vx_coverage == routine_cov) & (econdf.mort_redux==mort_redux)& (econdf.rel_imm == rel_imm_scen) & (
                        econdf.plwh == False)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()
            econdf_plwh_cancers = econdf[
                (econdf.vx_coverage == routine_cov) & (econdf.mort_redux==mort_redux)& (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == True)].groupby(
                'year')[
                ['new_cancers', 'new_cancer_deaths', 'new_vaccinations']].sum()

            econdf_plwh_ages = econdf[
                (econdf.vx_coverage == routine_cov) & (econdf.mort_redux==mort_redux) & (econdf.rel_imm == rel_imm_scen) & (econdf.plwh == True)].groupby(
                'year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            cancers = econdf_cancers['new_cancers'].values
            cancer_deaths = econdf_cancers['new_cancer_deaths'].values
            cancers_plwh = econdf_plwh_cancers['new_cancers'].values
            cancer_deaths_plwh = econdf_plwh_cancers['new_cancer_deaths'].values
            vaccinations = econdf_cancers['new_vaccinations'].values
            vaccinations_plwh = econdf_plwh_cancers['new_vaccinations'].values
            additional_vaccinations = np.sum(vaccinations_plwh) - np.sum(vaccinations)

            avg_age_ca_death = np.mean(econdf_ages['av_age_cancer_deaths'])
            avg_age_ca = np.mean(econdf_ages['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            dalys = yll + yld

            avg_age_ca_death_plwh = np.mean(econdf_plwh_ages['av_age_cancer_deaths'])
            avg_age_ca_plwh = np.mean(econdf_plwh_ages['av_age_cancers'])
            ca_years_plwh = avg_age_ca_death_plwh - avg_age_ca_plwh
            yld_plwh = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years_plwh * cancers_plwh)
            yll_plwh = np.sum((standard_le - avg_age_ca_death_plwh) * cancer_deaths_plwh)
            dalys_plwh = yll_plwh + yld_plwh

            summary_df['cancers_averted'] = [
                np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])]
            summary_df['cancer_deaths_averted'] = [
                np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye])]
            summary_df['dalys_averted'] = [dalys - dalys_plwh]
            summary_df['perc_cancers_averted'] = [
                100 * (np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])) /
                np.sum(np.array(df['cancers'])[ys:ye])]
            summary_df['perc_cancer_deaths_averted'] = [
                100 * (np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(
                    np.array(plwh_df['cancer_deaths'])[ys:ye])) /
                np.sum(np.array(df['cancer_deaths'])[ys:ye])]
            summary_df['perc_dalys_averted'] = [100 * (dalys - dalys_plwh) / dalys]
            summary_df['additional_vaccinations'] = [additional_vaccinations]
            summary_df['dalys_averted/dose'] = [1000*(dalys - dalys_plwh)/additional_vaccinations]
            summary_df['cancers_averted/dose'] = [1000 * (np.sum(np.array(df['cancers'])[ys:ye]) - np.sum(np.array(plwh_df['cancers'])[ys:ye])) / additional_vaccinations]
            summary_df['cancer_deaths_averted/dose'] = [1000 * (np.sum(np.array(df['cancer_deaths'])[ys:ye]) - np.sum(np.array(plwh_df['cancer_deaths'])[ys:ye])) / additional_vaccinations]
            summary_df['vx_coverage'] = routine_cov
            summary_df['HIV mort redux'] = mort_label
            dfs += summary_df

    final_df = pd.concat(dfs)

    label_dict = {
        'cancers_averted': 'Cancers averted',
        'cancer_deaths_averted': 'Cancer deaths averted',
        'dalys_averted': 'DALYs averted',
        'perc_cancers_averted': 'Percent cancers averted',
        'perc_cancer_deaths_averted': 'Percent cancer deaths averted',
        'perc_dalys_averted': 'Percent DALYs averted',
        'cancers_averted/dose': 'Cancers averted \nper 1,000 doses',
        'cancer_deaths_averted/dose': 'Cancer deaths averted \nper 1,000 doses',
        'dalys_averted/dose': 'DALYs averted \nper 1,000 doses',

    }
    fig, axes = pl.subplots(3, 3, figsize=(12, 12), sharex=True)
    to_plot = ['cancers_averted', 'perc_cancers_averted', 'cancers_averted/dose',
               'cancer_deaths_averted', 'perc_cancer_deaths_averted', 'cancer_deaths_averted/dose',
               'dalys_averted', 'perc_dalys_averted', 'dalys_averted/dose']
    colors = sc.gridcolors(5)
    for i, ax in enumerate(axes.flatten()):
        val = to_plot[i]
        df_pivot = pd.pivot_table(
            final_df,
            values=val,
            index="vx_coverage",
            columns="HIV mort redux"
        )

        df_pivot = df_pivot.reindex(mortlabels, axis=1)
        df_pivot.plot(kind="bar", ax=ax, color=colors)
        ax.set_ylabel(label_dict[val])
        if i > 0:
            ax.get_legend().remove()
        sc.SIticks(ax)

    axes[2, 0].set_xlabel('Routine Vaccine Coverage')
    axes[2, 1].set_xlabel('Routine Vaccine Coverage')
    axes[2,2].set_xlabel('Routine Vaccine Coverage')
    # axes[2,0].set_xticklabels(['20%', '40%', '80%'], rotation=0)
    # axes[2, 1].set_xticklabels(['20%', '40%', '80%'], rotation=0)

    fig.tight_layout()
    fig_name = f'{figfolder}/summary_{location}_compare.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_hiv_ts(location, routine_coverage, plwh, filestem):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    location = location.replace(' ', '_')
    bigdf = sc.loadobj(f'{resfolder}/{location}_results{filestem}.obj')

    years = bigdf['year'].unique().astype(int)
    ys = sc.findinds(years, 1985)[0]
    ye = sc.findinds(years, 2100)[0]

    rsa_df = pd.read_csv('data/RSA_data.csv').set_index('Unnamed: 0').T

    title_dict = dict(
        female_hiv_prevalence='HIV prevalence, females 15+',
        hiv_incidence='HIV incidence',
        art_coverage='ART coverage',
    )

    fig, axes = pl.subplots(1, 3, figsize=(12, 6))
    to_plot = ['female_hiv_prevalence', 'art_coverage', ['cancer_incidence_with_hiv', 'cancer_incidence_no_hiv']]
    for iv, ax in enumerate(axes.flatten()):
        val = to_plot[iv]
        if isinstance(val, list):
            for val_to_plot in val:
                label = 'HIV+' if 'with_hiv' in val_to_plot else 'HIV-'
                df = bigdf[(bigdf.vx_coverage == routine_coverage) & (bigdf.plwh == plwh) & bigdf.rel_imm == 1]
                years = np.array(df['year'])[ys:ye]
                result = np.array(df[val])[ys:ye]
                low = np.array(df[f'{val}_low'])[ys:ye]
                high = np.array(df[f'{val}_high'])[ys:ye]
                ax.plot(years, result, label=label)
                ax.fill_between(years, low, high, alpha=0.3)
            ax.legend()
            ax.set_title('Cancer incidence')
        else:

            df = bigdf[(bigdf.vx_coverage == routine_coverage) & (bigdf.plwh == plwh) & bigdf.rel_imm == 1]
            years = np.array(df['year'])[ys:ye]
            result = np.array(df[val])[ys:ye]
            low = np.array(df[f'{val}_low'])[ys:ye]
            high = np.array(df[f'{val}_high'])[ys:ye]
            if val in ['female_hiv_prevalence', 'art_coverage']:
                result *= 100
                low *= 100
                high *= 100
            if iv == 0:
                ax.plot(years, result, label='HPVsim')
            else:
                ax.plot(years, result)

            ax.fill_between(years, low, high, alpha=0.3)

            if val in rsa_df.columns:
                if iv == 0:
                    ax.scatter(years[:-69], 100 * rsa_df[val], label='Thembisa', color='grey')
                else:
                    ax.scatter(years[:-69], 100 * rsa_df[val], color='grey')

            ax.set_title(title_dict[val])
            if iv == 0:
                ax.legend(title='Source')

        ax.set_ylim(bottom=0)
        sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f'{figfolder}/hiv_time_series_{location}{filestem}.png'
    sc.savefig(fig_name, dpi=100)
    return