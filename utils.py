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


def plot_residual_burden(location=None, vx_scens=None, screen_scens=None,
                         label_dict=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''
    
    set_font(size=20)

    alldfs = sc.autolist()
    for screen_scen_label in screen_scens:
        for vx_scen_label in vx_scens:
            filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}'
            try:
                alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                alldfs += alldf
            except:
                print(f'df not available for {filestem_label}')

    bigdf = pd.concat(alldfs)

    colors = sc.gridcolors(20)
    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancer cases',
                                          'asr_cancer_incidence': 'Age standardized cervical cancer incidence rate (per 100,000)',}.items()):
        fig, ax = pl.subplots(figsize=(18, 10))
        cn = 0
        for screen_scen_label in screen_scens:
            for vx_scen_label in vx_scens:
                df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)].groupby('year')[
                    [f'{res}', f'{res}_low', f'{res}_high']].sum()
                if len(df):
                    years = np.array(df.index)[50:106]
                    best = np.array(df[res])[50:106]
                    low = np.array(df[f'{res}_low'])[50:106]
                    high = np.array(df[f'{res}_high'])[50:106]
                    label = f'{screen_scen_label}, {vx_scen_label}'
                    ax.plot(years, best, color=colors[cn], label=label)
                    ax.fill_between(years, low, high, color=colors[cn], alpha=0.3)
                    cn += 1


        if res == 'asr_cancer_incidence':
            ax.plot(years, np.full(len(years), fill_value=4), linestyle='dashed', label='Elimination target')
        ax.set_ylim(bottom=0)
        ax.legend()
        # ax.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True)
        sc.SIticks(ax)
        ax.set_ylabel(f'{reslabel}')
        ax.set_title(f'{reslabel} in {location.capitalize()}')
        fig.tight_layout()
        fig_name = f'{figfolder}/{res}_residual_burden_{location}.png'
        sc.savefig(fig_name, dpi=100)

    return


def plot_txv_impact_time_series(location=None, vx_scens=None, screen_scens=None, txvx_scens=None, label_dict=None,
                                fig_filestem=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)

    alldfs = sc.autolist()
    for screen_scen_label in screen_scens:
        for vx_scen_label in vx_scens:
            for txvx_scen_label in txvx_scens:
                filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                alldfs += alldf
    bigdf = pd.concat(alldfs)

    colors = sc.gridcolors(20)
    n_vx = len(vx_scens)
    n_sc = len(screen_scens)

    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancer cases',
                                          'asr_cancer_incidence': 'CC incidence rate (per 100,000)', }.items()):
        fig, axes = pl.subplots(n_vx, n_sc, figsize=(16, 16), sharex=True, sharey=True)
        for sn, screen_scen_label in enumerate(screen_scens):
            for vn, vx_scen_label in enumerate(vx_scens):
                if n_vx > 1 and n_sc > 1:
                    ax = axes[vn, sn]
                elif n_vx == 1:
                    ax = axes[sn]
                elif n_sc == 1:
                    ax = axes[vn]
                for tn, txvx_scen_label in enumerate(txvx_scens):
                    df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()

                    years = np.array(df.index)[70:106]
                    best = np.array(df[res])[70:106]
                    low = np.array(df[f'{res}_low'])[70:106]
                    high = np.array(df[f'{res}_high'])[70:106]
                    ax.plot(years, best, color=colors[tn], label=txvx_scen_label)
                    ax.fill_between(years, low, high, color=colors[tn], alpha=0.3)
                if res == 'asr_cancer_incidence':
                    ax.plot(years, np.full(len(years), fill_value=4), linestyle='dashed', label='Elimination target')
                ax.set_ylim(bottom=0)
                if sn == 2 and vn == 1:
                    ax.legend(bbox_to_anchor=(1.05, 0.95), fancybox=True)
                sc.SIticks(ax)
                if sn == 0:
                    ax.set_ylabel(f'{reslabel}')
                ax.set_title(f'{screen_scen_label},\n{vx_scen_label}')

        fig.suptitle(f'{reslabel} in {location.capitalize()}')
        fig.tight_layout()
        fig_name = f'{figfolder}/{res}_{location}{fig_filestem}.png'
        sc.savefig(fig_name, dpi=100)

    return


def plot_txv_relative_impact(locations=None, vx_scens=None, screen_scens=None, txvx_scens=None, label_dict=None,
                             fig_filestem=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    bigdfs = sc.autolist()
    for location in locations:
        alldfs = sc.autolist()
        for screen_scen_label in screen_scens:
            for vx_scen_label in vx_scens:
                for txvx_scen_label in txvx_scens:
                    filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                    alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                    alldfs += alldf
        bigdf = pd.concat(alldfs)
        bigdfs += bigdf
    bigdf = pd.concat(bigdfs)

    colors = sc.gridcolors(20)
    n_vx = len(vx_scens)
    n_sc = len(screen_scens)

    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancers', 'cancer_deaths': 'Cervical cancer deaths'}.items()):
        fig, axes = pl.subplots(n_vx, n_sc, figsize=(16, 16), sharex=True, sharey=True)
        for sn, screen_scen_label in enumerate(screen_scens):
            for vn, vx_scen_label in enumerate(vx_scens):
                ax = axes[vn, sn]
                x = np.arange(len(txvx_scens))
                NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == 'No TxV')].groupby('year')[[f'{res}']].sum()

                NoTxV = np.sum(np.array(NoTxV_df[res])[70:106])
                for tn, txvx_scen_label in enumerate(txvx_scens):
                    df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()

                    best = np.sum(np.array(df[res])[70:106])
                    averted = NoTxV - best
                    perc_averted = averted/NoTxV
                    ax.bar(x[tn], 100*perc_averted, color=colors[tn])
                    # ax.bar(x[tn], averted, color=colors[tn])

                ax.set_xticks(x, txvx_scens, rotation=90)
                sc.SIticks(ax)
                if sn == 0:
                    ax.set_ylabel(f'{reslabel} averted')
                ax.set_title(f'{screen_scen_label},\n{vx_scen_label}')

        fig.tight_layout()
        fig_name = f'{figfolder}/{res}_perc_averted{fig_filestem}.png'
        sc.savefig(fig_name, dpi=100)


    return

def plot_txv_relative_impact_pairedPxV(location=None, vx_scens=None, screen_scens=None, txvx_scens=None,
                             fig_filestem=None, label_dict=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)
    all_txvs = ['No TxV'] + txvx_scens
    alldfs = sc.autolist()
    for screen_scen_label in screen_scens:
        for vx_scen_label in vx_scens:
            for txvx_scen_label in all_txvs:
                if txvx_scen_label == 'No TxV':
                    filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                    alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                    alldfs += alldf
                else:
                    for paired_px in ['', ', paired PxV']:
                        txvx_scen_label = f'{txvx_scen_label}{paired_px}'
                        filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                        alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                        alldfs += alldf
    bigdf = pd.concat(alldfs)

    colors = sc.gridcolors(20)
    n_vx = len(vx_scens)
    n_sc = len(screen_scens)

    if fig_filestem == 'perc_averted_pairedPxV':
        perc = True
    else:
        perc = False

    fig, axes = pl.subplots(n_vx, n_sc, figsize=(16, 16), sharex=True, sharey=True)
    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancers', 'cancer_deaths': 'Cervical cancer deaths'}.items()):
        for sn, screen_scen_label in enumerate(screen_scens):
            for vn, vx_scen_label in enumerate(vx_scens):
                ax = axes[vn, sn]
                x = np.arange(len(txvx_scens))
                NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == 'No TxV')].groupby('year')[[f'{res}']].sum()

                NoTxV = np.sum(np.array(NoTxV_df[res])[70:106])
                for tn, txvx_scen_label in enumerate(txvx_scens):
                    txvx_scen_label_PxV = f'{txvx_scen_label}, paired PxV'
                    PxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label_PxV)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()
                    no_PxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()

                    width = 0.35

                    best_noPxV = np.sum(np.array(no_PxV_df[res])[70:106])
                    averted_noPxV = NoTxV - best_noPxV
                    perc_averted_noPxV = 100*averted_noPxV/NoTxV
                    if perc:
                        to_plot = 100*averted_noPxV/NoTxV
                    else:
                        to_plot = averted_noPxV

                    if ir:
                        nopxv_handle_death, = ax.bar(x[tn]-width/2, to_plot, width, color=colors[tn], hatch='xx')
                    else:
                        nopxv_handle_cancer, = ax.bar(x[tn]+width/2, to_plot, width, color=colors[tn])

                    best_PxV = np.sum(np.array(PxV_df[res])[70:106])
                    averted_PxV = NoTxV - best_PxV
                    perc_averted_PxV = 100*averted_PxV / NoTxV
                    if perc:
                        to_plot_diff = perc_averted_PxV - perc_averted_noPxV
                    else:
                        to_plot_diff = averted_PxV - averted_noPxV
                    if ir:
                        pxv_handle, = ax.bar(x[tn]-width/2, to_plot_diff, width, bottom=to_plot, alpha=0.3, color=colors[tn], hatch='xx')
                    else:
                        pxv_handle, = ax.bar(x[tn]+width/2, to_plot_diff, width, bottom=to_plot, alpha=0.3, color=colors[tn])

                    if sn + vn + tn + ir == 0:
                        ax.legend([nopxv_handle_cancer, pxv_handle], ['No PxV', 'Paired PxV'])
                ax.set_xticks(x, txvx_scens, rotation=90)
                sc.SIticks(ax)
                if sn == 0:
                    if perc:
                        ax.set_ylabel(f'Percent averted')
                    else:
                        ax.set_ylabel(f'Number averted')
                ax.set_title(f'{screen_scen_label},\n{vx_scen_label}')

    axes[0,2].legend([nopxv_handle_cancer, nopxv_handle_death], ['Cervical cancers', 'Cervical cancer deaths'])

    fig.tight_layout()
    fig_name = f'{figfolder}/{location}_{fig_filestem}.png'
    sc.savefig(fig_name, dpi=100)


    return

def plot_txv_relative_impact_pairedPxV_combined(locations=None, vx_scens=None, screen_scens=None, txvx_scens=None,
                                                label_dict=None, fig_filestem=None):
    '''
    Plot the residual burden of HPV under different scenarios
    '''

    set_font(size=20)

    all_txvs = ['No TxV'] + txvx_scens
    bigdfs = sc.autolist()
    for location in locations:
        alldfs = sc.autolist()
        for screen_scen_label in screen_scens:
            for vx_scen_label in vx_scens:
                for txvx_scen_label in all_txvs:
                    if txvx_scen_label == 'No TxV':
                        filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                        alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                        alldfs += alldf
                    else:
                        for paired_px in ['', ', paired PxV']:
                            txvx_scen_label = f'{txvx_scen_label}{paired_px}'
                            filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}_{label_dict[txvx_scen_label]}'
                            alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                            alldfs += alldf
        bigdf = pd.concat(alldfs)
        bigdfs += bigdf
    bigdf = pd.concat(bigdfs)


    colors = sc.gridcolors(20)
    n_vx = len(vx_scens)
    n_sc = len(screen_scens)

    if fig_filestem == 'perc_averted_pairedPxV':
        perc = True
    else:
        perc = False

    fig, axes = pl.subplots(n_vx, n_sc, figsize=(16, 16), sharex=True, sharey=True)
    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancers', 'cancer_deaths': 'Cervical cancer deaths'}.items()):
        for sn, screen_scen_label in enumerate(screen_scens):
            for vn, vx_scen_label in enumerate(vx_scens):
                ax = axes[vn, sn]
                x = np.arange(len(txvx_scens))
                NoTxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == 'No TxV')].groupby('year')[[f'{res}']].sum()

                NoTxV = np.sum(np.array(NoTxV_df[res])[70:106])
                for tn, txvx_scen_label in enumerate(txvx_scens):
                    txvx_scen_label_PxV = f'{txvx_scen_label}, paired PxV'
                    PxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label_PxV)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()
                    no_PxV_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                               & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                        [f'{res}', f'{res}_low', f'{res}_high']].sum()

                    width = 0.35

                    best_noPxV = np.sum(np.array(no_PxV_df[res])[70:106])
                    averted_noPxV = NoTxV - best_noPxV
                    perc_averted_noPxV = 100*averted_noPxV/NoTxV
                    if perc:
                        to_plot = 100*averted_noPxV/NoTxV
                    else:
                        to_plot = averted_noPxV

                    if ir:
                        nopxv_handle_death, = ax.bar(x[tn]-width/2, to_plot, width, color=colors[tn], hatch='xx')
                    else:
                        nopxv_handle_cancer, = ax.bar(x[tn]+width/2, to_plot, width, color=colors[tn])

                    best_PxV = np.sum(np.array(PxV_df[res])[70:106])
                    averted_PxV = NoTxV - best_PxV
                    perc_averted_PxV = 100*averted_PxV / NoTxV
                    if perc:
                        to_plot_diff = perc_averted_PxV - perc_averted_noPxV
                    else:
                        to_plot_diff = averted_PxV - averted_noPxV
                    if ir:
                        pxv_handle, = ax.bar(x[tn]-width/2, to_plot_diff, width, bottom=to_plot, alpha=0.3, color=colors[tn], hatch='xx')
                    else:
                        pxv_handle, = ax.bar(x[tn]+width/2, to_plot_diff, width, bottom=to_plot, alpha=0.3, color=colors[tn])

                    if sn + vn + tn + ir == 0:
                        ax.legend([nopxv_handle_cancer, pxv_handle], ['No PxV', 'Paired PxV'])
                ax.set_xticks(x, txvx_scens, rotation=90)
                sc.SIticks(ax)
                if sn == 0:
                    if perc:
                        ax.set_ylabel(f'Percent averted')
                    else:
                        ax.set_ylabel(f'Number averted')
                ax.set_title(f'{screen_scen_label},\n{vx_scen_label}')

    axes[0,2].legend([nopxv_handle_cancer, nopxv_handle_death], ['Cervical cancers', 'Cervical cancer deaths'])

    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}.png'
    sc.savefig(fig_name, dpi=100)


    return

def plot_fit(data=None, locations=None, filestem=None, fig_filestem=None):
    '''
    Plot the age-standardized and crude incidence rate
    '''

    set_font(size=24)

    try:
        bigdf = sc.loadobj(f'{resfolder}/{filestem}.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_{filestem}.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)

    colors = sc.gridcolors(10)

    fig, ax = pl.subplots(figsize=(16, 10))
    for ir, (res, reslabel) in enumerate(
            {'total_cancer_incidence': 'Crude incidence',
             'asr_cancer': 'Age standardized incidence', }.items()):

        df = bigdf[(bigdf.scen_label == 'No screening')].groupby('year')[[f'{res}', f'{res}_low', f'{res}_high']].sum()

        years = np.array(df.index)[50:106]
        best = np.array(df[res])[50:106]
        low = np.array(df[f'{res}_low'])[50:106]
        high = np.array(df[f'{res}_high'])[50:106]

        ax.plot(years, best, color=colors[ir], label=reslabel)
        ax.fill_between(years, low, high, color=colors[ir], alpha=0.3)

    ax.legend(loc='best', fancybox=True)
    sc.SIticks(ax)
    ax.set_ylabel(f'new cancers per 100,000')
    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}.png'
    sc.savefig(fig_name, dpi=100)

    return


def plot_ICER(locations=None, scens=None, filestem=None, fig_filestem=None):
    '''
    Plot the residual burden of HPV
    '''

    set_font(size=24)

    try:
        bigdf = sc.loadobj(f'{resfolder}/{filestem}.obj')
    except:
        print('bigdf not available, trying to load for each location and generate it')
        alldfs = sc.autolist()
        for location in locations:
            alldf = sc.loadobj(f'{resfolder}/{location}_{filestem}.obj')
            alldfs += alldf
        bigdf = pd.concat(alldfs)

    cancers = dict()
    cancer_deaths = dict()
    cin_treatments = dict()

    for cn, scen_label in enumerate(scens):
        df = bigdf[(bigdf.scen_label == scen_label)].groupby('year')[['total_cancers', 'total_cancer_deaths', 'n_cin_treated']].sum()
        cancers[scen_label] = np.array(df['total_cancers'])[50:106].sum()
        cancer_deaths[scen_label] = np.array(df['total_cancer_deaths'])[50:106].sum()
        cin_treatments[scen_label] = np.array(df['n_cin_treated'])[50:106].sum()

    data_for_plot = pd.DataFrame()
    data_for_plot['scen'] = np.array(list(cancers.keys()))
    data_for_plot['cases'] = np.array(list(cancers.values()))
    data_for_plot['deaths'] = np.array(list(cancer_deaths.values()))
    data_for_plot['cin_txs'] = np.array(list(cin_treatments.values()))

    colors = sc.gridcolors(len(data_for_plot))
    fig, axes = pl.subplots(1, 2, figsize=(16, 8), sharey=True)
    grouped = data_for_plot.groupby('scen')
    for i, (key, group) in enumerate(grouped):
        group.plot(ax=axes[0], kind='scatter', x='cases', y='cin_txs', label=key, color=colors[i], s=100)
        group.plot(ax=axes[1], kind='scatter', x='deaths', y='cin_txs', label=key, color=colors[i], s=100)

    axes[0].set_xlabel('Cancer cases')
    axes[1].set_xlabel('Cancer deaths')
    axes[0].set_ylabel('CIN treatments')
    axes[0].get_legend().remove()
    axes[1].legend(loc='upper center', bbox_to_anchor=(1.65, 0.95), fancybox=True, title='Screening method')
    sc.SIticks(axes[0])
    sc.SIticks(axes[1])
    fig.tight_layout()
    fig_name = f'{figfolder}/{fig_filestem}.png'
    sc.savefig(fig_name, dpi=100)
    return


def gauss2d(x, y, z, xi, yi, eps=1.0, xscale=1.0, yscale=1.0):
    def arr32(arr): return np.array(arr, dtype=np.float32)
    def f32(x):     return np.float32(x)
    x, y, z, xi, yi = arr32(x), arr32(y), arr32(z), arr32(xi), arr32(yi)
    eps, xscale, yscale = f32(eps), f32(xscale), f32(yscale)

    # Actual computation
    nx = len(xi)
    ny = len(yi)
    zz = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            dist = np.sqrt(((x - xi[i])/xscale)**2 + ((y - yi[j])/yscale)**2)
            weights = np.exp(-(dist/eps)**2)
            weights = weights/np.sum(weights)
            val = np.sum(weights*z)
            zz[j,i] = val

    return np.array(zz, dtype=np.float64) # Convert back



def plot_sweeps(location=None, tx_vx_scens=None, filestem=None, z_scale=1e6):

    fulldf = sc.loadobj(f'{resfolder}/{location}_sweep_results_{filestem}.obj')
    fig = pl.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, len(tx_vx_scens)+1, width_ratios=[10,10,10, 1])
    cmaps = ['plasma', 'viridis', 'jet']

    background_labels = {
        '90vx_9to14_35sc_50tx': '90% PxV 9-14, 35% screen coverage, 50% treatment coverage',

    }

    labels = {
        'cancers_averted': 'Cancers averted (millions)',
        'perc_cancers_averted': 'Percent of cancers averted',
        'NNV': 'NNV to avert a cancer case'
    }

    zmax_dict = dict(cancers_averted=sc.autolist(),
                     perc_cancers_averted=sc.autolist(),
                     NNV=sc.autolist())


    # get_maxes = fulldf.groupby(['txvx_age','lo_eff','hi_eff']).sum().reset_index()
    for zn, z_val in enumerate(['cancers_averted', 'perc_cancers_averted', 'NNV']):
        if z_val == 'NNV':
            z = np.array(fulldf['n_tx_vaccinated'] / fulldf['cancers_averted'])
        elif z_val == 'cancers_averted':
            z = np.array(fulldf[z_val]) / z_scale
        else:
            z = 100*np.array(fulldf[z_val])
        zmax_dict[z_val] = (np.min(z), np.max(z))

    for pn, tx_vx_scen in enumerate(tx_vx_scens):
        df = fulldf[(fulldf.txvx_age==tx_vx_scen)]
        print(f'TxV age {tx_vx_scen}: {np.mean(df["n_tx_vaccinated"])} women tx vaccinated')
        x = np.array(df['lo_eff'])
        y = np.array(df['hi_eff'])
        for zn, z_val in enumerate(['cancers_averted', 'perc_cancers_averted', 'NNV']):
            if z_val == 'NNV':
                z = np.array(df['n_tx_vaccinated']/df['cancers_averted'])
            elif z_val == 'cancers_averted':
                z = np.array(df[z_val])/z_scale
            else:
                z = 100*np.array(df[z_val])

            ax = fig.add_subplot(gs[zn, pn])
            z_min, z_max = zmax_dict[z_val]

            npts = 100
            scale = 0.08

            xi = np.linspace(0, 1, npts)
            yi = np.linspace(0, 1, npts)
            xx, yy = np.meshgrid(xi, yi)
            zz = sc.gauss2d(x, y, z, xi, yi, scale=scale, xscale=1, yscale=1, grid=True)
            scolors = sc.vectocolor(z, cmap=cmaps[zn], minval=z_min, maxval=z_max)
            ima = ax.contourf(xx, yy, zz, cmap=cmaps[zn], levels=np.linspace(z_min, z_max, 100))
            # ax.scatter(x, y, marker='o', c=scolors, edgecolor=[0.3] * 3, s=50, linewidth=0.1, alpha=0.5)
            ax.contour(xx, yy, zz, levels=7, linewidths=0.5, colors='k')

            ax.set_xlabel('Virological clearance efficacy', fontsize=18)
            ax.set_ylabel('Lesion regression efficacy', fontsize=18)
            # ax.set_xlim([0, 1])
            # ax.set_ylim([0, 1])
            if zn == 0:
                ax.set_title(f'TxV age {tx_vx_scen}', fontsize=24)

            if pn == (len(tx_vx_scens)-1):
                # Colorbar
                zmin, zmax = zmax_dict[z_val]
                ima.set_clim((zmin, zmax))

                cbargs = dict(labelpad=15, rotation=270, fontsize=16)
                axc = fig.add_subplot(gs[zn, -1])
                cb = pl.colorbar(ima, ticks=np.linspace(zmin, zmax, 10), cax=axc)
                cb.set_label(labels[z_val], **cbargs)
                sc.commaticks(axc)

    # fig.suptitle(f'{background_labels[filestem]}, {location.capitalize()}', fontsize=22)
    fig.tight_layout()
    fig_name = f'{figfolder}/{location}_sweeps_{filestem}.png'
    pl.savefig(fig_name, dpi=100)




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
