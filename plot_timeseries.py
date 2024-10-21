"""
Plot time series figure
"""


import pylab as pl
import sciris as sc
from run_scenarios import coverage_arr, make_hiv_scenarios
import utils as ut

 
def plot_single(ax, mres, to_plot, si, color, label=None):
    years = mres.year[si:]
    best = mres[to_plot][si:]
    low = mres[to_plot].low[si:]
    high = mres[to_plot].high[si:]
    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.5, color=color)
    return ax


def plot_timeseries(msim_dict):

    ut.set_font(16)

    # How many HIV scenarios?
    hiv_scens = make_hiv_scenarios()
    n_hiv_scens = len(hiv_scens)

    # Make colors and figure
    colors = sc.gridcolors(n_hiv_scens)
    fig, axes = pl.subplots(3, 1, figsize=(11, 10))
    axes = axes.flatten()
    start_year = 1980

    # Insert plotting code

    fig.tight_layout()
    fig_name = 'figures/vx_scens.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens.obj')
    plot_timeseries(msim_dict)




