"""
Plot impact bars
"""


import pylab as pl
import sciris as sc
from run_scenarios import coverage_arr, make_hiv_scenarios
import utils as ut


def plot_impact_bars(msim_dict):

    ut.set_font(16)

    # How many HIV scenarios?
    hiv_scens = make_hiv_scenarios()
    n_hiv_scens = len(hiv_scens)

    # Make colors and figure
    colors = sc.gridcolors(n_hiv_scens)
    fig, axes = pl.subplots(3, 3, figsize=(11, 10))
    axes = axes.flatten()

    # Insert plotting code

    fig.tight_layout()
    fig_name = 'figures/vx_impact.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens.obj')
    plot_impact_bars(msim_dict)




