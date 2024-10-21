"""
Plotting and other utilities
"""

import sciris as sc


def set_font(size=None, font='Libertinus Sans'):
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return

