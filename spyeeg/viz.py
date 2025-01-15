#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:38:24 2020

@author: phg17
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
from scipy import signal
import mne

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb

def colormap_masked(ncolors=256, knee_index=None, cmap='inferno', alpha=0.3):
    """
    Create a colormap with value below a threshold being greyed out and transparent.

    Parameters
    ----------
    ncolors : int
        default to 256
    knee_index : int
        index from which transparency stops
        e.g. knee_index = np.argmin(abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))

    Returns
    -------
    cm : LinearSegmentedColormap
        Colormap instance
    """
    cm = plt.cm.get_cmap(cmap)(np.linspace(0, 1, ncolors))
    if knee_index is None:
        # Then map to pvals, as -log(p) between 0 and 3.5, and threshold at 0.05
        knee_index = np.argmin(
            abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))

    cm[:knee_index, :] = np.c_[cm[:knee_index, 0], cm[:knee_index, 1],
                               cm[:knee_index, 2], alpha*np.ones((len(cm[:knee_index, 1])))]
    return LinearSegmentedColormap.from_list('my_colormap', cm)


def plot_trf_signi(trf, reject, time_highlight=None, shades=None):
    """
    Plot trf with significant portions highlighted and with thicker lines.
    
    Parameters
    ----------
    trf: class
        trf class object
    reject: ndarray
        mask of trf coefficients to grey out, of shape (n_times, n_channels, n_feat)
    time_highlight: list or None
        list of [tmin, tmax] to highlight for each coefficients, of size (n_feats)
        if None, set depending on reject
    shades: array-like or None
        rgb values for highlights background
        if None, set to [.2, .2, .2] (light grey)

    Returns
    -------
    fig: mpl Figure
    ax: mpl Axes or array of Axes
    """
    fig, ax = trf.plot_kernel()
    signi_trf = np.ones_like(reject) * np.nan
    list_axes = plt.gcf().axes
    for feat, cax in enumerate(list_axes):
        if shades is None:
            color_shade = 'w' if np.mean(
                to_rgb(plt.rcParams['axes.facecolor'])) < .5 else [.2, .2, .2]
        else:
            color_shade = shades
        if time_highlight is None:
            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                             where=np.any(reject[:, feat, :], 1),
                             color=color_shade, alpha=0.2)
        else:  # fill regions of time of interest
            toi = np.zeros_like(trf.times, dtype=bool)
            for tlims in time_highlight[feat]:
                toi = np.logical_or(toi, np.logical_and(
                    trf.times >= tlims[0], trf.times < tlims[1]))

            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                             where=toi,
                             color=color_shade, alpha=0.2)
        lines = cax.get_lines()
        for k, l in enumerate(lines):
            signi_trf[reject[:, feat, k], feat, k] = l.get_data()[
                1][reject[:, feat, k]]
        newlines = cax.plot(trf.times, signi_trf[:, feat, :], linewidth=4)

    if ax is None:
        return plt.gcf()

    return fig, ax

def barplot_annotate_brackets(num1, num2, data, center, height, color='k', yerr=None, dh=.05, barh=.05, fontsize=None, maxasterix=None, figax = None):
    """ 
    Annotate barplot with p-values.

    Parameters
    ----------
    num1: int
        Index of first column
    num2: int
        Index of second column
    data: string | float
        - If string: Relevancy, e.g. '*', '**', 'n.s.'
        - If float: p-value
    center: array-like
        centers of all bars (plt.bar first input)
    height: array-like
        heights of all bars (plt.bar second input)
    yerr: array-like
        yerr of all bars (plt.bar second input)
    dh: float
        height offset over bar|bar+yerr in axes coordinates (0 to 1)
    barh: float
        bar height in axes coordinates (0 to 1)
    fontsize: int
        fontsize of asterisk
    maxasterix: int
        maximum number of asterixes to write (for very small p-values)
    figax: tuple
        (fig, ax) of existing bar plot. If None, plot on the latest figure.

    Returns
    -------
    None
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    if figax == None:
        plt.plot(barx, bary, c=color, linewidth=1.8)
    else:
        fig, ax = figax
        ax.plot(barx, bary, c=color, linewidth=1.8)

    kwargs_t = dict(ha='center', va='bottom')
    if fontsize is not None:
        kwargs_t['fontsize'] = fontsize

    if figax == None:
        plt.text(*mid, text, color=color, **kwargs_t)
    else:
        ax.text(*mid, text, color=color, **kwargs_t)
