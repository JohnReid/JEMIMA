#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import pandas as pds
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as npy

picklefilename = 'statsdf-01000-0015.pkl'
statsdf = pds.read_pickle(picklefilename)
statsdf.methodidx = pds.Categorical(statsdf.method)
statsdf.index = npy.arange(statsdf.shape[0])
statsdf.columns

colors = ['b', 'r', 'g']
methodcolors = [colors[label] for label in statsdf.methodidx.labels]
alpha = .05

plt.close('all')
statsdf.plot(
    kind='scatter', x='KLtrueestimate', y='KLestimatetrue',
    colors=methodcolors, alpha=alpha, loglog=True)
upperlim = min(statsdf.KLtrueestimate.max(), statsdf.KLestimatetrue.max())
lowerlim = max(statsdf.KLtrueestimate.min(), statsdf.KLestimatetrue.min())
plt.plot([lowerlim, upperlim], [lowerlim, upperlim], 'k--')

statsdf.plot(
    kind='scatter', x='distperbase', y='numsamples', color=methodcolors,
    alpha=alpha, loglog=True)

statsdf.plot(
    kind='scatter', x='distperbase', y='iteration', color=methodcolors,
    alpha=alpha, logx=True)

statsdf.plot(
    kind='scatter', x='distperbase', y='lambdatrue', color=methodcolors,
    alpha=alpha)


def comparemethods(statsdf, method1, method2, distance='distperbase'):
    bymethod = statsdf.groupby('method')
    idx = npy.arange(statsdf.shape[0] / 2)
    stats1 = statsdf.loc[bymethod.groups[method1]]
    stats2 = statsdf.loc[bymethod.groups[method2]]
    stats1.index = idx
    stats2.index = idx
    dist = pds.DataFrame({
        method1: stats1[distance],
        method2: stats2[distance],
        'iteration': stats1.iteration
    })
    return dist

distance = 'KLestimatetrue'
distance = 'KLtrueestimate'
distance = 'distperbase'
method1, method2 = 'PWMweights', 'uniformweights'
dist = comparemethods(statsdf, method1, method2, distance=distance)
maxiter = dist.iteration.max()
summercm = plt.get_cmap('summer')
customcmap = [(x/24.0,  x/48.0, 0.05) for x in xrange(maxiter + 1)]
colors = [customcmap[iteration] for iteration in dist.iteration]
dist.plot(
    kind='scatter', x=method1, y=method2, color=colors,
    loglog=True, alpha=alpha)
upperlim = min(dist[method1].max(), dist[method2].max())
lowerlim = max(dist[method1].min(), dist[method2].min())
plt.plot([lowerlim, upperlim], [lowerlim, upperlim], 'k--')
plt.title(distance)
# Create a fake colorbar
ctb = LinearSegmentedColormap.from_list('custombar', customcmap, N=2048)
sm = plt.cm.ScalarMappable(
    cmap=ctb, norm=plt.Normalize(vmin=0, vmax=maxiter))
# Fake up the array of the scalar mappable
sm._A = []
# Set colorbar, aspect ratio
cbar = plt.colorbar(sm, alpha=alpha, aspect=16, shrink=0.9)
cbar.solids.set_edgecolor("face")
# Remove colorbar container frame
cbar.outline.set_visible(False)
# Fontsize for colorbar ticklabels
cbar.ax.tick_params(labelsize=16)
# Customize colorbar tick labels
mytks = npy.arange(0, maxiter, 5)
cbar.set_ticks(mytks)
cbar.ax.set_yticklabels(map(str, mytks))
# Colorbar label, customize fontsize and distance to colorbar
cbar.set_label('Iterations', rotation=270, fontsize=20, labelpad=20)
# Remove color bar tick lines, while keeping the tick labels
cbarytks = plt.getp(cbar.ax.axes, 'yticklines')
plt.setp(cbarytks, visible=False)
