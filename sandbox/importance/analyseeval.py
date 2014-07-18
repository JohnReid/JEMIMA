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


picklefilename = 'statsdf-00100-015.pkl'
statsdf = pds.read_pickle(picklefilename)
statsdf.methodidx = pds.Categorical(statsdf.method)
statsdf.columns

colors = ['b', 'r', 'g']
methodcolors = [colors[label] for label in statsdf.methodidx.labels]

plt.close('all')
statsdf.plot(
    kind='scatter',
    x='KLtrueestimate', y='KLestimatetrue', colors=methodcolors)
lim = npy.ceil(min(statsdf.KLtrueestimate.max(), statsdf.KLestimatetrue.max()))
plt.plot([0, lim], [0, lim], 'k--')

statsdf.plot(
    kind='scatter', x='distperbase', y='numsamples', colors=methodcolors)

statsdf.plot(
    kind='scatter', x='distperbase', y='iteration', colors=methodcolors)

statsdf.plot(
    kind='scatter', x='distperbase', y='lambdatrue', colors=methodcolors)
