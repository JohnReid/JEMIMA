#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import jemima
reload(jemima)
import jemima.importancesampling
reload(jemima.importancesampling)
import jemima.wmers
reload(jemima.wmers)
import jemima.evaluation
reload(jemima.evaluation)

import pandas as pds
import matplotlib.pyplot as plt
import numpy as npy
import numpy.random as rdm
import argparse


parser = argparse.ArgumentParser(
    description='Evaluate importance sampling methods.')
args = parser.parse_args(args=[])


rdm.seed(1)
args.fastas = ['T00759-small.fa']
args.Ws = (6, 8, 11, 14)
args.numseeds = 100
args.maxiters = 15
args.pseudocount = 1.
args.stopthreshold = 1e-3  # Stopping threshold (distance per base)
args.methods = [
    'PWMweights',
    'uniformweights',
]
args.writelogos = False


#
# Initialise parallel
#
from IPython.parallel import Client
rc = Client()
lview = rc.load_balanced_view()
lview.block = True


@lview.parallel()
def doseed(seedidx):
    import jemima.evaluation
    return jemima.evaluation.doseed(seedidx, args)


#
# Pass work to engines
#
dview = rc[:]
dview.push({'args': args})
result = doseed.map(xrange(args.numseeds))

statsdf = pds.concat(map(pds.DataFrame, result))
statsdf.to_pickle('statsdf.pkl')
statsdf = pds.read_pickle('statsdf.pkl')
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
