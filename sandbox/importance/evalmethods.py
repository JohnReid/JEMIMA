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
import jemima.importancesampling
import jemima.wmers
import jemima.evaluation

import pandas as pds
import numpy as npy
import numpy.random as rdm
import argparse
import functools


#
# Parse arguments
#
parser = argparse.ArgumentParser(
    description='Evaluate importance sampling methods.')
parser.add_argument(
    '--numseeds', metavar='NUMSEEDS', type=int, default=48,
    help='How many seeds to evaluate')
parser.add_argument(
    '--maxiters', metavar='MAXITERS', type=int, default=3,
    help='Maximum number of EM iterations for each seed')
args = parser.parse_args()
args.rngseed = 1
args.fastas = ['T00759-small.fa']
args.Ws = (6, 8, 11, 14)
# args.numseeds = 40
# args.maxiters = 2
args.pseudocount = 1.
args.stopthreshold = 1e-3  # Stopping threshold (distance per base)
args.methods = [
    'PWMweights',
    'uniformweights',
]
args.writelogos = False
args.parallel = True
logger.info('Evaluating on %d seeds', args.numseeds)
logger.info('Maximum EM iterations: %d', args.maxiters)


#
# Choose whether to run parallel or not
#
if args.parallel:
    #
    # Initialise parallel
    #
    from IPython.parallel import Client
    rc = Client()
    dview = rc[:]
    lview = rc.load_balanced_view()
    lview.block = True

    #
    # Initialise RNGs on each engine
    #
    logger.info('Initialising engine RNGs.')
    def initrng(rngseed):
        import numpy.random
        numpy.random.seed(rngseed)
    for engineidx, engine in enumerate(rc):
        engine.apply(initrng, args.rngseed + engineidx)

    @lview.parallel()
    def doseed(seedidx):
        import jemima.evaluation
        return jemima.evaluation.doseed(seedidx, args)

    #
    # Pass work to engines
    #
    logger.info('Passing work to engines.')
    dview.push({'args': args})
    result = doseed.map(xrange(args.numseeds))

else:
    #
    # Running locally
    #
    rdm.seed(args.rngseed)
    logger.info('Executing work locally.')
    result = map(
        functools.partial(jemima.evaluation.doseed, args=args),
        xrange(args.numseeds))

#
# Combine results
#
statsdf = pds.concat(map(pds.DataFrame, result))

#
# Save results
#
picklefilename = 'statsdf-%05d-%04d.pkl' % (args.numseeds, args.maxiters)
logger.info('Saving results to %s', picklefilename)
statsdf.to_pickle(picklefilename)
