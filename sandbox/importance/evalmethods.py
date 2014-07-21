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
import numpy.random as rdm
import argparse
import functools
import gzip


#
# Parse arguments
#
parser = argparse.ArgumentParser(
    description='Evaluate importance sampling methods.')
parser.add_argument(
    "--write-logos", dest="writelogos",
    help="Write logos showing learnt motifs", action="store_true")
parser.add_argument(
    "--parallel", help="Evaluate in parallel", action="store_true")
parser.add_argument(
    '--numseeds', metavar='NUMSEEDS', type=int, default=48,
    help='How many seeds to evaluate')
parser.add_argument(
    '--maxiters', metavar='MAXITERS', type=int, default=3,
    help='Maximum number of EM iterations for each seed')
# args = parser.parse_args()
args = parser.parse_args(['--maxiters=1', '--numseeds=1'])
args.rngseed = 1
args.fastas = ['T00759-small.fa']
args.Ws = (6, 8, 11, 14)
# args.numseeds = 40
# args.maxiters = 2
args.pseudocount = 1.
args.stopthreshold = 1e-3  # Stopping threshold (distance per base)
args.methods = [
    'PWMoccs',
    'uniformoccs',
    'PWMunique',
    'uniformunique',
]
# args.writelogos = False
# args.parallel = True
logger.info(
    'Evaluating %d methods on %d seeds (up to %d iterations)',
    len(args.methods), args.numseeds, args.maxiters)


#
# Choose whether to run parallel or not
#
if args.parallel:
    logger.info('Evaluating in parallel')
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
    def initrng(rngseed):
        import numpy.random
        numpy.random.seed(rngseed)

    logger.info('Initialising engine RNGs.')
    for engineidx, engine in enumerate(rc):
        engine.apply(initrng, args.rngseed + engineidx)

    #
    # Pass work to engines
    #
    @lview.parallel()
    def doseed(seedidx):
        import jemima.evaluation
        return jemima.evaluation.doseed(seedidx, args)

    logger.info('Passing work to engines.')
    dview.push({'args': args})
    result = doseed.map(xrange(args.numseeds))

else:
    #
    # Running locally
    #
    logger.info('Evaluating locally.')
    rdm.seed(args.rngseed)
    result = map(
        functools.partial(jemima.evaluation.doseed, args=args),
        xrange(args.numseeds))

#
# Combine results
#
statsdf = pds.concat(map(pds.DataFrame, result), ignore_index=True)

#
# Save results
#
resultsfilename = 'statsdf-%05d-%04d.csv.gz' % (args.numseeds, args.maxiters)
logger.info('Saving results to %s', resultsfilename)
statsdf.to_csv(gzip.open(resultsfilename, 'wb'))
