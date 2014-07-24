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
parser.add_argument(
    '--fasta', '-f', dest="fastas", metavar='FASTA', type=str, nargs='+',
    help='FASTA files to use')
parser.add_argument(
    '--method', '-m', dest="methods", metavar='METHOD', type=str, nargs='+',
    help='Methods to evaluate')
parser.add_argument(
    '-W', metavar='W', type=int, nargs='+', help='Motif widths to evaluate')
args = parser.parse_args()
# args = parser.parse_args(['--maxiters=1', '--numseeds=1'])
args.rngseed = 1
if not args.fastas:
    args.fastas = ['T00759-small.fa']
if not args.methods:
    args.methods = jemima.evaluation.METHODS.keys()
if not args.Ws:
    args.Ws = (6, 8, 11, 14)
else:
    # Need a tuple rather than a list to pass as argument to memoized functions
    args.Ws = tuple(args.Ws)
args.pseudocount = 1.
args.stopthreshold = 1e-3  # Stopping threshold (distance per base)
logger.info(
    'Evaluating %d methods on %d seeds (up to %d iterations) '
    'over %d FASTA files',
    len(args.methods), args.numseeds, args.maxiters, len(args.fastas))
for fasta in args.fastas:
    logger.info('Using FASTA: %s', fasta)


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
