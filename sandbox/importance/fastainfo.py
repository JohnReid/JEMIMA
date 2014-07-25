#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import jemima.evaluation
import pandas as pds
import argparse
import collections


#
# Parse arguments
#
parser = argparse.ArgumentParser(
    description='Calculate FASTA file metadata.')
parser.add_argument(
    '-W', metavar='W', dest="Ws", type=int, nargs='+',
    help='Motif widths to evaluate')
parser.add_argument(
    dest="fastas", metavar='FASTA', type=str, nargs='+',
    help='FASTA files to use')
args = parser.parse_args()
if not args.Ws:
    args.Ws = (6, 8, 11, 14)
else:
    # Need a tuple rather than a list to pass as argument to memoized functions
    args.Ws = tuple(args.Ws)

fastastats = collections.defaultdict(list)
for fasta in args.fastas:
    seqsdata = jemima.evaluation.getseqsdata(fasta, args.Ws)
    logger.info('Loaded %d bases from %d sequences in %s',
                seqsdata.numbases, len(seqsdata.seqs), fasta)
    for Widx, W in enumerate(args.Ws):
        logger.info(
            'Sequences have %6d occurrences of %6d unique %2d-mers',
            seqsdata.numoccs[Widx], seqsdata.numunique[Widx], W)
        fastastats['fasta'].append(jemima.evaluation.stripfastaname(fasta))
        fastastats['numbases'].append(seqsdata.numbases)
        fastastats['numseqs'].append(len(seqsdata.seqs))
        fastastats['W'].append(W)
        fastastats['numoccs'].append(seqsdata.numoccs[Widx])
        fastastats['numunique'].append(seqsdata.numunique[Widx])
        for b, base in enumerate(jemima.ALLBASES):
            fastastats[str(base)].append(seqsdata.bgfreqs[b])

#
# Save results
#
resultsfilename = 'fasta-%s.csv' % ('-'.join(map(str, args.Ws)))
logger.info('Saving statistics to %s', resultsfilename)
pds.DataFrame(fastastats).to_csv(open(resultsfilename, 'wb'), index=False)
