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
import jemima as jem
import jemima.importancesampling as jis
from jemima import wmers

import pandas as pds
import matplotlib.pyplot as plt
import seqan.traverse
import numpy as npy
import numpy.random as rdm
import collections
import functools
import itertools
import time

import matplotlib as mpl
mpl.use('agg')


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


@memoized
def loadfasta(fasta):
    """Load sequences from fasta file."""
    logger.info('Loading sequences from: %s', fasta)
    numbases, seqs, ids = seqan.readFastaDNA5(fasta)
    logger.info('Loaded %d bases from %d sequences', numbases, len(seqs))
    return numbases, seqs, ids


@memoized
def buildindex(fasta):
    """Build an index of the sequences."""
    logger.info('Building index for: %s', fasta)
    numbases, seqs, ids = loadfasta(fasta)
    index = seqan.IndexStringDNA5SetESA(seqs)

    def numbases(base):
        it = index.topdown()
        if it.goDown(base):
            return it.numOccurrences
        else:
            return 0

    bgfreqs = npy.array(map(numbases, jem.ALLBASES), dtype=float)
    return index, bgfreqs / bgfreqs.sum()


@memoized
def countWmers(fasta):
    """Count the W-mers in the sequences."""
    logger.info('Counting W-mers for: %s', fasta)
    index, bgfreqs = buildindex(fasta)
    Wmercounts = npy.zeros((2*len(index), len(Ws)), dtype=npy.uint)
    numWmers = wmers.countWmersMulti(index.topdownhistory(), Ws, Wmercounts)
    for Widx, W in enumerate(Ws):
        logger.info('Got %d %2d-mers', numWmers[Widx], W)
    # Count how many W-mers are represented by the children
    # of each node
    childWmerfreqs = npy.zeros((2*len(index), len(Ws), jem.SIGMA))
    wmers.countWmerChildren(
        index.topdownhistory(), Ws, Wmercounts, childWmerfreqs)
    childWmerfreqs = jem.normalisearray(childWmerfreqs)
    return numWmers, Wmercounts, childWmerfreqs


SequencesData = collections.namedtuple(
    'SequencesData',
    [
        'fasta',
        'numbases',
        'seqs',
        'ids',
        'index',
        'bgfreqs',
        'numWmers',
        'Wmercounts',
        'childWmerfreqs',
    ])


@memoized
def getseqsdata(fasta):
    numbases, seqs, ids = loadfasta(fasta)
    index, bgfreqs = buildindex(fasta)
    numWmers, Wmercounts, childWmerfreqs = countWmers(fasta)
    return SequencesData(
        fasta=fasta,
        numbases=numbases,
        seqs=seqs,
        ids=ids,
        index=index,
        bgfreqs=bgfreqs,
        numWmers=numWmers,
        Wmercounts=Wmercounts,
        childWmerfreqs=childWmerfreqs,
    )


def startgenerator():
    """Generate starts from possible fasta files and motif widths."""
    while True:
        fasta = rdm.choice(fastas)
        Widx = rdm.randint(len(Ws))
        W = Ws[Widx]
        logger.info('Generating start of width %d from %s', W, fasta)
        seqsdata = getseqsdata(fasta)
        logger.info('Importance sampling using background model to find seed')
        memocb = jis.importancesample(
            seqsdata.index, W, seqsdata.childWmerfreqs[:, Widx],
            jis.UniformImportanceWeight(),
            numsamples=1, callback=jis.ISCbMemo())
        yield seqsdata, Widx, memocb.Xns[0]


def dotrueiteration(seqsdata, W, pwm, lambda_):
    """Do one true iteration of EM. I.e. iterate over all W-mers."""
    logger.debug('Calculating true Zn sums')
    summer = jis.ZnSumCb(W)
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    sumvisitor = jis.ZnCalcVisitor(W, calculateZn, summer)
    seqan.traverse.topdownhistorytraversal(
        seqsdata.index.topdownhistory(), sumvisitor)
    return summer


def pwmweightsmethod(seqsdata, pwm, lambda_, Widx):
    logger.debug('Importance sampling using PWM importance weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    return jis.importancesample(
        seqsdata.index,
        Ws[Widx],
        seqsdata.childWmerfreqs[:, Widx],
        jis.PWMImportanceWeight(pwm),
        numsamples,
        jis.ISMemoCbAdaptor(jis.ZnSumCb(W), calculateZn))


def uniformweightsmethod(seqsdata, pwm, lambda_, Widx):
    logger.debug('Importance sampling using uniform weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    return jis.importancesample(
        seqsdata.index,
        Ws[Widx],
        seqsdata.childWmerfreqs[:, Widx],
        jis.UniformImportanceWeight(),
        numsamples,
        jis.ISMemoCbAdaptor(jis.ZnSumCb(W), calculateZn))


def makedf(iscb, Zncalculator, **kwargs):
    """Make a pandas data frame containing the Zs and importance ratios
    from the callback."""
    # Create data frame
    kwargs.update({
        'Z' : map(Zncalculator, iscb.Xns),
        'ir': iscb.irs,
    })
    df = pds.DataFrame(kwargs)
    df['Zweighted'] = df['Z'] * df['ir']
    return df


#
# Generate starts
#
rdm.seed(1)
fastas = ['T00759-small.fa']
Ws = [6, 8, 11, 14]
numseeds = 3
maxiters = 3
pseudocount = 1.
stopthreshold = 1e-3  # Stopping threshold (distance per base)
methods = {
    'PWMweights'     : pwmweightsmethod,
    'uniformweights' : uniformweightsmethod,
}
stats = collections.defaultdict(list)
for seedidx, (seqsdata, Widx, seed) in enumerate(
        itertools.islice(startgenerator(), numseeds)):
    W = Ws[Widx]
    numseqs = len(seqsdata.seqs)
    numWmers = seqsdata.numWmers[Widx]
    numseedsites = rdm.randint(max(1, numseqs / 10), numseqs * 2)
    Zscale = float(numWmers) / numsamples
    sumestimator = jis.makesumestimator(numWmers)
    lambda_ = numseqs / float(numWmers)
    pwm = jem.pwmfromWmer(seed, numseedsites, pseudocount)
    jem.logo(pwm, 'seed-%03d' % seedidx)

    for iteration in xrange(maxiters):
        numsamples = rdm.randint(max(1, numWmers / 10), numWmers * 2)
        pwmIC = jem.informationcontent(pwm, seqsdata.bgfreqs)
        summer = dotrueiteration(seqsdata, W, pwm, lambda_)
        logger.debug('Sums:\n%s', summer.sums)
        Znsumtrue = summer.sums[0].sum()
        pwmtrue = jem.normalisearray(summer.sums)
        pwmtrueIC = jem.informationcontent(pwmtrue, seqsdata.bgfreqs)
        lambdatrue = Znsumtrue / float(numWmers)
        jem.logo(
            pwmtrue,
            'seed-%03d-%03d-true' % (seedidx, iteration))
        distperbase = npy.linalg.norm(pwmtrue - pwm, ord=1) / W
        logging.info(
            'Iteration: %3d, IC/base=%.2f bits, PWM distance/base=%.4f',
            iteration, pwmtrueIC/W,
            npy.linalg.norm(pwmtrue - pwm, ord=1) / W)

        for methodname, method in methods.iteritems():
            start = time.time()
            iscb = method(seqsdata, pwm, lambda_, Widx)
            duration = time.time() - start
            pwmestimate = jem.normalisearray(iscb.cb.sums)
            Znsumestimate = iscb.cb.sums[0].sum() * Zscale
            stats['fasta'].append(seqsdata.fasta)
            stats['seed'].append(str(seed))
            stats['W'].append(W)
            stats['iteration'].append(iteration)
            stats['numsamples'].append(numsamples)
            stats['method'].append(methodname)
            stats['duration'].append(duration)
            stats['ICstart'].append(pwmIC)
            stats['ICtrue'].append(pwmtrueIC)
            stats['ICestimate'].append(
                jem.informationcontent(pwmestimate, seqsdata.bgfreqs))
            stats['Znsumtrue'].append(Znsumtrue)
            stats['Znsumestimate'].append(Znsumestimate)
            stats['var'].append(iscb.var())
            stats['lambdastart'].append(lambda_)
            stats['lambdatrue'].append(lambdatrue)
            stats['lambdaestimate'].append(
                Znsumestimate / float(numWmers))
            stats['distperbase'].append(
                npy.linalg.norm(pwmtrue - pwmestimate, ord=1) / W)
            stats['KLtrueestimate'].append(
                jem.pwmKL(pwmtrue, pwmestimate))
            stats['KLestimatetrue'].append(
                jem.pwmKL(pwmestimate, pwmtrue))
            jem.logo(
                pwmestimate,
                'seed-%03d-%03d-%s' % (seedidx, iteration, methodname))

        pwm = pwmtrue
        lambda_ = lambdatrue

        if distperbase < stopthreshold:
            break


statsdf = pds.DataFrame(stats)
statsdf.to_pickle('statsdf.pkl')
statsdf = pds.read_pickle('statsdf.pkl')

plt.close('all')
fig = plt.figure()
statsdf.plot(kind='scatter', x='KLtrueestimate', y='KLestimatetrue')
ax = fig.gca()
lim = min(statsdf.KLtrueestimate.max(), statsdf.KLestimatetrue.max())
plt.plot([0, lim], [0, lim], 'k--')

