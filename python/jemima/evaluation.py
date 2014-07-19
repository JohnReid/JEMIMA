#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import logging
logger = logging.getLogger(__name__)

import pandas as pds
import numpy as npy
import numpy.random as rdm

import jemima as jem
import jemima.importancesampling as jis
from jemima import wmers
import seqan.traverse
import collections
import functools
import time


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
def countWmers(fasta, Ws):
    """Count the W-mers in the sequences."""
    logger.info('Counting W-mers for: %s', fasta)
    index, bgfreqs = buildindex(fasta)
    Wmercounts = npy.zeros((2*len(index), len(Ws)), dtype=npy.uint)
    numWmers = wmers.countWmersMulti(
        index.topdownhistory(), Ws, Wmercounts)
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
def getseqsdata(fasta, Ws):
    numbases, seqs, ids = loadfasta(fasta)
    index, bgfreqs = buildindex(fasta)
    numWmers, Wmercounts, childWmerfreqs = countWmers(fasta, Ws)
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


def generateseed(args):
    """Generate seed from possible fasta files and motif widths."""
    fasta = rdm.choice(args.fastas)
    Widx = rdm.randint(len(args.Ws))
    W = args.Ws[Widx]
    logger.info('Generating seed of width %d from %s', W, fasta)
    seqsdata = getseqsdata(fasta, args.Ws)
    logger.info('Importance sampling using background model to find seed')
    memocb = jis.importancesample(
        seqsdata.index,
        args.Ws[Widx],
        jis.WeightedSamplingDist(
            jis.DistForFreqs(seqsdata.childWmerfreqs[:, Widx]),
            jis.UniformImportanceWeight()),
        numsamples=1,
        callback=jis.ISCbMemo())
    return seqsdata, Widx, memocb.Xns[0]


def dotrueiteration(seqsdata, W, pwm, lambda_):
    """Do one true iteration of EM. I.e. iterate over all W-mers."""
    logger.debug('Calculating true Zn sums')
    summer = jis.ZnSumCb(W)
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    sumvisitor = jis.ZnCalcVisitor(W, calculateZn, summer)
    seqan.traverse.topdownhistorytraversal(
        seqsdata.index.topdownhistory(), sumvisitor)
    return summer


def pwmweightsmethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug('Importance sampling using PWM importance weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    numpositive = numsamples / 2  # Sample half in each orientation
    cb = jis.importancesample(
        seqsdata.index,
        args.Ws[Widx],
        jis.WeightedSamplingDist(
            jis.DistForFreqs(seqsdata.childWmerfreqs[:, Widx]),
            jis.PWMImportanceWeight(pwm)),
        numpositive,
        jis.ISMemoCbAdaptor(jis.ZnSumCb(args.Ws[Widx]), calculateZn))
    return jis.importancesample(
        seqsdata.index,
        args.Ws[Widx],
        jis.WeightedSamplingDist(
            jis.DistForFreqs(seqsdata.childWmerfreqs[:, Widx]),
            jis.PWMImportanceWeight(jem.pwmrevcomp(pwm))),
        numsamples - numpositive,
        cb)


def uniformweightsmethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug('Importance sampling using uniform weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    return jis.importancesample(
        seqsdata.index,
        args.Ws[Widx],
        jis.WeightedSamplingDist(
            jis.DistForFreqs(seqsdata.childWmerfreqs[:, Widx]),
            jis.UniformImportanceWeight()),
        numsamples,
        jis.ISMemoCbAdaptor(jis.ZnSumCb(args.Ws[Widx]), calculateZn))


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


def handleseed(seedidx, seqsdata, Widx, seed, args):
    """Test the methods on one seed."""
    stats = collections.defaultdict(list)
    W = args.Ws[Widx]
    numseqs = len(seqsdata.seqs)
    numWmers = seqsdata.numWmers[Widx]
    numseedsites = rdm.randint(max(1, numseqs / 10), numseqs * 2)
    lambda_ = numseqs / float(numWmers)
    pwm = jem.pwmfromWmer(seed, numseedsites, args.pseudocount)
    if args.writelogos:
        jem.logo(pwm, 'seed-%03d' % seedidx)

    for iteration in xrange(args.maxiters):
        numsamples = rdm.randint(max(1, numWmers / 10), numWmers / 2)
        Zscale = float(numWmers) / numsamples
        pwmIC = jem.informationcontent(pwm, seqsdata.bgfreqs)
        summer = dotrueiteration(seqsdata, W, pwm, lambda_)
        logger.debug('Sums:\n%s', summer.sums)
        Znsumtrue = summer.sums[0].sum()
        pwmtrue = jem.normalisearray(summer.sums)
        pwmtrueIC = jem.informationcontent(pwmtrue, seqsdata.bgfreqs)
        lambdatrue = Znsumtrue / float(numWmers)
        if args.writelogos:
            jem.logo(
                pwmtrue,
                'seed-%03d-%03d-true' % (seedidx, iteration))
        distperbase = npy.linalg.norm(pwmtrue - pwm, ord=1) / W
        logging.info(
            'Iteration: %3d, IC/base=%.2f bits, PWM distance/base=%.4f',
            iteration, pwmtrueIC/W,
            npy.linalg.norm(pwmtrue - pwm, ord=1) / W)

        for methodname in args.methods:
            start = time.time()
            iscb = METHODS[methodname](
                seqsdata, pwm, lambda_, Widx, numsamples, args)
            duration = time.time() - start
            pwmestimate = jem.normalisearray(iscb.cb.sums)
            Znsumestimate = iscb.cb.sums[0].sum() * Zscale
            stats['fasta'].append(seqsdata.fasta)
            stats['seed'].append(str(seed))
            stats['seedidx'].append(seedidx)
            stats['numseedsites'].append(numseedsites)
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
            if args.writelogos:
                jem.logo(
                    pwmestimate,
                    'seed-%03d-%03d-%s' % (seedidx, iteration, methodname))

        pwm = pwmtrue
        # lambda_ = lambdatrue

        if distperbase < args.stopthreshold:
            break
    return stats


def doseed(seedidx, args):
    seqsdata, Widx, seed = generateseed(args)
    return handleseed(seedidx, seqsdata, Widx, seed, args)


METHODS = {
    'PWMweights'     : pwmweightsmethod,
    'uniformweights' : uniformweightsmethod,
}
