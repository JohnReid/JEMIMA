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
import time
import re


METHODS = dict()


def stripfastaname(fasta):
    """Return a stripped version of the argument with leading directories
    and anything after ".fa" or ".fasta" removed."""
    fasta = re.sub(".+/", "", fasta)
    fasta = re.sub("\\.[Ff][Aa].*", "", fasta)
    return fasta


def samplingmethod(name):
    def decorator(f):
        METHODS[name] = f
        return f
    return decorator


@jem.memoized
@jem.logtime('load FASTA')
def loadfasta(fasta):
    """Load sequences from fasta file."""
    logger.info('Loading sequences from: %s', fasta)
    numbases, seqs, ids = seqan.readFastaDNA5(fasta)
    logger.info('Loaded %d bases from %d sequences', numbases, len(seqs))
    return numbases, seqs, ids


@jem.memoized
@jem.logtime('build index')
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


def _countWmers(index, Ws, countunique):
    counts = npy.zeros((2*len(index), len(Ws)), dtype=npy.uint)
    rootcounts = wmers.countWmersMulti(
        index.topdownhistory(), Ws, counts, countunique=countunique)
    # Count how many W-mers are represented by the children
    # of each node
    childfreqs = npy.zeros((2*len(index), len(Ws), jem.SIGMA))
    wmers.countChildren(
        index.topdownhistory(), Ws, counts, childfreqs)
    childfreqs = jem.normalisearray(childfreqs)
    return rootcounts, counts, childfreqs


@jem.memoized
@jem.logtime('count W-mers')
def countWmers(fasta, Ws):
    """Count the W-mers in the sequences."""
    cachedWmersfilename = fasta + ('.Wmers-%s.npz' % '-'.join(map(str, Ws)))

    @jem.npzcache(cachedWmersfilename, "W-mer counts cache")
    def cached():
        logger.info('Counting W-mers for: %s', fasta)
        index, bgfreqs = buildindex(fasta)
        # Count W-mer occurrences
        numoccs, occcounts, childoccfreqs = \
            _countWmers(index, Ws, countunique=False)
        # Count unique W-mers
        numunique, uniquecounts, childuniquefreqs = \
            _countWmers(index, Ws, countunique=True)
        # Log how many we have for each width
        for Widx, W in enumerate(Ws):
            logger.info(
                'Got %6d occurrences of %6d unique %2d-mers',
                numoccs[Widx], numunique[Widx], W)
        result = numoccs, occcounts, childoccfreqs, \
            numunique, uniquecounts, childuniquefreqs
        return {
            'numoccs' : numoccs,
            'occcounts' : occcounts,
            'childoccfreqs' : childoccfreqs,
            'numunique' : numunique,
            'uniquecounts' : uniquecounts,
            'childuniquefreqs' : childuniquefreqs
        }
    result = cached()
    return (
        result['numoccs'],
        result['occcounts'],
        result['childoccfreqs'],
        result['numunique'],
        result['uniquecounts'],
        result['childuniquefreqs'],
    )


def _storeresult(result):
    """Test function to try various ways to store a tuple of numpy arrays.
    """
    if False:
        # Try numpy
        npy.savez_compressed('store-npy.npz', *result)
    if False:
        # Try h5py
        import h5py
        store = h5py.File("store-h5py.hdf5", "w", compression='lzf')
        store['numoccs'] = numoccs
        store['occcounts'] = occcounts
        store['childoccfreqs'] = childoccfreqs
        store['numunique'] = numunique
        store['uniquecounts'] = childuniquefreqs
        store['childuniquefreqs'] = childuniquefreqs
    if False:
        # Try PyTables
        import tables
        store = tables.open_file(
            'store-pytables.hdf5', mode="w",
            filters=tables.Filters(complib='bzip2', complevel=6))
        def storearray(name, x):
            atom = tables.Atom.from_dtype(x.dtype)
            ds = store.createCArray(store.root, name, atom, x.shape)
            ds[:] = x
        storearray('numoccs', numoccs)
        storearray('occcounts', occcounts)
        storearray('childoccfreqs', childoccfreqs)
        storearray('numunique', numunique)
        storearray('uniquecounts', childuniquefreqs)
        storearray('childuniquefreqs', childuniquefreqs)
        store.close()


SequencesData = collections.namedtuple(
    'SequencesData',
    [
        'fasta',
        'numbases',
        'seqs',
        'ids',
        'index',
        'bgfreqs',
        'numoccs',
        'occcounts',
        'childoccfreqs',
        'numunique',
        'uniquecounts',
        'childuniquefreqs',
    ])


@jem.memoized
@jem.logtime('get sequences data')
def getseqsdata(fasta, Ws):
    numbases, seqs, ids = loadfasta(fasta)
    index, bgfreqs = buildindex(fasta)
    numoccs, occcounts, childoccfreqs, \
        numunique, uniquecounts, childuniquefreqs = countWmers(fasta, Ws)
    return SequencesData(
        fasta=fasta,
        numbases=numbases,
        seqs=seqs,
        ids=ids,
        index=index,
        bgfreqs=bgfreqs,
        numoccs=numoccs,
        occcounts=occcounts,
        childoccfreqs=childoccfreqs,
        numunique=numunique,
        uniquecounts=uniquecounts,
        childuniquefreqs=childuniquefreqs,
    )


def generateseed(args):
    """Generate seed from possible fasta files and motif widths."""
    fasta = rdm.choice(args.fastas)
    Widx = rdm.randint(len(args.Ws))
    W = args.Ws[Widx]
    logger.info('Generating seed of width %d from %s', W, fasta)
    seqsdata = getseqsdata(fasta, args.Ws)
    logger.info('Importance sampling using background model to find seed')
    W = args.Ws[Widx]
    childfreqs = jis.DistForFreqs(seqsdata.childoccfreqs[:, Widx])
    memocb = jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(childfreqs, jis.UniformImportanceWeight()),
        numsamples=1,
        callback=jis.ISCbMemo())
    return seqsdata, Widx, memocb.its[0].representative[:W]


@jem.logtime('do one true iteration')
def dotrueiteration(seqsdata, W, pwm, lambda_):
    """Do one true iteration of EM. I.e. iterate over all W-mers."""
    logger.debug('Calculating true Zn sums')
    summer = jis.ZnSumCb(W)
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    sumvisitor = jis.ZnCalcVisitor(W, calculateZn, summer)
    seqan.traverse.topdownhistorytraversal(
        seqsdata.index.topdownhistory(), sumvisitor)
    return summer


@samplingmethod('PWMoccs')
def pwmoccsmethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug('Importance sampling using PWM importance weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    numpositive = numsamples / 2  # Sample half in each orientation
    W = args.Ws[Widx]
    childoccfreqs = jis.DistForFreqs(seqsdata.childoccfreqs[:, Widx])
    cb = jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childoccfreqs,
            jis.PWMImportanceWeight(pwm)),
        numpositive,
        jis.ISMemoCbAdaptor(W, jis.ZnSumCb(W), calculateZn))
    return seqsdata.numoccs[Widx], jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childoccfreqs,
            jis.PWMImportanceWeight(jem.pwmrevcomp(pwm))),
        numsamples - numpositive,
        cb)


@samplingmethod('uniformoccs')
def uniformoccsmethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug(
        'Importance sampling using uniform weights over unique W-mers')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    W = args.Ws[Widx]
    childoccfreqs = jis.DistForFreqs(seqsdata.childoccfreqs[:, Widx])
    return seqsdata.numoccs[Widx], jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childoccfreqs, jis.UniformImportanceWeight()),
        numsamples,
        jis.ISMemoCbAdaptor(W, jis.ZnSumCb(W), calculateZn))


@samplingmethod('PWMunique')
def pwmuniquemethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug('Importance sampling using PWM weights over unique W-mers')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    W = args.Ws[Widx]
    childuniquefreqs = jis.DistForFreqs(seqsdata.childuniquefreqs[:, Widx])
    childoccfreqs = jis.DistForFreqs(seqsdata.childoccfreqs[:, Widx])
    numpositive = numsamples / 2  # Sample half in each orientation
    cb = jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childuniquefreqs,
            jis.PWMImportanceWeightUnique(pwm, childoccfreqs)),
        numpositive,
        jis.ISMemoCbAdaptor(W, jis.ZnSumCb(W), calculateZn, unique=True))
    return seqsdata.numunique[Widx], jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childuniquefreqs,
            jis.PWMImportanceWeightUnique(jem.pwmrevcomp(pwm), childoccfreqs)),
        numsamples - numpositive,
        cb)


@samplingmethod('uniformunique')
def uniformuniquemethod(seqsdata, pwm, lambda_, Widx, numsamples, args):
    logger.debug('Importance sampling using uniform weights')
    calculateZn = jem.createZncalculatorFn(pwm, lambda_)
    W = args.Ws[Widx]
    childuniquefreqs = jis.DistForFreqs(seqsdata.childuniquefreqs[:, Widx])
    return seqsdata.numunique[Widx], jis.importancesample(
        seqsdata.index,
        W,
        jis.WeightedSamplingDist(
            childuniquefreqs, jis.UniformImportanceWeight()),
        numsamples,
        jis.ISMemoCbAdaptor(W, jis.ZnSumCb(W), calculateZn, unique=True))


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
    seedstats = collections.defaultdict(list)
    iterstats = collections.defaultdict(list)
    methodstats = collections.defaultdict(list)
    strippedfasta = stripfastaname(seqsdata.fasta)
    logger.info("Stripped FASTA: %s", strippedfasta)
    W = args.Ws[Widx]
    logger.info('Seed: %s; W=%2d', seed, W)
    numseqs = len(seqsdata.seqs)
    numoccs = seqsdata.numoccs[Widx]
    numunique = seqsdata.numunique[Widx]
    meannumsamples = npy.log10(numunique) * 600
    numseedsites = rdm.randint(max(1, numseqs / 10), numseqs * 2)
    lambda_ = numseqs / float(numoccs)
    pwm = jem.pwmfromWmer(seed, numseedsites, args.pseudocount)
    if args.writelogos:
        jem.logo(pwm, 'seed-%03d' % seedidx)
    seedstats['seedidx'].append(seedidx)
    seedstats['fasta'].append(strippedfasta)
    seedstats['seed'].append(str(seed))
    seedstats['W'].append(W)
    seedstats['numseedsites'].append(numseedsites)

    for iteration in xrange(args.maxiters):
        # numsamples = rdm.randint(max(1, numoccs / 10), numoccs / 2)
        numsamples = \
            int(rdm.lognormal(mean=npy.log(meannumsamples), sigma=.5)) + 1
        pwmIC = jem.informationcontent(pwm, seqsdata.bgfreqs)
        start = time.time()
        summer = dotrueiteration(seqsdata, W, pwm, lambda_)
        truetime = time.time() - start
        logger.debug('Sums:\n%s', summer.sums)
        Znsumtrue = summer.sums[0].sum()
        pwmtrue = jem.normalisearray(summer.sums)
        pwmtrueIC = jem.informationcontent(pwmtrue, seqsdata.bgfreqs)
        lambdatrue = Znsumtrue / float(numoccs)
        if args.writelogos:
            jem.logo(
                pwmtrue,
                'seed-%03d-%03d-true' % (seedidx, iteration))
        distperbase = npy.linalg.norm(pwmtrue - pwm, ord=1) / W
        logging.info(
            'Iteration: %3d, IC/base=%.2f bits, PWM distance/base=%.4f',
            iteration, pwmtrueIC/W,
            npy.linalg.norm(pwmtrue - pwm, ord=1) / W)
        iterstats['seedidx'].append(seedidx)
        iterstats['iteration'].append(iteration)
        iterstats['truetime'].append(truetime)
        iterstats['numsamples'].append(numsamples)
        iterstats['ICstart'].append(pwmIC)
        iterstats['ICtrue'].append(pwmtrueIC)
        iterstats['Znsumtrue'].append(Znsumtrue)
        iterstats['lambdastart'].append(lambda_)
        iterstats['lambdatrue'].append(lambdatrue)

        for methodname in args.methods:
            start = time.time()
            domainsize, iscb = METHODS[methodname](
                seqsdata, pwm, lambda_, Widx, numsamples, args)
            duration = time.time() - start
            pwmestimate = jem.normalisearray(iscb.cb.sums)
            Znsumestimate = iscb.cb.sums[0].sum() * \
                float(domainsize) / numsamples
            methodstats['seedidx'].append(seedidx)
            methodstats['iteration'].append(iteration)
            methodstats['method'].append(methodname)
            methodstats['methodtime'].append(duration)
            methodstats['ICestimate'].append(
                jem.informationcontent(pwmestimate, seqsdata.bgfreqs))
            methodstats['Znsumestimate'].append(Znsumestimate)
            methodstats['var'].append(iscb.var())
            methodstats['lambdaestimate'].append(
                Znsumestimate / float(numoccs))
            # Various measures of how different the estimated PWM is from the
            # true PWM.
            methodstats['frobeniusdist'].append(
                npy.linalg.norm(pwmtrue - pwmestimate, ord='fro'))
            methodstats['maxdist'].append(npy.abs(pwmtrue - pwmestimate).max())
            methodstats['absdist'].append(npy.abs(pwmtrue - pwmestimate).sum())
            methodstats['euclideandist'].append(
                npy.linalg.norm((pwmtrue - pwmestimate).flatten(), ord=2))
            methodstats['KLtrueestimate'].append(
                jem.pwmKL(pwmtrue, pwmestimate))
            methodstats['KLestimatetrue'].append(
                jem.pwmKL(pwmestimate, pwmtrue))
            if args.writelogos:
                jem.logo(
                    pwmestimate,
                    'seed-%03d-%03d-%s' % (seedidx, iteration, methodname))

        pwm = pwmtrue
        # lambda_ = lambdatrue

        if distperbase < args.stopthreshold:
            break
    return seedstats, iterstats, methodstats


def doseed(seedidx, args):
    seqsdata, Widx, seed = generateseed(args)
    return handleseed(seedidx, seqsdata, Widx, seed, args)
