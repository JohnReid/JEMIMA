#!/usr/bin/env python

#
# Copyright John Reid 2014
#


r"""
MEME's EM algorithm is dominated by calculations of the form

.. math::

    \Sigma_T = \sum_{n \in T} \langle Z_n \rangle

for some set :math:`T`. STEME approximates these sums by ignoring those
:math:`n` for which :math:`\langle Z_n \rangle` is small: :math:`T_\delta = \{
n : Z_n \ge \delta \}` and iterating efficiently over the :math:`n` with
large :math:`Z_n`.

Another way to estimate these sums is importance sampling. Suppose :math:`f(n)`
is a uniform distribution over :math:`T`, then

.. math::

    \Sigma_T = \sum_{n \in T} \langle Z_n \rangle
    = |T| \bigg\langle \langle Z_n \rangle \bigg\rangle_{n \sim f(.)}

where the outer expectation is with respect to :math:`f(.)` and the inner
expectation is with respect to MEME's model. Sampling :math:`n` from
:math:`f(.)` may give us the correct expectation but the variance of the
estimator will be high as many of the :math:`Z_n` are tiny.  We can do much
better using importance sampling. We sample from :math:`g(.)` which we choose
to favour :math:`n` with large :math:`Z_n`. We can correct for sampling from
the wrong distribution by reweighting by the ratio :math:`\frac{f(.)}{g(.)}`

.. math::

    \Sigma_T = |T| \bigg\langle \frac{f(n)}{g(n)} \langle Z_n \rangle \bigg\rangle_{n \sim g(.)}

All that remains is to choose a suitable :math:`g(.)`. The main difficulty with
importance sampling is the choice of this biased distribution :math:`g(.)`. We
wish to maximise the ratio of the variance of the two estimators

.. math::

    \rho = \frac{\textrm{Var}_f(\hat{\Sigma}_T)}{\textrm{Var}_g(\hat{\Sigma}_T)}

The choice of :math:`g(.)` is normally made empirically because if an
analytically optimal solution was available there would probably not be a need
for a Monte Carlo estimate of :math:`\Sigma_T`. In general, the more complex the model,
the harder it is to choose :math:`g(.)`. Fortunately, MEME has a
straightforward mixture model and we can easily design a :math:`g(.)` that
concentrates its probability mass on :math:`n` with large :math:`Z(n)`. As the variance
of independent variables shrinks as :math:`n`, :math:`\rho` is our expected speed-up
which we can estimate empirically.
"""

import logging
logger = logging.getLogger(__name__)

import time
from itertools import imap
import pandas as pd
import pandas.rpy.common as com
import seqan.traverse
import numpy as npy
import numpy.random as rdm
from numpy import log, exp

from . import wmers
import jemima as jem

class ZnSumCb(object):
    """
    Callback to sum :math:`Zn`
    """

    def __init__(self, W):
        self.sums = npy.zeros((W, jem.SIGMA))

    def __call__(self, Xn, weightedZn):
        for w, base in enumerate(Xn):
            self.sums[w,base.ordValue] += weightedZn


class ZnCalcVisitor(object):
    """
    Visit the nodes of a SeqAn index to sum the :math:`Z_n`.
    """

    def __init__(self, W, Zncalculator, cb):
        self.W = W
        self.Zncalculator = Zncalculator
        self.cb = cb

    def __call__(self, it):
        if it.repLength >= self.W:
            # Get the word
            Xn = it.representative[:self.W]
            # Calculate Zn
            Zn = self.Zncalculator(Xn)
            # Update sums
            self.cb(Xn, Zn * it.numOccurrences)
            # Have gone deep enough in index, truncate traversal
            return False
        else:
            # Keep descending
            return True


class ImportanceSampler(object):
    """Importance sampler for :math:`X_n`."""

    def __init__(self, baselikelihoodfn, W, childWmerfreqs, Zncalculator, cb):
        self.W = W
        self.childWmerfreqs = childWmerfreqs
        self.baselikelihoodfn = baselikelihoodfn
        self.Zncalculator = Zncalculator
        self.cb = cb

    def __call__(self, it, likelihoodratio=1.):
        w = it.repLength
        if w >= self.W:
            # Get the word
            Xn = it.representative[:self.W]
            # Calculate Zn
            Zn = self.Zncalculator(Xn)
            # Callback
            self.cb(Xn, Zn, likelihoodratio)
        else:
            # Get the frequency of each base after this prefix
            freqs = self.childWmerfreqs[it.value.id]
            # Get the importance weights
            impweights = self.baselikelihoodfn(w)
            # combine with Wmer frequencies to create sampling distribution
            weights = freqs * impweights
            samplingdist = weights / weights.sum()
            #logger.info('%-10s: %s', representative, samplingdist)
            # Sample one of the bases
            sample = rdm.choice(jem.ALLBASES, p=samplingdist)
            # Descend the sample
            wentDown = it.goDown(sample)
            assert wentDown
            # Calculate the updated likelihood ratio
            likelihoodratioupdate = samplingdist[sample.ordValue] / freqs[sample.ordValue]
            # recurse
            self(it, likelihoodratio * likelihoodratioupdate)


class ISSumCb(object):
    """
    Importance sampling callback to sum :math:`Z_n`
    """

    def __init__(self, W):
        self.summer = ZnSumCb(W)

    def __call__(self, Xn, Zn, likelihoodratio):
        self.summer(Xn, Zn / likelihoodratio)


class ISMemoCbWrapper(object):
    """
    Wraps an importance sampling callback to remember each
    individual :math:`Z_n` and likelihood ratio.
    """

    def __init__(self, cb):
        self.cb = cb
        self.Zns = []
        self.lrs = []

    def __call__(self, Xn, Zn, likelihoodratio):
        self.Zns.append(Zn)
        self.lrs.append(likelihoodratio)
        self.cb(Xn, Zn, likelihoodratio)


def importancesample(index, baselikelihoodfn, W, childWmerfreqs, Zncalculator, numsamples, **kwargs):
    start = time.time()
    # Set up sampler
    iscb = ISMemoCbWrapper(ISSumCb(W))
    sampler = ImportanceSampler(baselikelihoodfn, W, childWmerfreqs, Zncalculator, iscb)
    # Sample
    for _ in xrange(numsamples):
        sampler(index.topdownhistory())
    # Create data frame
    kwargs.update({
        'Z' : iscb.Zns,
        'lr': iscb.lrs,
    })
    df = pd.DataFrame(kwargs)
    df['Zweighted'] = df['Z'] / df['lr']
    duration = time.time() - start
    logger.info('Took %.3fs to sample %d samples at a rate of %.1f samples/sec',
                duration, numsamples, numsamples / duration)
    return df, iscb


def estimatesum(weightedZ, numWmers):
    return weightedZ.sum() * numWmers / len(weightedZ)


def makesumestimator(numWmers):
    def estimator(weightedZ):
        return estimatesum(weightedZ, numWmers)
    return estimator

