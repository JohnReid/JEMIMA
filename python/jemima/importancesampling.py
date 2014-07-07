#!/usr/bin/env python

#
# Copyright John Reid 2014
#


r"""
MEME's EM algorithm is dominated by calculations of the form

.. math::

    S_T = \sum_{n \in T} \langle Z_n \rangle

for some set :math:`T`. STEME approximates these sums by ignoring those
:math:`n` for which :math:`\langle Z_n \rangle` is small:

.. math::

    S_T = \sum_{n \in T} \langle Z_n \rangle
        \approx \sum_{n \in T_\delta} \langle Z_n \rangle

where :math:`T_\delta = \{n : Z_n \ge \delta \}`. STEME iterates efficiently
over the :math:`n` with large :math:`Z_n`.

Another way to estimate these sums is sampling. Suppose :math:`f(n)`
is a uniform distribution over :math:`T`, then

.. math::

    S_T = \sum_{n \in T} \langle Z_n \rangle
        = |T| \bigg\langle \langle Z_n \rangle \bigg\rangle_{n \sim f(.)}
        \approx \frac{|T|}{K} \sum_{k=1}^K \langle Z_{n_k} \rangle
        = \hat{S}^f_T

where :math:`n_k \sim f(.)`. Sampling :math:`n` from :math:`f(.)` may give us
the correct expectation but the variance of the estimator will be high as many
of the :math:`Z_n` are tiny.  We can do much better using importance sampling.
We sample from a distribution :math:`g(.)` which we choose to favour :math:`n`
with large :math:`Z_n`. We can correct for sampling from the wrong distribution
by reweighting with the ratio :math:`\frac{f(.)}{g(.)}`

.. math::

    S_T =
        |T|
        \bigg\langle
            \frac{f(n)}{g(n)} \langle Z_n \rangle
        \bigg\rangle_{n \sim g(.)}
        \approx \frac{|T|}{K}
            \sum_{k=1}^K \frac{f(n_k)}{g(n_k)} \langle Z_{n_k} \rangle
        = \hat{S}^g_T

All that remains is to choose a suitable :math:`g(.)` and :math:`K`. The main
difficulty with importance sampling is the choice of this biased distribution
:math:`g(.)`. We wish to maximise the ratio of the variance of the two
estimators

.. math::

    \rho =
        \frac{\textrm{Var}(\hat{S}^f_T)}{\textrm{Var}(\hat{S}^g_T)}

The choice of :math:`g(.)` is normally made empirically because if an
analytically optimal solution was available there would probably not be a need
for a Monte Carlo estimate of :math:`S_T`. In general, the more complex the
model, the harder it is to choose :math:`g(.)`. Fortunately, MEME is a
straightforward mixture model and we can easily design a :math:`g(.)` that
concentrates its probability mass on :math:`n` with large :math:`Z(n)`.
:math:`\rho` is the expected speed-up we gain by sampling from :math:`g(.)`
instead of :math:`f(.)`. We can estimate :math:`\rho` empirically.  """


import logging
logger = logging.getLogger(__name__)

import time
import pandas as pd
import numpy as npy
import numpy.random as rdm

import jemima as jem


class ZnSumCb(object):
    """
    Callback to sum :math:`Zn`
    """

    def __init__(self, W):
        self.sums = npy.zeros((W, jem.SIGMA))

    def __call__(self, Xn, weightedZn):
        for w, base in enumerate(Xn):
            self.sums[w, base.ordValue] += weightedZn


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

    r"""Importance sampler for :math:`\langle Z_n \rangle`.
    This sampler descends a suffix tree (or other such index) to
    depth :math:`W`.

    At each node referenced by the iterator :math:`i`, the sampler chooses
    an edge to descend. An edge starting with base :math:`b` is descended
    with probability

    .. math::

        p_{i,b} \propto \phi_{i,b} \psi_{i,b}

    where :math:`\phi_{i,b}` is the frequency with which base :math:`b` follows
    the given node and :math:`\psi_{i,b}` is an importance weight.

    The sampling is recursive and returns an iterator to a node at least depth
    :math:`W` in the index together with the product of
    :math:`\frac{\phi_{i,b}}{p_{i,b}}` for each :math:`i` and :math:`b` that
    were sampled. This product corresponds to the :math:`\frac{f(n_k)}{g(n_k)}`
    in the equations above.
    """

    def __init__(self, W, phi, psi):
        self.W = W
        self.phi = phi
        self.psi = psi

    def __call__(self, it, lr=1.):
        w = it.repLength
        if w >= self.W:
            # Get the word
            return it, lr
        else:
            # Get the frequency of each base after this prefix
            phi = self.phi[it.value.id]
            # Get the importance weights
            psi = self.psi(it)
            # combine with Wmer frequencies to create sampling distribution
            p = phi * psi
            p /= p.sum()
            # logger.info('%-10s: %s', representative, samplingdist)
            # Sample one of the bases
            sample = rdm.choice(jem.ALLBASES, p=p)
            # Descend the sample
            wentDown = it.goDown(sample)
            assert wentDown
            # Calculate the updated likelihood ratio
            likelihoodratioupdate = phi[sample.ordValue] / p[sample.ordValue]
            # recurse
            return self(it, lr * likelihoodratioupdate)


class VarianceOnline(object):
    """Calculate variance in an online fashion. From
    http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
    """

    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n = self.n + 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        self.M2 = self.M2 + delta*(x - self.mean)

    def calculate(self):
        if self.n < 2:
            return 0
        return self.M2 / (self.n - 1)


class ISSumCb(object):
    """
    Importance sampling callback to sum :math:`Z_n`
    """

    def __init__(self, W):
        self.summer = ZnSumCb(W)

    def __call__(self, Xn, Zn, likelihoodratio):
        self.summer(Xn, Zn * likelihoodratio)


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


def importancesample(
        index, W, phi, psi,
        Zncalculator, numsamples, **kwargs):
    start = time.time()
    # Set up sampler
    iscb = ISMemoCbWrapper(ISSumCb(W))
    sampler = ImportanceSampler(W, phi, psi)
    # Sample
    for _ in xrange(numsamples):
        it, lr = sampler(index.topdownhistory())
        Xn = it.representative[:W]
        # Calculate Zn
        Zn = Zncalculator(Xn)
        # Callback
        iscb(Xn, Zn, lr)
    # Create data frame
    kwargs.update({
        'Z' : iscb.Zns,
        'lr': iscb.lrs,
    })
    df = pd.DataFrame(kwargs)
    df['Zweighted'] = df['Z'] * df['lr']
    duration = time.time() - start
    logger.info(
        'Took %.3fs to sample %d samples at a rate of %.1f samples/sec',
        duration, numsamples, numsamples / duration)
    return df, iscb


def estimatesum(weightedZ, numWmers):
    return weightedZ.sum() * numWmers / len(weightedZ)


def makesumestimator(numWmers):
    def estimator(weightedZ):
        return estimatesum(weightedZ, numWmers)
    return estimator
