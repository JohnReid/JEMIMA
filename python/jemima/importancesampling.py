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
over the :math:`n` with large :math:`langle Z_n \rangle`.

Another way to estimate these sums is sampling. Suppose :math:`f(n)`
is a uniform distribution over :math:`T`, then

.. math::

    S_T = \sum_{n \in T} \langle Z_n \rangle
        = |T| \bigg\langle \langle Z_n \rangle \bigg\rangle_{n \sim f(.)}
        \approx \frac{|T|}{K} \sum_{k=1}^K \langle Z_{n_k} \rangle
        = \hat{S}^f_T

where :math:`n_k \sim f(.)`. Sampling :math:`n` from :math:`f(.)` may give us
the correct expectation but the variance of the estimator will be high as many
of the :math:`\langle Z_n \rangle` are tiny.  We can do much better using
importance sampling.  We sample from a distribution :math:`g(.)` which we
choose to favour :math:`n` with large :math:`\langle Z_n \rangle`. We can
correct for sampling from the wrong distribution by reweighting with the
importance ratio :math:`\frac{f(.)}{g(.)}`

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
import itertools
import numpy as npy
import numpy.random as rdm

from jemima import SIGMA, ALLBASES, arrayforXn, \
    UNIFORM0ORDER, UNKNOWNBASE, parentEdgeLabelUpToW


class PWMImportanceWeight(object):
    """Standard importance weights for a PWM."""

    def __init__(self, pwm):
        self.pwm = pwm

    def __call__(self, it):
        """Return the importance weights for the correct column of the PWM."""
        return self.pwm[it.repLength]


class UniformImportanceWeight(object):
    """Uniform importance weights."""

    def setorientation(self, positive):
        pass

    def __call__(self, it):
        """Return the importance weights for the correct column of the PWM."""
        return UNIFORM0ORDER


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


class VarianceOnlineMulti(VarianceOnline):
    """Calculate variance in an online fashion for multiple elements of
    an array. From
    http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
    """

    def __init__(self, W):
        self.n = 0
        self.mean = npy.zeros((W, SIGMA))
        self.M2 = npy.zeros((W, SIGMA))

    def update(self, x):
        self.n = self.n + 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        self.M2 = self.M2 + delta*(x - self.mean)

    def calculate(self):
        if self.n < 2:
            return 0
        return self.M2 / (self.n - 1)


class ZnSumCb(object):
    """
    Callback to sum :math:`Zn`
    """

    def __init__(self, W):
        self.sums = npy.zeros((W, SIGMA))
        self._variances = VarianceOnlineMulti(W)

    def __call__(self, Xnarray):
        self.sums += Xnarray
        self._variances.update(Xnarray)

    def variances(self):
        return self._variances.calculate() * self._variances.n


class ISCbMemo(object):
    """
    Importance sampling callback to remember :math:`X_n`,
    :math:`Z_n`, and the importance ratios.
    """

    def __init__(self):
        self.Xns = []
        self.irs = []

    def __call__(self, Xn, importanceratio):
        self.Xns.append(Xn)
        self.irs.append(importanceratio)


class ISCbAdaptor(object):
    """
    Importance sampling callback to sum :math:`Z_n`
    """

    def __init__(self, cb, Zncalculator):
        self.cb = cb
        self.Zncalculator = Zncalculator

    def __call__(self, Xn, importanceratio):
        self.cb(arrayforXn(Xn, self.Zncalculator(Xn) * importanceratio))


class ISMemoCbAdaptor(ISCbAdaptor):
    """
    Wraps an importance sampling callback to remember each
    individual :math:`X_n` and importance ratio.
    """

    def __init__(self, cb, Zncalculator):
        super(ISMemoCbAdaptor, self).__init__(cb, Zncalculator)
        self.Xns = []
        self.irs = []

    def __call__(self, Xn, importanceratio):
        self.Xns.append(Xn)
        self.irs.append(importanceratio)
        super(ISMemoCbAdaptor, self).__call__(Xn, importanceratio)

    def var(self):
        """Calculate the variance of the Zn estimates."""
        return npy.var(map(
            float.__mul__,
            self.irs,
            itertools.imap(self.Zncalculator, self.Xns)))


class ZnCalcVisitor(object):
    """
    Visit the nodes of a SeqAn index to sum the :math:`Z_n`.
    """

    def __init__(self, W, Zncalculator, cb):
        self.W = W
        self.Zncalculator = Zncalculator
        self.cb = cb

    def __call__(self, it):
        # Always go further than the root
        if 0 == it.repLength:
            return True
        # Don't go any further if we have an unknown base
        if UNKNOWNBASE in parentEdgeLabelUpToW(it, self.W):
            return False
        if it.repLength >= self.W:
            # Get the word
            Xn = it.representative[:self.W]
            # Calculate Zn
            Zn = self.Zncalculator(Xn)
            # Update sums
            self.cb(arrayforXn(Xn, Zn * it.numOccurrences))
            # Have gone deep enough in index, truncate traversal
            return False
        else:
            # Keep descending
            return True


class IndexSample(object):
    r"""
    This class samples from a SeqAn index (e.g. a suffix tree or array)
    to depth :math:`W` according to a given sampling distribution.
    """

    def __init__(self, W, sampling):
        self.sampling = sampling
        self.W = W

    def __call__(self, it):
        "Descend the tree to depth :math:`W` to sample a :math:`W`-mer."
        if it.repLength >= self.W:
            # Return the iterator referring to the W-mer
            return it
        else:
            # Sample one of the bases
            sample = self.sampling(it)
            # Descend the sample
            wentDown = it.goDown(sample)
            assert wentDown
            # Recurse
            return self(it)


class DistForFreqs(object):

    def __init__(self, freqs):
        self.freqs = freqs

    def __call__(self, it):
        return self.freqs[it.value.id]


class WeightedSamplingDist(object):
    r"""A weighted importance sampling distribution."""

    def __init__(self, target, weights):
        self.target = target
        self.weights = weights

    def reset(self):
        self.ir = 1.

    def __call__(self, it):
        # The target distribution
        target = self.target(it)
        assert npy.abs(1 - target.sum()) < 1e-7  # Check is proper dist.
        # Get the importance weights to adjust the target by
        weights = self.weights(it)
        # Combine with target to create sampling distribution
        sampling = target * weights
        sampling /= sampling.sum()
        # logger.info('%-10s: %s', representative, samplingdist)
        # Sample one of the bases
        sample = rdm.choice(ALLBASES, p=sampling)
        # Update the importance ratio
        self.ir *= target[sample.ordValue] / sampling[sample.ordValue]
        # Return the sampled base
        return sample


class ImportanceSampler(object):

    r"""Importance sampler for a SeqAn index.
    This sampler chooses a single path down a suffix tree
    (or other such index) to depth :math:`W` to select a
    :math:`W`-mer according to a given target distribution.

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

    def __call__(self, it, ir=1.):
        """Descend the index to sample a W-mer.

        Args:
            - it: iterator pointing to current node
            - ir: current importance ratio

        Returns an iterator pointing to the node for the sampled W-mer
        and the importance ratio for the W-mer.
        """
        w = it.repLength
        if w >= self.W:
            # Get the word
            return it, ir
        else:
            # Get the frequency of each base after this prefix
            phi = self.phi[it.value.id]
            assert npy.abs(1 - phi.sum()) < 1e-7  # Check is proper dist.
            # Get the importance weights
            psi = self.psi(it)
            # combine with Wmer frequencies to create sampling distribution
            p = phi * psi
            p /= p.sum()
            # logger.info('%-10s: %s', representative, samplingdist)
            # Sample one of the bases
            sample = rdm.choice(ALLBASES, p=p)
            # Descend the sample
            wentDown = it.goDown(sample)
            assert wentDown
            # Calculate the updated importance ratio
            importanceratioupdate = phi[sample.ordValue] / p[sample.ordValue]
            # recurse
            return self(it, ir * importanceratioupdate)


def importancesample(index, W, sampling, numsamples, callback):
    """Importance sample from the index given:

        - *index*: The index to importance sample from.
        - *W*: The width of the :math:`X_n` to sample.
        - *sampling*: The inmportance sampling distribution.
        - *numsamples*: How many samples.
        - *callback*: A callback to pass the samples to.
    """
    # Record start time
    start = time.time()
    # Set up sampler
    sampler = IndexSample(W, sampling)
    # Sample
    for _ in xrange(numsamples):
        # Reset our importance sampling distribution
        sampling.reset()
        # Sample our W-mer
        it = sampler(index.topdownhistory())
        Xn = it.representative[:W]
        # Callback
        callback(Xn, sampling.ir)
    duration = time.time() - start
    logger.info(
        'Took %.3fs to sample %d samples at a rate of %.1f samples/sec',
        duration, numsamples, numsamples / duration)
    return callback


def estimatesum(weightedZ, numWmers):
    return weightedZ.sum() * numWmers / len(weightedZ)


def makesumestimator(numWmers):
    def estimator(weightedZ):
        return estimatesum(weightedZ, numWmers)
    return estimator
