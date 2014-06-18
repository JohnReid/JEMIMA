#
# Copyright John Reid 2014
#


"""
JEMIMA: a motif finder and tools based upon suffix arrays and importance sampling.
"""


import seqan
import numpy as npy

ALLBASES = tuple(map(seqan.DNA, 'ACGT'))
SIGMA = len(ALLBASES)
LOGQUARTER = npy.log(.25)
UNIFORM0ORDER = npy.ones(SIGMA) / SIGMA


def logo(dist, tag, make_png=False, make_eps=True, write_title=True):
    "Generate a logo named with the given tag."
    import weblogolib as W
    import corebio.seq
    data = W.LogoData.from_counts(corebio.seq.unambiguous_dna_alphabet, dist)
    scale = 5.4 * 4
    options = W.LogoOptions(
        logo_title=write_title and tag or None,
        stack_width=scale,
        stack_aspect_ratio=5,
        color_scheme=W.colorscheme.nucleotide,
        show_xaxis=False,
        show_yaxis=False,
        show_fineprint=False,
    )
    format_ = W.LogoFormat(data, options)
    filename = 'logo-%s' % tag
    if make_eps:
        W.eps_formatter(data, format_, open('%s.eps' % filename, 'w'))
    if make_png:
        W.png_formatter(data, format_, open('%s.png' % filename, 'w'))


def uniform0orderloglikelihood(X):
    """A likelihood function that assigns 1/4 probability to each base at each position."""
    return len(X) * LOGQUARTER


def createloglikelihoodforpwmfn(pwm):
    """Create a function that computes the log likelihood for the pwm."""
    logpwm = npy.log(pwm)
    def loglikelihood(X):
        return sum(logpwm[w,base.ordValue] for w, base in enumerate(X))
    return loglikelihood


def addpseudocounts(pwm, numsites, pseudocount):
    """Smooth a PWM by adding pseudo-counts."""
    return (pwm * numsites + pseudocount) / (numsites + pseudocount)


def normalisepwm(pwm):
    """
    Normalise a PWM, that is make its entries sum to 1 for each position.

    In general this function returns an array such that the last dimension
    always sums to 1.

    .. doctest::

        >>> import numpy
        >>> import jemima
        >>> a = numpy.array([
        ...     [
        ...         [2, 2, 0, 0],
        ...         [0, 0, 1, 1],
        ...     ],
        ...     [
        ...         [1, 1, 2, 1],
        ...         [0, 3, 3, 0],
        ...     ],
        ...     [
        ...         [2, 2, 4, 2],
        ...         [0, 2, 2, 0],
        ...     ],
        ... ], dtype=numpy.float)
        >>> jemima.normalisepwm(a)
        array([[[ 0.5,  0.5,  0. ,  0. ],
                [ 0. ,  0. ,  0.5,  0.5]],
        <BLANKLINE>
               [[ 0.2,  0.2,  0.4,  0.2],
                [ 0. ,  0.5,  0.5,  0. ]],
        <BLANKLINE>
               [[ 0.2,  0.2,  0.4,  0.2],
                [ 0. ,  0.5,  0.5,  0. ]]])
    """
    return (pwm.T / pwm.sum(axis=-1).T).T


def createpwmlikelihoodfn(pwm):
    def baselikelihoodfn(w):
        return pwm[w]
    return baselikelihoodfn


def createpwmlikelihoodsquaredfn(pwm):
    pwmsquared = pwm ** 2
    def baselikelihoodfn(w):
        return pwmsquared[w]
    return baselikelihoodfn


def bglikelihoodfn(w):
    return UNIFORM0ORDER


def createZncalculatorFn(pwm, lambda_):
    """Create a function that calculates :math:`Z_n` from :math:`X_n`"""
    bsloglikelihoodfn = createloglikelihoodforpwmfn(pwm)
    lambdaratio = lambda_ / (1. - lambda_)
    def calculateZn(X):
        logodds = bsloglikelihoodfn(X) - uniform0orderloglikelihood(X)
        return 1./(1 + 1/(lambdaratio * npy.exp(logodds)))
    return calculateZn
