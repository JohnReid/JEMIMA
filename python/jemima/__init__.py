#
# Copyright John Reid 2014
#


"""
JEMIMA: a motif finder and tools based upon suffix arrays and
importance sampling.
"""


import seqan
import numpy as npy

ALLBASES = tuple(map(seqan.DNA, 'ACGT'))
SIGMA = len(ALLBASES)
LOGQUARTER = npy.log(.25)
UNIFORM0ORDER = npy.ones(SIGMA) / SIGMA


def logo(dist, tag, make_png=True, make_eps=False, write_title=True):
    "Generate a logo named with the given tag."
    import corebio.seq
    import weblogolib as W
    if tuple(map(int, W.__version__.split('.'))) < (3, 4):
        raise ValueError('weblogolib version 3.4 or higher required')
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
        open('%s.eps' % filename, 'w').write(W.eps_formatter(data, format_))
    if make_png:
        open('%s.png' % filename, 'w').write(W.png_formatter(data, format_))


def uniform0orderloglikelihood(X):
    """A likelihood function that assigns 1/4 probability to each base
    at each position."""
    return len(X) * LOGQUARTER


def createloglikelihoodforpwmfn(pwm):
    """Create a function that computes the log likelihood for the pwm."""
    logpwm = npy.log(pwm)

    def loglikelihood(X):
        return sum(logpwm[w, base.ordValue] for w, base in enumerate(X))
    return loglikelihood


def addpseudocounts(pwm, numsites, pseudocount):
    """Smooth a PWM by adding pseudo-counts."""
    return (pwm * numsites + pseudocount) / (numsites + pseudocount)


def arrayforXn(Xn, weight=1.):
    r"""Return a :math:`(W, \Sigma)` shaped array for :math:`X_n` with ones
    in the positions for the :math:`X_{n,w}` and zeros elsewhere."""
    result = npy.zeros((len(Xn), SIGMA))
    for w, base in enumerate(Xn):
        result[w, base.ordValue] = weight
    return result


def pwmfromWmer(Xn, N, pseudocount):
    r"""Create a PWM, :math:`\theta`, from N copies of the word Xn
    smoothed by the given pseudocount, :math:`b`.

    .. math::

        \theta_{w,b} = \frac{N \delta_{X_{n,w}=b} + b}{N+4b}
    """
    theta = arrayforXn(Xn, weight=N)
    theta += pseudocount
    return normalisearray(theta)


def normalisearray(pwm):
    """
    Normalise an array, that is make every slice in the last dimension
    sum to 1.

    .. doctest::

        >>> import numpy
        >>> import jemima
        >>> a = numpy.array([3, 6, 3, 0], dtype=numpy.float)
        >>> jemima.normalisearray(a)
        array([ 0.25,  0.5 ,  0.25,  0.  ])
        >>> a = numpy.array([
        ...     [2, 2, 0, 0],
        ...     [0, 0, 1, 1],
        ... ], dtype=numpy.float)
        >>> jemima.normalisearray(a)
        array([[ 0.5,  0.5,  0. ,  0. ],
               [ 0. ,  0. ,  0.5,  0.5]])
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
        >>> jemima.normalisearray(a)
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
    def baselikelihoodfn(it):
        return pwm[it.repLength]
    return baselikelihoodfn


def createpwmlikelihoodsquaredfn(pwm):
    pwmsquared = pwm ** 2

    def baselikelihoodfn(w):
        return pwmsquared[w]
    return baselikelihoodfn


def bglikelihoodfn(it):
    return UNIFORM0ORDER


def createZncalculatorFn(pwm, lambda_):
    """Create a function that calculates :math:`Z_n` from :math:`X_n`"""
    bsloglikelihoodfn = createloglikelihoodforpwmfn(pwm)
    lambdaratio = lambda_ / (1. - lambda_)

    def calculateZn(X):
        logodds = bsloglikelihoodfn(X) - uniform0orderloglikelihood(X)
        return 1./(1 + 1/(lambdaratio * npy.exp(logodds)))
    return calculateZn
