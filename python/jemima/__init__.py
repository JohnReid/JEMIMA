#
# Copyright John Reid 2014
#


"""
JEMIMA: a motif finder and tools based upon suffix arrays and
importance sampling.
"""


import seqan
import numpy as npy
import contextlib
import functools
import time
import logging
import collections
logger = logging.getLogger(__name__)

Base = seqan.DNA5
String = seqan.StringDNA5
UNKNOWNBASE = Base('N')
ALLBASES = tuple(map(Base, 'ACGT'))
SIGMA = len(ALLBASES)
LOGQUARTER = npy.log(.25)
UNIFORM0ORDER = npy.ones(SIGMA) / SIGMA


@contextlib.contextmanager
def loggingtimer(taskmsg, logger=None, level=logging.INFO):
    if logger is None:
        logger = logging.getLogger(__name__)
    start = time.time()
    yield
    duration = time.time() - start
    logger.log(level, 'Took %.2fs to %s', duration, taskmsg)


def logtime(taskmsg, logger=None, level=logging.INFO):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with loggingtimer(taskmsg, logger, level):
                return fn(*args, **kwargs)

        return wrapper
    return decorator


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


def parentEdgeLabelUpToW(it, W):
    """Return the parent edge label of the iterator but only
    as far as the W'th base in the representative."""
    surplus = it.repLength - W
    if surplus <= 0:
        return it.parentEdgeLabel
    else:
        return it.parentEdgeLabel[:-surplus]


def findfirstparentunknown(it, maxW=None):
    """Find the first unknown in the parent edge label of this iterator.
    The parent edge label is searched only as far as the maxW'th
    position in it.representative. If no maxW is given, it is set to
    it.repLength.
    The position of the first unknown is returned as an index into
    it.representative, not it.parentEdgeLabel.
    If no unknown is found, it.repLength is returned."""
    if maxW is None:
        maxW = it.repLength
    if it.isRoot:
        return 0
    else:
        for w, b in enumerate(parentEdgeLabelUpToW(it, maxW)):
            if UNKNOWNBASE == b:
                return it.repLength - it.parentEdgeLength + w
        else:
            return it.repLength


def informationcontent(pwm, bgfreqs):
    """The information content of a PWM in bits.

    Args:
        - *bgfreqs*: The background frequencies of each
            base :math:`b`, :math:`f_b`.
        - *pwm*: The frequencies of each base, :math:`b`
            at each position, :math:`w` in the PWM,
            :math:`\theta_{w,b}`.

    Returns the information content of a PWM:

    .. math::

        \sum_w \theta_{w,b} \log \frac{\theta_{w,b}}{f_b}
    """
    return (pwm * (npy.log2(pwm) - npy.log2(bgfreqs))).sum()


def pwmKL(pwm1, pwm2):
    """The Kullback-Leibler divergence between two PWMs in bits."""
    return (pwm1 * (npy.log2(pwm1) - npy.log2(pwm2))).sum()


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
    logpwmrevcomp = pwmrevcomp(logpwm)

    def loglikelihood(X):
        return npy.log(
            .5 * npy.exp(sum(logpwmrevcomp[w, base.ordValue]
                             for w, base in enumerate(X)))
            +
            .5 * npy.exp(sum(logpwm[w, base.ordValue]
                             for w, base in enumerate(X)))
        )
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


def pwmrevcomp(pwm):
    """Return the reverse complement of the PWM."""
    return pwm[::-1, ::-1]


def createZncalculatorFn(pwm, lambda_):
    """Create a function that calculates :math:`Z_n` from :math:`X_n`"""
    bsloglikelihoodfn = createloglikelihoodforpwmfn(pwm)
    lambdaratio = lambda_ / (1. - lambda_)

    def calculateZn(X):
        logodds = bsloglikelihoodfn(X) - uniform0orderloglikelihood(X)
        return 1./(1 + 1/(lambdaratio * npy.exp(logodds)))
    return calculateZn
