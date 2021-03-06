#
# Copyright John Reid 2014
#


r"""
When iterating over all the unique :math:`W`-mers in a text or their
occurrences, we often need to know the total number of such :math:`W`-mers
below any given vertex in a suffix tree (or array).
This module provides functionality to count unique :math:`W`-mers or their
occurrences.

Counting example
----------------

.. doctest::

    >>> import jemima.wmers
    >>> import numpy
    >>> import seqan.traverse

Build an index from some short sequences.

.. doctest::

    >>> seqs = seqan.StringDNA5Set(
    ...     ('ACGT', 'AAAAA', 'AAGG', 'AAGG', 'AC', 'AGNNCCC'))
    >>> index = seqan.IndexStringDNA5SetESA(seqs)

Count W-mers of lengths 2, 3 and 5 (we should have seventeen 2-mers, ten 3-mers
and one 5-mer).

.. doctest::

    >>> Ws = [2, 3, 5]
    >>> counts = numpy.zeros(((2*len(index)), len(Ws)), dtype=numpy.uint)
    >>> print jemima.wmers.countWmersMulti(
    ...     index.topdownhistory(), Ws, counts, countunique=False)
    [17 10  1]

Six 2-mers, five 3-mers and one 5-mer start with 'AA'

.. doctest::

    >>> it = index.topdown()
    >>> it.goDown('AA')
    True
    >>> print counts[it.value.id]
    [6 5 1]

Two 2-mers, one 3-mer and no 5-mers start with 'AC'.

.. doctest::

    >>> it = index.topdown()
    >>> it.goDown('AC')
    True
    >>> print counts[it.value.id]
    [2 1 0]


Once we have the counts of W-mers for each node, we can calculate the
frequencies of each possible subsequent base.

.. doctest::

    >>> childcounts = numpy.zeros(((2*len(index)), len(Ws), 4))
    >>> jemima.wmers.countChildren(
    ...     index.topdownhistory(), Ws, counts, childcounts)

Examine the distribution of bases in all W-mers (for W=2, 3 and 5) after the
prefix 'AA'

.. doctest::

    >>> it = index.topdown()
    >>> it.goDown('AA')
    True
    >>> print childcounts[it.value.id]
    [[ 0.  0.  0.  0.]
     [ 3.  0.  2.  0.]
     [ 1.  0.  0.  0.]]

For example, none of the 2-mers starting with 'AA' are followed by anything.
Three of the 3-mers starting with 'AA' are followed by 'A' and two by 'G'.
The only 5-mer starting with 'AA' is followed by 'A'.

"""


import bisect
from . import UNKNOWNBASE, findfirstparentunknown


def countWmersMulti(it, Ws, counts, countunique=True):
    """Count all the :math:`W`-mer occurrences (or unique W-mers)
    below the iterator for multiple widths, Ws.

    Arguments:
        - *it*: The iterator below which to count occurrences.
        - *Ws*: The widths to count for.
        - *counts*: The counts array of shape
            (2*len(index), len(Ws))
        - *countunique*: If true, count the number of unique W-mers below
            each vertex for each width. Otherwise count the number of
            occurrences.
    """
    nodecounts = counts[it.value.id]
    maxW = Ws[-1]
    firstunknown = findfirstparentunknown(it, maxW)
    # Do we have to descend any further? Is our representative as long as
    # largest W? Did we find an unknown base?
    if firstunknown == it.repLength and it.repLength < maxW:
        # Yes we should descend so go down and add up counts from child nodes
        if it.goDown():
            while True:
                nodecounts += countWmersMulti(it, Ws, counts, countunique)
                if not it.goRight():
                    break
            it.goUp()
    # Determine which Ws our representative is longer than
    longestWidx = bisect.bisect(Ws, firstunknown)
    # Determine which Ws our parent representative is longer than
    parentWidx = not it.isRoot and bisect.bisect(
        Ws[:longestWidx],
        it.repLength - it.parentEdgeLength) or 0
    # Set those counts to number of occurrences
    if countunique:
        nodecounts[parentWidx:longestWidx] = 1
    else:
        nodecounts[parentWidx:longestWidx] = it.numOccurrences
    return nodecounts


def countChildren(it, Ws, counts, childcounts):
    """
    Given a set of counts for given widths, summarise how many counts are
    beneath each node.

    Args:
        - it: A top-down history iterator.
        - W: The maximum W.
        - counts: An array of shape (2*len(index), len(Ws)) of the counts.
        - childcounts: An array of shape (2*len(index), len(Ws), 4) to store
            the child counts in.

    The childcounts array is typically initialised as::

        childcounts = numpy.zeros((2*len(index), len(Ws), 4))
    """
    # nodecounts = counts[it.value.id]
    nodechild = childcounts[it.value.id]
    # Do we have to descend any further? Is our representative as long as
    # largest W?
    if it.repLength < Ws[-1]:
        # Yes so go down and add up counts from child nodes
        if it.goDown():
            while True:
                if UNKNOWNBASE != it.parentEdgeFirstChar:
                    nodechild[:, it.parentEdgeFirstChar.ordValue] += \
                        counts[it.value.id]
                    countChildren(it, Ws, counts, childcounts)
                if not it.goRight():
                    break
            it.goUp()


def _countUnique(seqs, Ws):
    """Count unique W-mers in slow way for testing."""
    from collections import defaultdict
    unique = defaultdict(set)
    for seq in seqs:
        for W in Ws:
            for pos in xrange(len(seq) - W + 1):
                wmer = str(seq[pos:pos+W])
                assert len(wmer) == W
                if wmer.find('N') == -1:
                    unique[W].add(wmer)
    return unique


def testUnique(seqs, Ws, prefixes=('ACA',)):
    """Check we count the unique W-mers correctly by counting them in
    an independent way."""
    import seqan
    import numpy as npy
    from jemima.wmers import countWmersMulti
    index = seqan.IndexStringDNA5SetESA(seqs)
    counts = npy.zeros((2*len(index), len(Ws)), dtype=int)
    numunique = countWmersMulti(
        index.topdownhistory(), Ws, counts, countunique=True)
    unique = _countUnique(seqs, Ws)
    if (npy.array([len(unique[W]) for W in Ws]) != numunique).all():
        raise ValueError('Counts did not match.')
    for prefix in prefixes:
        it = index.topdownhistory()
        if not it.goDown(prefix):
            raise ValueError('Prefix "%s" does not exist in text' % prefix)
        for Widx, W in enumerate(Ws):
            count1 = len(filter(
                lambda wmer: wmer.startswith(prefix), unique[W]))
            count2 = counts[it.value.id, Widx]
            if count1 != count2:
                raise ValueError(
                    'Counts for "%s" did not match: %s != %s' % (
                        prefix, count1, count2))
