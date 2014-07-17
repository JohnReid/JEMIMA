#
# Copyright John Reid 2014
#


r"""
In general when finding motifs we need to consider several different motif
widths, W.  When descending a suffix tree or array we typically need to know
how many W-mers for a given W exist are descendants of the current iterator.
The code in this module counts these W-mers.

Counting example
----------------

.. doctest::

    >>> import jemima.wmers
    >>> import numpy
    >>> import seqan.traverse

Build an index from some short sequences.

.. doctest::

    >>> seqs = seqan.StringDNA5Set(('ACGT', 'AAAAA', 'AAGG', 'AAGG', 'AC', 'AGNNCCC'))
    >>> index = seqan.IndexStringDNA5SetESA(seqs)

Count W-mers of lengths 2, 3 and 5 (we should have seventeen 2-mers, ten 3-mers
and one 5-mer).

.. doctest::

    >>> Ws = [2, 3, 5]
    >>> counts = numpy.zeros(((2*len(index)), len(Ws)), dtype=numpy.uint)
    >>> print jemima.wmers.countWmersMulti(index.topdownhistory(), Ws, counts)
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
    >>> jemima.wmers.countWmerChildren(
    ...     index.topdownhistory(), Ws, counts, childcounts)

Examine the distribution of bases in all W-mers (for W=2, 3 and 5) after the
prefix 'AA'

.. doctest::

    >>> it = index.topdown()
    >>> it.goDown('AA')
    True
    >>> print childcounts[it.value.id]
    [[ 3.  0.  2.  0.]
     [ 3.  0.  2.  0.]
     [ 1.  0.  0.  0.]]

For example, three of the 2-mers starting with 'AA' are followed by 'A'.  Two
of the 2-mers starting with 'AA' are followed by 'G'.

"""


import bisect
from . import UNKNOWNBASE, parentEdgeLabelUpToW, findfirstparentunknown


def countWmers(it, W, counts):
    """
    Count how many W-mers are descendants of this iterator.

    Args:
        - it: A top-down history iterator.
        - W: The length of the W-mers we are counting.
        - counts: An array to place the counts in.

    The counts array is typically initialised as::

        counts = [0] * (2 * len(index))

    and is indexed by::

        counts[it.value.id]
    """
    if it.repLength >= W:
        count = it.numOccurrences
    else:
        count = 0
        if it.goDown():
            while True:
                if UNKNOWNBASE not in parentEdgeLabelUpToW(it, W):
                    count += countWmers(it, W, counts)
                if not it.goRight():
                    break
            it.goUp()
    counts[it.value.id] = count
    return count


def countWmersMulti(it, Ws, counts):
    """Count all the W-mers below the iterator for the given
    widths, Ws."""
    nodecounts = counts[it.value.id]
    maxW = Ws[-1]
    firstunknown = findfirstparentunknown(it, maxW)
    # Do we have to descend any further? Is our representative as long as
    # largest W? Did we find an unknown base?
    if firstunknown == it.repLength and it.repLength < maxW:
        # Yes we should descend so go down and add up counts from child nodes
        if it.goDown():
            while True:
                nodecounts += countWmersMulti(it, Ws, counts)
                if not it.goRight():
                    break
            it.goUp()
    # Determine which Ws our representative is longer than
    longestWidx = bisect.bisect(Ws, firstunknown)
    # Set those counts to number of occurrences
    nodecounts[:longestWidx] = it.numOccurrences
    # print it.representative, longestWidx, it.numOccurrences, firstunknown, nodecounts
    return nodecounts


def countWmerChildren(it, Ws, counts, childcounts):
    """
    Count how many Wmers are descendants of each child of each node.

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
                    countWmerChildren(it, Ws, counts, childcounts)
                if not it.goRight():
                    break
            it.goUp()
