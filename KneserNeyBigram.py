from __future__ import print_function, unicode_literals, division
from collections import defaultdict, Counter
from nltk import compat
from nltk import probability

@compat.python_2_unicode_compatible
class KneserNeyBi(probability.ProbDistI):
    """
    Kneser-Ney estimate of a probability distribution. This is a version of
    back-off that counts how likely an n-gram is provided the n-1-gram had
    been seen in training. Extends the ProbDistI interface, requires a bigram
    FreqDist instance to train on. Optionally, a different from default discount
    value can be specified. The default discount is set to 0.75.


    """
    def __init__(self, freqdist, bins=None, discount=0.75):
        """
        :param freqdist: The trigram frequency distribution upon which to base
            the estimation
        :type freqdist: FreqDist
        :param bins: Included for compatibility with nltk.tag.hmm
        :type bins: int or float
        :param discount: The discount applied when retrieving counts of
            bigrams
        :type discount: float (preferred, but can be set to int)
        """

        if not bins:
            self._bins = freqdist.B()
        else:
            self._bins = bins
        self._D = discount

        # cache for probability calculation
        self._cacheBi = {}
        self._cacheUni = {}

        # internal uni and bigram frequency distributions
        self._unigrams = defaultdict(int)
        self._bigrams = freqdist

        # helper dictionaries used to calculate probabilities
        self._wordtypes_after = defaultdict(float)
        self._wordtypes_before = defaultdict(float)
        for w0, w1 in freqdist:
            self._unigrams[(w0)] += freqdist[(w0, w1)]
            self._wordtypes_after[(w0)] += 1
            self._wordtypes_before[(w1)] += 1



    def probUni(self, unigram):
        if unigram in self._cacheUni:
            return self._cacheUni[unigram]
        self._cacheUni[unigram] = self._wordtypes_before[unigram]/ len(self._bigrams)
        return self._cacheUni[unigram]



    def prob(self, bigram):
        # sample must be a double
        if len(bigram) != 2:
            raise ValueError('Expected an iterable with 2 members.')
        bigram = tuple(bigram)
        w0, w1 = bigram

        if bigram in self._cacheBi:
            return self._cacheBi[bigram]
        else:

            prob = (((self._bigrams[bigram] - self.discount()) if (self._bigrams[bigram] - self.discount()) > 0 else 0 ) +
                    (self.discount() * self._wordtypes_after[w0]) * self.probUni(w1))/self._unigrams[(w0)]

        self._cacheBi[bigram] = prob
        return prob

    def discount(self):
        """
        Return the value by which counts are discounted. By default set to 0.75.

        :rtype: float
        """
        return self._D

    def set_discount(self, discount):
        """
        Set the value by which counts are discounted to the value of discount.

        :param discount: the new value to discount counts by
        :type discount: float (preferred, but int possible)
        :rtype: None
        """
        self._D = discount

    def samples(self):
        return self._bigrams.keys()

    def max(self):
        return self._bigrams.max()

    def __repr__(self):
        '''
        Return a string representation of this ProbDist

        :rtype: str
        '''
        return '<KneserNeyProbDist based on {0} trigrams'.format(self._bigrams.N())
