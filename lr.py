"""
The code is based on LexRank summarizer from sumy.

"""

import math
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.summarizers._summarizer import AbstractSummarizer
from sumy._compat import Counter

STOPWORDS = set({})

try:
    import numpy
except ImportError:
    numpy = None

important = ['experiment', 'method', 'approach', 'result', 'idea', 'issue', 'concern',
             'work', 'contribution', 'evaluation', 'writing', 'novelty', 'analysis', 'revised',
             'improvement', 'algorithm', 'plagiarism', 'novel', 'study', 'rebuttal', 'revision'
             'concerns', 'revised', 'updated', 'improved', 'proposes', 'propose', 'written']


start = ['paper', 'work', 'propose', 'study', 'idea', 'approach', 'contribution']


def dependent(s1, s2):
    if s2.dependent:
        if s1.reviewNumber == s2.reviewNumber and s1.number == s2.number - 1:
            return 1
        else:
            return -1
    else:
        return 0


def has_starter(words):
    for i in words:
        if i in start:
            return True
    return False


def get_starter(sentences):
    candidates = []
    for i in range(len(sentences)):
        if has_starter(sentences[i][0]) > 0 and not sentences[i][1].dependent:
            candidates.append(i)
    if len(candidates) == 0:
        candidates = [i for i in range(len(sentences)) if has_nitpick(sentences[i][0]) == 0]
    if len(candidates) == 0:
        candidates = [i for i in range(len(sentences))]
    best = candidates[0]
    for i in candidates:
        if sentences[i][1].number < sentences[best][1].number:
            best = i
    return best


def get_distance(cluster1, cluster2, sentence2index, distance):
    rep_1 = sentence2index[
        str(cluster1.reviewNumber) + str(cluster1.number)]
    rep_2 = sentence2index[
        str(cluster2.reviewNumber) + str(cluster2.number)]
    return distance[rep_1][rep_2]


def important_count(words):
    k = 0
    for i in words:
        if i in important:
            k = k + 1
    return 0.2 * k


class Summarizer(AbstractSummarizer):

    threshold = 0.1
    epsilon = 0.1
    _stop_words = frozenset(STOPWORDS)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        sentences_words = [self._to_words_set(s) for s in document.sentences]
        sentences = [(self._to_words_set(s),s) for s in document.sentences]
        if not sentences_words:
            return []

        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        matrix = self._create_matrix(sentences, self.threshold, tf_metrics, idf_metrics)
        starter = get_starter(sentences)
        scores = self.rwr(matrix, starter)
        return self.get_sentences(document.sentences, sentences_count, scores)

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.word_list)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)
            for term, tf in sentence.items():
                metrics[term] = tf / max_tf
            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        idf_metrics = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))
        for term in important:
            if term not in idf_metrics:
                idf_metrics[term] = math.log(sentences_count/2)
        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                if row != col:
                    matrix[row, col] = self.cosine_similarity(sentence1[0], sentence2[0], tf1, tf2, idf_metrics) + \
                                       (important_count(sentence2[0]))
                else:
                    matrix[row, col] = 1
                degrees[row] += matrix[row, col]
        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1
                matrix[row][col] = matrix[row][col] / degrees[row]
        return matrix

    @staticmethod
    def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
        unique_words1 = frozenset(sentence1)
        unique_words2 = frozenset(sentence2)
        common_words = unique_words1 & unique_words2

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term]*tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t]*idf_metrics[t])**2 for t in unique_words1)
        denominator2 = sum((tf2[t]*idf_metrics[t])**2 for t in unique_words2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def rwr(matrix, starter):
        c = 0.85
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.zeros((sentences_count,))
        p_vector[starter] = 1

        r = numpy.zeros((sentences_count,))
        r[starter] = 1
        q = numpy.linalg.inv(numpy.subtract(numpy.identity(len(transposed_matrix)), numpy.multiply(c, transposed_matrix)))
        mat = (numpy.multiply(1 - c, numpy.dot(q, r))).reshape(len(transposed_matrix), )
        return mat

    def get_sentences(self, sentences, count, rating):
        x = rating.argsort()[::-1]
        output = [sentences[i] for i in x[:count]]
        return output
