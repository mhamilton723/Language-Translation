__author__ = 'Mark'

import copy
import random
from operator import itemgetter
from itertools import izip
import re
import pickle
import heapq

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Embedding(object):
    def __init__(self, file_name, words=None, embeddings=None):

        if file_name is not None:
            self.words, self.embeddings = pickle.load(open(file_name, 'rb'))
        else:
            self.words, self.embeddings = words, embeddings

        # Special tokens
        self.Token_ID = {"<UNK>": 0, "<S>": 1, "</S>": 2, "<PAD>": 3}
        self.ID_Token = {v: k for k, v in self.Token_ID.iteritems()}
        # Noramlize digits by replacing them with #
        self.DIGITS = re.compile("[0-9]", re.UNICODE)
        # Map words to indices and vice versa
        self.word_id = {w: i for (i, w) in enumerate(self.words)}
        self.id_word = dict(enumerate(self.words))
        self.word_set = set(self.words)

    def __copy__(self):
        new_emb = Embedding(None, words=np.copy(self.words), embeddings=np.copy(self.embeddings))
        new_emb.word_id = self.word_id
        new_emb.id_word = self.id_word
        return new_emb

    def case_normalizer(self, word):
        """ In case the word is not available in the vocabulary,
         we can try multiple case normalizing procedure.
         We consider the best substitute to be the one with the lowest index,
         which is equivalent to the most frequent alternative."""
        w = word
        lower = (self.word_id.get(w.lower(), 1e12), w.lower())
        upper = (self.word_id.get(w.upper(), 1e12), w.upper())
        title = (self.word_id.get(w.title(), 1e12), w.title())
        results = [lower, upper, title]
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return w
        return word

    def in_vocab_words(self, words):
        norm_words = [self.normalize(word) for word in words]
        return [word for word in norm_words if word is not None]

    def normalize(self, word):
        """ Find the closest alternative in case the word is OOV."""
        if not word in self.word_id:
            word = self.DIGITS.sub("#", word)
        if not word in self.word_id:
            word = self.case_normalizer(word)

        if not word in self.word_id:
            return None
        return word

    def word_to_embedding(self, words, log=False):
        if isinstance(words, list) or isinstance(words, set):
            single = False
        else:
            words = [words]
            single = True

        embs = []
        for word in words:
            wordn = self.normalize(word)

            if not wordn:
                if log: print("OOV word: " + repr(word))
                embs.append(None)
            else:
                word_index = self.word_id[wordn]
                embs.append(self.embeddings[word_index, :])

        if single:
            return embs[0]
        else:
            return embs

    def knn(self, word, k=5, method='cosine', return_distances=True):
        word = self.normalize(word)
        if not word:
            print("OOV word")
            return
        word_index = self.word_id[word]
        return self._nearest(word_index, k, method, return_distances)

    def _nearest(self, word_index, k, method='cosine', return_distances=True):
        """Sorts words according to their Euclidean distance.
           To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

        e = self.embeddings[word_index]
        return self.words_closest_to_point(e, k, method, return_distances)

    def words_closest_to_point(self, point, k=5, method='cosine', return_distances=True, select_method='numpy'):
        if method == 'l2':
            distances = (((self.embeddings - point) ** 2).sum(axis=1) ** 0.5)
        elif method == 'cosine':
            if np.dot(point, point) < 10**-9:
                point *= 1. / max(point)
            distances = 1 - np.dot(self.embeddings, point) / (((self.embeddings ** 2).sum(axis=1) ** .5)*(np.dot(point,point)** .5))
        else:
            raise ValueError('input correct distance name')
        if max(distances) > 2 or min(distances) < 0:
            raise ValueError('Distance error')

        if select_method == 'heappq':
            k_smallest = heapq.nsmallest(k, enumerate(distances), key=itemgetter(1))
        elif select_method == 'numpy':
            if k == 1:
                k_smallest = [np.argmin(np.array(distances))]
            else:
                k_smallest = np.argpartition(np.array(distances), k)[:k]
            k_smallest = [(i, distances[i]) for i in k_smallest]

        elif select_method == 'python':
            k_smallest = sorted(enumerate(distances), key=itemgetter(1))[:k]
        else:
            raise ValueError("input a proper method")

        neighbors = [self.id_word[index] for index, dist in k_smallest]
        distances = [dist for index, dist in k_smallest]

        if return_distances:
            return zip(neighbors, distances)
        else:
            return neighbors


    def plot_emb(self, model, show=True, save=None, alpha=.005):

        if model is not None:
            comps = model.fit_transform(self.embeddings)
        else:
            comps = self.embeddings[:, 0:2]

        plt.scatter(comps[:, 0], comps[:, 1], alpha=alpha)

        if save is not None:
            plt.savefig(save, dpi=100)
        if show:
            plt.show()


def sub_embedding(embedding, word_subset):
    if isinstance(word_subset, list):
        word_subset = set(word_subset)
    word_subset = word_subset.intersection(embedding.word_set)
    new_embedding = copy.copy(embedding)
    new_embedding.word_id = {k: v for k, v in embedding.word_id.iteritems() if k in word_subset}
    new_embedding.id_word = {v: k for k, v in new_embedding.word_id.iteritems()}
    sorted_id_word = sorted(new_embedding.id_word.iteritems())
    new_embedding.embeddings = embedding.embeddings[np.array([id for id, word in sorted_id_word])]

    # Set the word id dictionary to start at 1
    for i, (id, word) in enumerate(sorted_id_word):
        new_embedding.word_id[word] = i
    new_embedding.id_word = {v: k for k, v in new_embedding.word_id.iteritems()}

    return new_embedding


def subsample_embedding(embedding, sample_size, include=None):
    if include is not None:
        sample = set(random.sample(embedding.word_set, sample_size)) | include
    else:
        sample = set(random.sample(embedding.word_set, sample_size))
    return sub_embedding(embedding, sample)


def transform(embedding, model):
    new_embedding = copy.copy(embedding)
    new_embedding.embeddings = model.fit_transform(embedding.embeddings)
    return new_embedding
