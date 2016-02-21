# coding=utf-8
__author__ = 'Mark'

import pickle
from operator import itemgetter
from itertools import izip, islice
import re
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, SparsePCA, DictionaryLearning
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    def normalize(self, word):
        """ Find the closest alternative in case the word is OOV."""
        if not word in self.word_id:
            word = self.DIGITS.sub("#", word)
        if not word in self.word_id:
            word = self.case_normalizer(word)

        if not word in self.word_id:
            return None
        return word

    def l2_nearest(self, word_index, k):
        """Sorts words according to their Euclidean distance.
           To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

        e = self.embeddings[word_index]
        distances = (((self.embeddings - e) ** 2).sum(axis=1) ** 0.5)
        sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
        return zip(*sorted_distances[:k])

    def word_to_embedding(self, word):
        wordn = self.normalize(word)

        if not wordn:
            print("OOV word: " + repr(word))
            return
        word_index = self.word_id[wordn]
        return self.embeddings[word_index, :]

    def knn(self, word, k=5):
        word = self.normalize(word)
        if not word:
            print("OOV word")
            return
        word_index = self.word_id[word]
        indices, distances = self.l2_nearest(self.embeddings, word_index, k)
        neighbors = [self.id_word[idx] for idx in indices]
        for i, (word, distance) in enumerate(izip(neighbors, distances)):
            print i, '\t', word, '\t\t', distance

    def words_closest_to_point(self, point, k=5):
        distances = (((self.embeddings - point) ** 2).sum(axis=1) ** 0.5)
        sorted_distances = sorted(enumerate(distances), key=itemgetter(1))

        indices, distances = zip(*sorted_distances[:k])
        neighbors = [self.id_word[idx] for idx in indices]
        return {n: d for n, d in izip(neighbors, distances)}

        # def embedding_dist(word1,word2,embedding,word_2_id)

    def plot_emb(self, model=TruncatedSVD(n_components=2), show=True, save=None, alpha=.005):

        if model is not None:
            comps = model.fit_transform(self.embeddings)
        else:
            comps = self.embeddings[:, 0:2]

        plt.scatter(comps[:,0], comps[:,1] ,alpha=alpha )

        if save is not None:
            plt.savefig(save, dpi=100)
        if show:
            plt.show()


en_2_es = pickle.load(open('data/en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/es_2_en.pkl', 'r'))
enws = set(en_2_es.keys())
esws = set(es_2_en.keys())

en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')


def softmax(scores):
    if isinstance(scores, list):
        denom = sum(math.exp(score) for score in scores)
        return [math.exp(score) / denom for score in scores]
    elif isinstance(scores, np.ndarray):
        denom = sum(np.exp(scores))
        return np.exp(scores) / denom
    elif isinstance(scores, dict):
        denom = sum(math.exp(score) for score in scores.values())
        return {word: math.exp(score) / denom for word, score in scores.iteritems()}


def translation_quality(candidate, true_translation=en_2_es, mode="wholistic"):
    matches = totals = 0
    for cand_o, cand_t in candidate.iteritems():

        if isinstance(cand_t, tuple):
            cand_t = cand_t[0]
        if isinstance(cand_t, set):
            # Assume an even distribution over set based translations
            cand_t = {word: 1. / len(cand_t) for word in cand_t}

        if cand_o in true_translation.keys():
            true_words = true_translation[cand_o][0]
            for cand_t_word, weight in cand_t.iteritems():
                if cand_t_word in true_words:
                    if mode == "wholistic":
                        matches += weight
                    if mode == "one correct":
                        matches += 1
                        break
        totals += 1
    return matches / float(totals)


'''
def average_translation_dist(candidate, true_translation=en_2_es, embedding=es_embedding, mode="wholistic"):
    matches = totals = 0
    for cand_o, cand_t in candidate.iteritems():

        if isinstance(cand_t, tuple):
            cand_t = cand_t[0]
        if isinstance(cand_t, set):
            # Assume an even distribution over set based translations
            cand_t = {word: 1. / len(cand_t) for word in cand_t}

        if cand_o in true_translation.keys():
            true_words = true_translation[cand_o][0]
            for cand_t_word, weight in cand_t.iteritems():
                for true_t_word in true_words:


        totals += 1
    return matches / float(totals)
'''


def sub_embedding(embedding, word_subset):
    if isinstance(word_subset, list):
        word_subset = set(word_subset)
    word_subset = word_subset.intersection(embedding.word_set)
    new_embedding = copy.copy(embedding)
    new_embedding.word_id = {k: v for k, v in embedding.word_id.iteritems() if k in word_subset}
    new_embedding.id_word = {v: k for k, v in new_embedding.word_id.iteritems()}
    sorted_id_word = sorted(new_embedding.id_word.iteritems())
    new_embedding.embeddings = embedding.embeddings[np.array([id for id, word in sorted_id_word])]

    #Set the word id dictionary to start at 1
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


def nn_embedding_translate(words=en_2_es.keys(), embedding1=en_embedding, embedding2=es_embedding,
                           constraint=es_2_en.keys(), k=5,
                           pre_transform=None, log=False):
    if pre_transform is not None:
        pre_transform_1 = clone(pre_transform)
        pre_transform_2 = clone(pre_transform)
        embedding1 = transform(embedding1, pre_transform_1)
        embedding2 = transform(embedding2, pre_transform_2)

    if constraint is not None:
        embedding2 = sub_embedding(embedding2, constraint)

    in_vocab_words = [word for word in words if embedding1.normalize(word) is not None]
    if log:
        print "{} of {} words in vocab".format(len(in_vocab_words), len(words))

    output = {}
    for i, word in enumerate(in_vocab_words):
        if i % 100 == 0:
            if log:
                print "{} of {} words".format(i, len(words))

        emb = embedding1.word_to_embedding(word)
        if emb is not None:
            trans = embedding2.words_closest_to_point(emb, k=k)
            trans = softmax(trans)
            output[word] = trans
    return output


def get_words_of_cat(cat, dict=en_2_es, language='both'):
    english_words = set(en for en, (es, c) in dict.iteritems() if cat in c)
    spanish_words = list(es for en, (es, c) in dict.iteritems() if cat in c)
    spanish_words = reduce(lambda x, y: x | y, spanish_words)
    if language == 'both':
        return english_words, spanish_words
    if language == 'en':
        return english_words
    elif language == 'es':
        return spanish_words
    else:
        raise ValueError("Need to enter in valid language param")

#def get_complement_of_cats(cats,dict,language='both'):


def plot_comp(model=TruncatedSVD(), subsample=1000, alpha=.005, alpha_cat=.5,
              cats=['color', 'profession', 'science', 'transport'], dict=en_2_es,
              en_embeddings=en_embedding, es_embeddings=es_embedding):
    colors = ['r', 'g', 'k', 'y']
    names = ["english", "spanish"]
    embeddings = [en_embedding, es_embedding]

    if subsample is not None:

        en_cat_words = reduce(lambda x, y : x | y, [get_words_of_cat(cat, dict,'en') for cat in cats])
        es_cat_words = reduce(lambda x, y : x | y, [get_words_of_cat(cat, dict,'es') for cat in cats])
        cat_words = [en_cat_words, es_cat_words]
        embeddings = [subsample_embedding(emb, subsample, cat_word)
                      for emb, cat_word in zip(embeddings, cat_words)]

    plt.figure(figsize=(15, 8))
    for i, emb in enumerate(embeddings):
        if model is not None:
            model_copy = clone(model)
            emb = transform(emb, model_copy)

        plt.subplot(1, len(embeddings), i + 1)
        plt.scatter(emb.embeddings[:, 0], emb.embeddings[:, 1], alpha=alpha)
        for j, cat in enumerate(cats):
            language_cat_words = get_words_of_cat(cat, dict)
            cat_emb = [emb.word_to_embedding(word) for word in language_cat_words[i]]
            # note: line above based on the output of get words of cat
            cat_emb = np.array(filter(lambda x: x is not None, cat_emb))

            plt.scatter(cat_emb[:, 0], cat_emb[:, 1], alpha=alpha_cat, color=colors[j], label=cat)
        plt.title("Projection onto {} of {} words".format(str(model_copy).split('(')[0], names[i]))
    plt.legend()
    plt.savefig(
        'plots/' +
        str(model_copy).split('(')[0] +
        '_'+str(subsample//1000) +
        'k_embedding.png', dpi=100)


run_knn = False
if run_knn:
    en_embedding.knn("Dog")
    print
    en_embedding.knn("Farm")
    print
    es_embedding.knn("perro")
    print
    en_embedding.knn("granja")


#model = Pipeline([('Scaler', StandardScaler()), ('embedder', TruncatedSVD(n_components=2))])
#model = TruncatedSVD(n_components=20)
#model = TSNE(n_components=2)

model = MDS(n_components=2)
plot_comp(model=model, subsample=1000, alpha=.01)
#embedding_translation = nn_embedding_translate()

#embedding_translation = nn_embedding_translate(pre_transform=model)
#print "Translation with pt:{} scored:{}".format(model, translation_quality(embedding_translation))



try_search=False
if try_search:
    dims = [2, 5, 10, 20, 30, None]
    for dim in dims:
        pre_transforms = [TruncatedSVD(n_components=dim),
                          PCA(n_components=dim),
                          # KernelPCA(n_components=dim, kernel='rbf'), #TODO fix memory error / try on larger machine
                          SparsePCA(n_components=dim)]  # ,
                          # DictionaryLearning(n_components=dim)]

        for pre_transform in pre_transforms:
            embedding_translation = nn_embedding_translate(pre_transform=pre_transform)
            print "Translation with pt:{} scored:{}".format(pre_transform,
                                                            translation_quality(embedding_translation))

    print translation_quality(en_2_es)
