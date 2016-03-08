import math
from embedding import *
import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import ShuffleSplit

__author__ = 'Mark'

class Translator(object):
    def __init__(self):
        pass

    def softmax(self, scores):
        if isinstance(scores, list):
            if isinstance(scores[0], tuple):
                words = [word for word, score in scores]
                scores = [score for word, score in scores]
                denom = sum(math.exp(score) for score in scores)
                return zip(words, [math.exp(score) / denom for score in scores])
            else:
                denom = sum(math.exp(score) for score in scores)
                return [math.exp(score) / denom for score in scores]
        elif isinstance(scores, np.ndarray):
            denom = sum(np.exp(scores))
            return np.exp(scores) / denom
        elif isinstance(scores, dict):
            denom = sum(math.exp(score) for score in scores.values())
            return {word: math.exp(score) / denom for word, score in scores.iteritems()}

    def _flatten(self, dictionary):
        X = []
        Y = []
        for en_word, es_words in dictionary.iteritems():
            for es_word in es_words:
                X.append(en_word)
                Y.append(es_word)
        return X, Y

    def _remove_nones(self, list_of_lists):
        out_list_of_lists = []
        for i in range(len(list_of_lists)):
            out_list_of_lists.append([])

        for words in zip(*list_of_lists):
            add_words = False not in map(lambda w: w is not None, words)
            if add_words:
                for i, word in enumerate(words):
                    out_list_of_lists[i].append(word)
        return out_list_of_lists


class EmbeddingTranslator(Translator):
    def __init__(self, embedding1, embedding2, constraint, pre_transform=None):
        Translator.__init__(self)
        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.constraint = constraint
        self.pre_transform = pre_transform

        self._pre_transform()
        self._constrain()

    def _pre_transform(self):
        if self.pre_transform is not None:
            pre_transform_1 = clone(self.pre_transform)
            pre_transform_2 = clone(self.pre_transform)
            self.embedding1 = transform(self.embedding1, pre_transform_1)
            self.embedding2 = transform(self.embedding2, pre_transform_2)

    def _constrain(self):
        if self.constraint is not None:
            constraint_words = self.embedding2.in_vocab_words(self.constraint)
            print 'constraint added: considering {} of {} words'.format(len(constraint_words),
                                                                        len(self.embedding2.word_set))
            self.embedding2 = sub_embedding(self.embedding2, constraint_words)

    def translate(self, words, log=False):
        raise NotImplementedError


class SimpleEmbeddingTranslator(EmbeddingTranslator):
    def __init__(self, embedding1, embedding2, constraint, k, pre_transform=None):
        EmbeddingTranslator.__init__(self, embedding1, embedding2, constraint, pre_transform)
        self.k = k

    def translate(self, words, log=False):
        output = {}
        for i, word in enumerate(words):
            if log and i % 100 == 0:
                print "{} of {} words".format(i, len(words))
            output[word] = self._translate_word(word)
        return output

    def _translate_word(self, word):
        emb = self.embedding1.word_to_embedding(word)
        if emb is not None:
            trans = self.embedding2.words_closest_to_point(emb, k=self.k)
            trans = self.softmax(trans)
            return trans


class RegressionEmbeddingTranslator(EmbeddingTranslator):
    def __init__(self, embedding1, embedding2, constraint, k, model, pre_transform=None):
        EmbeddingTranslator.__init__(self, embedding1, embedding2, constraint, pre_transform)
        self.k = k
        self.model = model
        self.is_fit = False
        self._transformed_embeddings = None

    def translate(self, words, log=False):
        if not self.is_fit:
            raise ValueError('Need to fit translator first')
        embs1 = self.embedding1.word_to_embedding(words)
        embs1_fix = [emb for emb in embs1 if emb is not None]
        indices_fix = [ind for ind,emb in zip(range(len(embs1)),embs1) if emb is not None]
        if log:
            print '{} of {} words in vocabulary'.format(len(embs1_fix), len(embs1))
        arr_embs1 = np.array(embs1_fix)
        pred_embs2 = self.model.predict(arr_embs1)

        output = {}
        for i, ind in enumerate(indices_fix):
            if log and i % 100 == 0:
                print "{} of {} words".format(i, len(words))
            pred_emb2 = pred_embs2[i]
            trans = self.embedding2.words_closest_to_point(pred_emb2, self.k)
            trans = self.softmax(trans)
            output[words[ind]] = trans
        return output

    def fit(self, dictionary):
        X, Y = self._flatten(dictionary)
        embs1 = self.embedding1.word_to_embedding(X)
        embs2 = self.embedding2.word_to_embedding(Y)
        embs1, embs2 = self._remove_nones([embs1, embs2])
        X_arr = np.array(embs1)
        Y_arr = np.array(embs2)
        self.model.fit(X_arr, Y_arr)
#        self.model.fit(X_arr.astype(np.double), Y_arr.astype(np.double))
        self.is_fit = True

def get_translation_parts(candidate,true_translation, mode="wholistic", log = True):
    matches = totals = 0
    candidate = {k: v for k, v in candidate.iteritems() if v is not None}
    good_translations={}
    bad_translations = {}
    for cand_o, cand_t in candidate.iteritems():

        if isinstance(cand_t, tuple):
            cand_t = cand_t[0]
        if isinstance(cand_t, set):
            # Assume an even distribution over set based translations
            cand_t = {word: 1. / len(cand_t) for word in cand_t}

        if cand_o in true_translation:
            true_words = true_translation[cand_o]
            for cand_t_word, weight in cand_t:
                if cand_t_word in true_words:
                    good_translations[cand_o] = {cand_t_word}
                else:
                    bad_translations[cand_o] = {cand_t_word}

    return good_translations, bad_translations

def _standardize_translation(cand_t):
    if isinstance(cand_t, tuple):
        cand_t = cand_t[0]
    if isinstance(cand_t, set):
        # Assume an even distribution over set based translations
        cand_t = [(word, 1. / len(cand_t)) for word in cand_t]
    if isinstance(cand_t, dict):
        cand_t = [(k, v) for k, v in cand_t.iteritems()]
    return cand_t


def translation_quality(candidate,true_translation, mode="wholistic", log = True):
    matches = totals = 0
    candidate = {k: v for k, v in candidate.iteritems() if v is not None}
    for cand_o, cand_t in candidate.iteritems():

        cand_t = _standardize_translation(cand_t)

        if cand_o in true_translation:
            true_words = true_translation[cand_o]
            for cand_t_word, weight in cand_t:
                if cand_t_word in true_words:
                    if mode == "wholistic":
                        matches += weight
                    if mode == "one correct":
                        matches += 1
                        break
        totals += 1
    if log: print 'matches: {} totals: {}'.format(matches, totals)
    return matches / float(totals)

def translation_distance(candidate,true_translation, embedding, mode="average",d='l2',return_all=False, log = True):
    candidate = {k: v for k, v in candidate.iteritems() if v is not None}
    accum_distances = []

    if d == 'l2':
        distance = lambda v1, v2: np.sum((v1-v2)**2)**.5
    else:
        raise ValueError('Input correct name of distance')

    for cand_o, cand_t in candidate.iteritems():

        cand_t = _standardize_translation(cand_t)

        if cand_o in true_translation:
            true_words = true_translation[cand_o]
            true_vects = embedding.word_to_embedding(true_words)
            true_vects_in_vocab = len([true_vect for true_vect in true_vects if true_vect is not None]) >= 1

            if true_vects_in_vocab:
                accum_distance = 0
                for trans_word, weight in cand_t:
                    trans_vect = embedding.word_to_embedding(trans_word)
                    distances = np.array([distance(true_vect, trans_vect)
                                        for true_vect in true_vects if true_vect is not None])

                    if mode == "average":
                        accum_distance += weight*np.mean(distances)
                accum_distances.append(accum_distance)

    if return_all:
        return accum_distances
    else:
        return np.mean(np.array(accum_distances))

def get_true_and_false_matched(candidate, true_translation, mode="wholistic", log = True):
    matches = totals = 0
    candidate = {k: v for k, v in candidate.iteritems() if v is not None}
    for cand_o, cand_t in candidate.iteritems():
        if isinstance(cand_t, tuple):
            cand_t = cand_t[0]
        if isinstance(cand_t, set):
            # Assume an even distribution over set based translations
            cand_t = {word: 1. / len(cand_t) for word in cand_t}

        if cand_o in true_translation:
            true_words = true_translation[cand_o]
            for cand_t_word, weight in cand_t:
                if cand_t_word in true_words:
                    if mode == "wholistic":
                        matches += weight
                    if mode == "one correct":
                        matches += 1
                        break
        totals += 1
    if log: print 'matches: {} totals: {}'.format(matches, totals)
    return matches / float(totals)



def dict_train_test_split(dictionary,train_size, cap_train=None, cap_test=None):
    d_list = list(dictionary.iteritems())

    if isinstance(train_size, int) and train_size > 1:
        train_size /= float(len(d_list))

    test_size = 1.-train_size
    indices_train, indices_test = iter(ShuffleSplit(len(d_list), n_iter=1, test_size=test_size)).next()
    d_train_list = [d_list[index_train] for index_train in indices_train]
    d_test_list = [d_list[index_test] for index_test in indices_test]

    if cap_train is not None:
        d_train_list = d_train_list[0:cap_train]
    if cap_test is not None:
        d_test_list = d_test_list[0:cap_test]

    return dict(d_train_list), dict(d_test_list)





