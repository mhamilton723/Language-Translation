import pickle
import math
from embedding import *
import numpy as np
from sklearn.base import clone

__author__ = 'Mark'

class Translator(object):

    def __init__(self):
        pass


en_2_es = pickle.load(open('data/en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/es_2_en.pkl', 'r'))
enws = set(en_2_es.keys())
esws = set(es_2_en.keys())

en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')


def softmax(scores):
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
            for cand_t_word, weight in cand_t:
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


def flatten(dictionary):
    X = []
    Y = []
    for en_word, (es_words, cat) in en_2_es.iteritems():
        for es_word in es_words:
            X.append(en_word)
            Y.append(es_word)
    return X, Y


def remove_nones(list_of_lists):
    out_list_of_lists = []
    for i in range(len(list_of_lists)):
        out_list_of_lists.append([])

    for words in zip(*list_of_lists):
        add_words = False not in map(lambda w: w is not None, words)
        if add_words:
            for i, word in enumerate(words):
                out_list_of_lists[i].append(word)
    return out_list_of_lists


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
        if log and i % 100 == 0:
            print "{} of {} words".format(i, len(words))

        emb = embedding1.word_to_embedding(word)
        if emb is not None:
            trans = embedding2.words_closest_to_point(emb, k=k)
            trans = softmax(trans)
            output[word] = trans
    return output


def regression_translation(dictionary=en_2_es, embedding1=en_embedding, embedding2=es_embedding,
                           constraint=es_2_en.keys(), k=1, test_size=.2,
                           model=LinearRegression(), log=False):
    X, Y = flatten(dictionary)
    embs1 = embedding1.word_to_embedding(X)
    embs2 = embedding2.word_to_embedding(Y)
    embs1, embs2, X, Y = remove_nones([embs1, embs2,X,Y])

    if constraint is not None:
        embedding2 = sub_embedding(embedding2, constraint)

    X_arr = np.array(embs1)
    Y_arr = np.array(embs2)
    indices = np.array(range(X_arr.shape[0]))
    X_train, X_test, Y_train, Y_test, indices_train, indices_test =\
        train_test_split(X_arr, Y_arr, indices, test_size=test_size)

    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    Y_pred_train = model.predict(X_train)
    test_dict = {}
    train_dict = {}

    for i in range(X_test.shape[0]):
        if log and i%100 == 0:
            print "testing set: {} of {}".format(i,X_test.shape[0])
        emb1 = X_test[i]
        word1 = X[indices_test[i]]
        pred_emb2 = Y_pred_test[i]
        trans = embedding2.words_closest_to_point(pred_emb2, k)
        trans = softmax(trans)
        test_dict[word1] = trans

    for i in range(X_train.shape[0]):
        if log and i%100 == 0:
            print "training set: {} of {}".format(i,X_train.shape[0])

        emb1 = X_train[i]
        word1 = X[indices_train[i]]
        pred_emb2 = Y_pred_train[i]
        trans = embedding2.words_closest_to_point(pred_emb2, k)
        trans = softmax(trans)
        train_dict[word1] = trans

    return train_dict, test_dict
