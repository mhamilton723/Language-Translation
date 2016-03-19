
__author__ = 'Mark'

import matplotlib.pyplot as plt
import math

from embedding import *

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, SparsePCA, DictionaryLearning
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def flatten(dictionary):
    X = []
    Y = []
    for en_word,(es_words, cat) in en_2_es.iteritems():
        for es_word in es_words:
            X.append(en_word)
            Y.append(es_word)
    return X, Y

#def remove_nones(list1,list2):
#    out1=[]
#    out2=[]
#    for word1,word2 in zip(list1,list2):
#        if word1 is not None and word2 is not None:
#            out1.append(word1)
#            out2.append(word2)
#    return out1, out2

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



en_2_es = pickle.load(open('data/en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/es_2_en.pkl', 'r'))
enws = set(en_2_es.keys())
esws = set(es_2_en.keys())
en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')



X,Y = flatten(en_2_es)

en_embs = en_embedding.word_to_embedding(X)
es_embs = es_embedding.word_to_embedding(Y)
en_embs, es_embs = remove_nones([en_embs, es_embs])

X_arr = np.array(en_embs)
Y_arr = np.array(es_embs)

X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

rmses = ((Y_pred-Y_test)**2).mean(axis=1)**.5
ranked_rmses = sorted([(rmse, i) for i, rmse in enumerate(rmses)])

best_10 = ranked_rmses[1:10]
worst_10 = ranked_rmses[-1:-10]

for rmse,i in best_10:
    en_emb = X_test[i]
    en_word = en_embedding.words_closest_to_point(en_emb,1)
    es_emb = Y_test[i]
    es_word = es_embedding.words_closest_to_point(es_emb,1)
    es_pred_emb = Y_pred[i]
    es_pred_word = es_embedding.words_closest_to_point(es_pred_emb,1)
    print "rmse = {}, en_word = {}, es_word = {}, es_pred = {}".format(rmse,en_word,es_word,es_pred_word)







