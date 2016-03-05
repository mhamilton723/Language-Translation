__author__ = 'Mark'

import matplotlib.pyplot as plt
from embedding import *
from translator import *

print "loading embeddings"
en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')
en_2_es = pickle.load(open('data/gt_en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/gt_es_2_en.pkl', 'r'))
print "loaded embeddings"

#simple_translator = SimpleEmbeddingTranslator(en_embedding, es_embedding, es_2_en.keys(), k=1)
#simple_trans = simple_translator.translate(en_2_es.keys())
#print translation_quality(simple_trans, en_2_es)

gt = True
if not gt:
    en_2_es = {k: v[0] for k, v in en_2_es.iteritems()}


es_words = set.union(*en_2_es.values())

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
import pickle

redo_calc = False
if redo_calc:
    print "redoing calculation"
    train_dict, test_dict = dict_train_test_split(en_2_es, test_size=.95)
    reg_translator = RegressionEmbeddingTranslator(en_embedding, es_embedding,
                                                               es_words, k=1, model=LinearRegression())
    reg_translator.fit(train_dict)
    print "beginning translation"
    train_trans = reg_translator.translate(train_dict.keys(), log=True)
    test_trans = reg_translator.translate(test_dict.keys(), log=True)
    pickle.dump((train_trans, test_trans), open('data/reg_trans.pkl', 'w+'))
else:
    train_trans, test_trans = pickle.load(open('data/reg_trans.pkl', 'r'))

print 'Scoring Dictionaries'
results_train = translation_quality(train_trans, en_2_es)
results_test = translation_quality(test_trans, en_2_es)

train_good_dict, train_bad_dict = get_translation_parts(train_trans, en_2_es)
test_good_dict, test_bad_dict = get_translation_parts(test_trans, en_2_es)

print "good test dict: "
print test_good_dict.items()[0:20]
print "bad test dict: "
print test_bad_dict.items()[0:20]
print results_train, results_test
