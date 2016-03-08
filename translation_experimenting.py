__author__ = 'Mark'

import matplotlib.pyplot as plt
from embedding import *
from translator import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
import pickle

def gen_random_translation(embedding1, embedding2, constraint=None):
    if constraint is not None:
        constraint = list(constraint)
        embedding2 = sub_embedding(embedding2, constraint)
    out = {word: {random.choice(embedding2.words)} for word in embedding1.word_set}
    return out


print "loading embeddings"
en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')
en_2_es = pickle.load(open('data/gt_en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/gt_es_2_en.pkl', 'r'))
print "loaded embeddings"

# simple_translator = SimpleEmbeddingTranslator(en_embedding, es_embedding, es_2_en.keys(), k=1)
# simple_trans = simple_translator.translate(en_2_es.keys())
# print translation_quality(simple_trans, en_2_es)

gt = True
if not gt:
    en_2_es = {k: v[0] for k, v in en_2_es.iteritems()}
es_words = set.union(*en_2_es.values())

redo_calc = True
reg_trans_fn = 'data/gt_poly_trans.pkl'
if redo_calc:
    print "redoing calculation"
    train_dict, test_dict = dict_train_test_split(en_2_es, train_size=20000, cap_test=5000)
    reg_translator = RegressionEmbeddingTranslator(en_embedding, es_embedding,
                                                   constraint=es_words, k=1, model=RandomForestRegressor())
    reg_translator.fit(train_dict)
    print "beginning translation"
    train_trans = reg_translator.translate(train_dict.keys(), log=True)
    test_trans = reg_translator.translate(test_dict.keys(), log=True)
    pickle.dump((train_trans, test_trans), open(reg_trans_fn, 'w+'))
else:
    train_trans, test_trans = pickle.load(open(reg_trans_fn, 'r'))

random_trans = gen_random_translation(en_embedding, es_embedding, constraint=es_words)

print 'Scoring Dictionaries'
results_train = translation_quality(train_trans, en_2_es)
results_test = translation_quality(test_trans, en_2_es)
results_random = translation_quality(random_trans, en_2_es)

distance_train = translation_distance(train_trans, en_2_es, es_embedding)
distance_test = translation_distance(test_trans, en_2_es, es_embedding)
distance_random = translation_distance(random_trans, en_2_es, es_embedding)

train_good_dict, train_bad_dict = get_translation_parts(train_trans, en_2_es)
test_good_dict, test_bad_dict = get_translation_parts(test_trans, en_2_es)

print "good test dict: "
print test_good_dict.items()[0:20]
print "bad test dict: "
print test_bad_dict.items()[0:20]
print results_train, results_test, results_random
print distance_train, distance_test, distance_random
