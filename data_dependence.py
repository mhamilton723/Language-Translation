__author__ = 'Mark'
import matplotlib.pyplot as plt
from embedding import *
from translator import *

en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')
en_2_es = pickle.load(open('data/gt_en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/gt_es_2_en.pkl', 'r'))

#simple_translator = SimpleEmbeddingTranslator(en_embedding, es_embedding, es_2_en.keys(), k=1)
#simple_trans = simple_translator.translate(en_2_es.keys())
#print translation_quality(simple_trans, en_2_es)

gt = True
if gt:
    no_cat_en_2_es = {k: v for k, v in en_2_es.iteritems()}
else:
    no_cat_en_2_es = {k: v[0] for k, v in en_2_es.iteritems()}

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
import pickle

n_iters = 5
test_sizes = [.1,.2,.3,.4,.5,.6,.8]
results_train = np.zeros((len(test_sizes), n_iters))
results_test = np.zeros((len(test_sizes), n_iters))

run_data_dependence = True
if run_data_dependence:
    for i, test_size in enumerate(test_sizes):
        for j in range(n_iters):
            train_dict, test_dict = dict_train_test_split(no_cat_en_2_es, test_size=test_size)
            reg_translator = RegressionEmbeddingTranslator(en_embedding, es_embedding,
                                                           es_2_en.keys(), k=1, model=LinearRegression())
            reg_translator.fit(train_dict)
            train_trans = reg_translator.translate(train_dict.keys())
            test_trans = reg_translator.translate(test_dict.keys())

            results_train[i, j] = translation_quality(train_trans, en_2_es)
            results_test[i, j] = translation_quality(test_trans, en_2_es)
    pickle.dump((results_train, results_test), open('data/data_dependence.pkl', 'w+'))
else:
    results_train, results_test = pickle.load(open('data/data_dependence.pkl', 'r'))

sample_sizes = np.array([round((1-test_size)*len(no_cat_en_2_es.items()),0) for test_size in test_sizes])
for j in range(n_iters):
    plt.scatter(list(sample_sizes), list(results_train[:, j]), color='b', alpha=.7)
    plt.scatter(list(sample_sizes), list(results_test[:, j]), color='r', alpha=.7)

plt.xlabel('Number of samples in the training set')
plt.ylabel('Percent of words correctly translated')
plt.title('Data Dependence of LinearRegression on Embeddings')
plt.savefig('plots/data_dependence.png')