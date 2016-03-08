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

en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')
en_2_es = pickle.load(open('data/gt_en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('data/gt_es_2_en.pkl', 'r'))

gt = True
if not gt:
    en_2_es = {k: v[0] for k, v in en_2_es.iteritems()}
es_words = set.union(*en_2_es.values())

n_iters = 5
train_sizes = [2**7, 2**9, 2**11, 2**13]
results_train = np.zeros((len(train_sizes), n_iters))
results_test = np.zeros((len(train_sizes), n_iters))
distances_train = np.zeros((len(train_sizes), n_iters))
distances_test = np.zeros((len(train_sizes), n_iters))

run_data_dependence = False
data_dependence_fn = 'data/data_dependence_linear.pkl'
data_dependence_plot_fn = 'plots/data_dependence_linear.png'

if run_data_dependence:
    for i, train_size in enumerate(train_sizes):
        for j in range(n_iters):
            print "training size: {}, iteration: {}".format(train_size, j)

            train_dict, test_dict = dict_train_test_split(en_2_es, train_size=train_size, cap_test=5000)
            reg_translator = RegressionEmbeddingTranslator(en_embedding, es_embedding,
                                                           constraint=es_words, k=1, model=LinearRegression())
            reg_translator.fit(train_dict)
            train_trans = reg_translator.translate(train_dict.keys(), log=False)
            test_trans = reg_translator.translate(test_dict.keys(), log=False)

            results_train[i, j] = translation_quality(train_trans, en_2_es)
            results_test[i, j] = translation_quality(test_trans, en_2_es)
            distances_train[i, j] = translation_distance(train_trans, en_2_es, es_embedding)
            distances_test[i, j] = translation_distance(test_trans, en_2_es, es_embedding)

    pickle.dump((results_train, results_test, distances_train, distances_test),
                open(data_dependence_fn, 'w+'))
else:
    results_train, results_test, distances_train, distances_test = pickle.load(open(data_dependence_fn, 'r'))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
for j in range(n_iters):
    #plt.scatter(train_sizes, list(results_train[:, j]*100), color='b', alpha=.7)
    plt.scatter(train_sizes, list(results_test[:, j]*100), color='r', alpha=.7)
plt.xscale('log')
plt.xlabel('Number of samples in the training set')
plt.ylabel('Percent of words correctly translated')

plt.subplot(1, 2, 2)
for j in range(n_iters):
    plt.scatter(train_sizes, list(distances_train[:, j]), color='b', alpha=.7)
    plt.scatter(train_sizes, list(distances_test[:, j]), color='r', alpha=.7)
plt.xscale('log')
plt.xlabel('Number of samples in the training set')
plt.ylabel('Distance between predicted and correct translation')

plt.suptitle('Data Dependence of LinearRegression on Embeddings')
plt.savefig(data_dependence_plot_fn)
