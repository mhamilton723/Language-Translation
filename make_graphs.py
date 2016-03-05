__author__ = 'Mark'
import networkx as nx
from embedding import *

from sklearn.neighbors import NearestNeighbors
import numpy as np

import os
'''
def embedding_to_nngraph(embedding, k, log=True, epsilon=.000001):
    graph = nx.Graph()
    graph.add_nodes_from(embedding.word_set)

    total = len(embedding.word_set)
    for i, word in enumerate(embedding.word_set):
        if log and i % 100 == 0:
            print "{} of {}".format(i, total)
        pairs = embedding.knn(word, k=k, return_distances=True)
        neighbors, distances = zip(*pairs)
        if epsilon is not None:
            new_edges = zip([word] * len(neighbors), neighbors, [1. / (d + epsilon) for d in distances])
            graph.add_weighted_edges_from(new_edges)
        else:
            new_edges = zip([word] * len(neighbors), neighbors)
            graph.add_edges_from(new_edges)
    return graph
'''

def embedding_to_nngraph(embedding, k, log=True, epsilon=.000001):
    graph = nx.DiGraph()
    graph.add_nodes_from(embedding.word_set)

    X = np.array(embedding.embeddings)
    dists, indices = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X).kneighbors(X)

    for dist, index in zip(dists, indices):
        for i in range(1, k+1):
            graph.add_weighted_edges_from([(embedding.id_word[index[0]], embedding.id_word[index[i]], dist[i])])
    return graph




'''
def embedding_to_distgraph(embedding,log=True, cutoff=1):
    graph = nx.Graph()
    graph.add_nodes_from(embedding.word_set)
    total = len(embedding.word_set)
    for i, word1 in enumerate(embedding.word_set):
                if log and i%10 == 0:
            print "{} of {}".format(i,total)
        for j, word2 in enumerate(embedding.word_set):
            emb1 = embedding.word_to_embedding(word1)
            emb2 = embedding.word_to_embedding(word2)
            dist = ((emb1-emb2)**2).sum() ** 0.5
            if dist < cutoff:
                graph.add_edge(word1, word2)
'''
#os.chdir(os.path.expanduser('~'))
#os.chdir('/Language-Translation')


en_embedding = Embedding('data/polyglot-en.pkl')
es_embedding = Embedding('data/polyglot-es.pkl')

en_nn_graph = embedding_to_nngraph(en_embedding, k=5)
pickle.dump(en_nn_graph, open('data/en_nn_graph.pkl', 'w+'))

es_nn_graph = embedding_to_nngraph(es_embedding, k=5)
pickle.dump(es_nn_graph, open('data/es_nn_graph.pkl', 'w+'))
#pickle.dump(es_nn_graph, open('data/es_nn_graph.pkl', 'w+'))

#for n, nbrs in en_nn_graph.adjacency_iter():
#    for nbr, eattr in nbrs.items():
#        data = eattr['weight']
#        print('(%s, %s, %.3f)' % (n, nbr, data))
