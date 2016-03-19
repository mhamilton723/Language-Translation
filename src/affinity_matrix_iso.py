__author__ = 'Mark'

from embedding import *
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.neighbors import NearestNeighbors

en_embedding = Embedding('../data/polyglot-en.pkl')
es_embedding = Embedding('../data/polyglot-es.pkl')
en_2_es = pickle.load(open('../data/gt_en_2_es.pkl', 'r'))
es_2_en = pickle.load(open('../data/gt_es_2_en.pkl', 'r'))

def distance_affinity_matrix(points, method='cosine', log=True, cutoff=.2):
    mat = dok_matrix((points.shape[0], points.shape[0]), dtype=np.float32)
    total=0
    for i in range(points.shape[0]):
        if log and i % 100 == 0:
            print("{} of {} points calculated, {} entries".format(i, points.shape[0],total))
        if method == 'l2':
            distances = (((points - points[i]) ** 2).sum(axis=1) ** 0.5)
        elif method == 'cosine':
            if np.dot(points[i], points[i]) < 10**-9:
                points[i] *= 1. / max(points[i])
            distances = 1 - np.dot(points, points[i]) / (((points ** 2).sum(axis=1) ** .5)*(np.dot(points[i],points[i])** .5))
        else:
            raise ValueError("{} not proper distance method".format(method))
        for j, dist in enumerate(distances):
            if dist < cutoff:
                mat[i, j] = 1. / (1+distances[j])
                total += 1
    return mat

def nn_affinity_matrix(points, k=5, log=True, epsilon=.000001):
    mat = dok_matrix((points.shape[0], points.shape[0]), dtype=np.float32)

    dists, indices = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points).kneighbors(points)

    for dist, index in zip(dists, indices):
        for i in range(1, k+1):
            mat[index[0], index[i]] = 1/dist[i]
    return mat

#mat = distance_affinity_matrix(subsample_embedding(en_embedding, 10000).embeddings)

mat = nn_affinity_matrix(subsample_embedding(en_embedding, 1000).embeddings)
print mat.nnz, mat.shape


