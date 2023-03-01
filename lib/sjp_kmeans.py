import numpy as np
from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class Kmean:
    def __init__(self, k, coord):
        self.k = k
        self.coord = coord
        self.nmember, self.ndim = coord.shape
        self.centroids = [np.empty(self.ndim) for x in range(k)]
        self.clusters = {x:{'idx':[],'coor':[]} for x in range(k)}
        self.pred = [-1 for i in range(self.nmember)]
        self.min_iter = 10
        self.max_iter = 1000
        self.has_update = False

    def init_centroids(self):
        i = 0
        for j in np.random.choice(range(self.nmember), self.k):
            self.centroids[i] = self.coord[j,:]
            i += 1

    def update_clusters(self):
        clusters = {x:{'idx':[],'coor':[]} for x in range(self.k)}
        for i in range(self.nmember):
            coor = self.coord[i,:]
            min_dist = np.infty
            k = None
            for j in range(self.k):
                centroid = self.centroids[j]
                distance = self.get_euclidian_distance(centroid, coor)
                if distance < min_dist:
                    min_dist = distance
                    k = j
            clusters[k]['idx'].append(i)
            clusters[k]['coor'].append(coor)
        self.has_update = False
        for i in range(self.k):
            if (clusters[i]['idx'] != self.clusters[i]['idx']):
                self.has_update = True
        self.clusters = clusters

    def update_centroids(self):
        centroids = [np.zeros(self.ndim) for x in range(self.k)]
        for i in self.clusters:
            members = self.clusters[i]['coor']
            if len(members) > 0:
                for coor in members:
                    centroids[i] += coor
                centroids[i] /= len(members)
        self.centroids = centroids

    def run(self):
        self.init_centroids()
        self.update_clusters()
        i = 0
        while i < self.max_iter:
            self.update_centroids()
            self.update_clusters()
            i += 1
            if self.has_update == False and i > self.min_iter: break

    def get_euclidian_distance(self, c0, c1):
        return np.sqrt(sum((c0 - c1) ** 2))

    def get_hybrid_distance(self, c0, c1):
        pearsonr = stats.pearsonr(c0, c1)[0]
        if np.isnan(pearsonr) or pearsonr < 0:
            pearsonr = 0
        distance = (1.0-pearsonr) + np.sqrt(self.get_euclidian_distance(c0, c1)/10.)
        return distance

    def set_pred(self):
        for i in range(self.k):
            for j in self.clusters[i]['idx']:
                self.pred[j] = i

    def valid_score(self, actual):
        self.set_pred()
        for i,xx in enumerate(self.pred):
            print("member %d is in cluster " % (i+1), xx)
        print("randIndex, Sihouette: ", adjusted_rand_score(actual, self.pred), silhouette_score(self.coord, self.pred))