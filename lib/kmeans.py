
import numpy as np

class Kmeans:

    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.ndim = data.shape[1]
        self.centroids = []
        self.clusters = []
        self.labels = []

    def init_centroids(self, centroids=None):
        if centroids is not None:
            self.centroids = centroids
            return
        for i in range(self.k):
            self.centroids.append([])
            self.centroids[i] = self.data[np.random.choice(range(self.data.shape[0]))]

    def init_clusters(self):
        for _ in range(self.k):
            self.clusters.append([])

    def get_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return distance ** 0.5

    def get_closest_centroid(self, point):
        min_distance = self.get_distance(point, self.centroids[0])
        min_index = 0
        for i in range(1, self.k):
            distance = self.get_distance(point, self.centroids[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return min_index

    def update_clusters(self):
        for i in range(self.k):
            self.clusters[i] = []
        self.labels = []
        for point in self.data:
            index = self.get_closest_centroid(point)
            self.clusters[index].append(point)
            self.labels.append(index)

    def update_centroids(self):
        for i in range(self.k):
            cluster = self.clusters[i]
            if len(cluster) > 0:
                centroid = [0] * len(cluster[0])
                for point in cluster:
                    for j in range(len(point)):
                        centroid[j] += point[j]
                for j in range(len(centroid)):
                    centroid[j] /= len(cluster)
                self.centroids[i] = centroid

    def get_cost(self):
        cost = 0
        for i in range(self.k):
            cluster = self.clusters[i]
            for point in cluster:
                cost += self.get_distance(point, self.centroids[i]) ** 2
        return cost

    def run(self, centroids=None):
        self.init_centroids(centroids)
        self.init_clusters()
        self.update_clusters()
        cost = self.get_cost()
        while True:
            self.update_centroids()
            self.update_clusters()
            new_cost = self.get_cost()
            if new_cost >= cost:
                break
            cost = new_cost