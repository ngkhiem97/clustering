import numpy as np
from .kmeans import Kmeans

class SpectralClustering:
    def __init__(self, k, data, similarity='cosine', sigma=1.0):
        self.k = k
        self.data = data
        self.similarity = similarity
        self.sigma = sigma
        self.nmember, self.ndim = data.shape
        self.W = np.zeros((self.nmember, self.nmember))
        self.D = np.zeros((self.nmember, self.nmember))
        self.L = np.zeros((self.nmember, self.nmember))
        self.eigvecs = np.zeros((self.nmember, self.nmember))
        self.eigvals = np.zeros((self.nmember, self.nmember))
    
    def get_cosine_distance(self, point1, point2):
        return np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    
    def get_gaussian_distance(self, point1, point2):
        distance = np.sqrt(np.sum((point1 - point2) ** 2))
        return np.exp(-distance**2 / (2 * self.sigma ** 2))

    def get_similarity_matrix(self):
        for i in range(self.nmember):
            for j in range(i, self.nmember):
                if self.similarity == 'cosine':
                    self.W[i][j] = self.get_cosine_distance(self.data[i], self.data[j])
                elif self.similarity == 'gaussian':
                    self.W[i][j] = self.get_gaussian_distance(self.data[i], self.data[j])
                self.W[j][i] = self.W[i][j]

    def get_degree_matrix(self):
        degreeMatrix = np.sum(self.W, axis=1)
        self.D = np.diag(degreeMatrix)

    def get_laplacian_matrix(self):
        self.L = self.D - self.W

    def get_normalized_laplacian_matrix(self):
        sqrtDegreeMatrix = np.diag(1.0 / (np.sum(self.W, axis=1) ** 0.5))
        self.L = np.dot(np.dot(sqrtDegreeMatrix, self.L), sqrtDegreeMatrix)

    def get_eigenvectors(self):
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        self.eigvals = zip(self.eigvals.real, range(self.nmember))
        self.eigvals = sorted(self.eigvals, key=lambda x: x[0])
        self.eigvecs = np.real(np.vstack([self.eigvecs[:,i[1]] for i in self.eigvals]).T)

    def run(self, normalize=True):
        self.get_similarity_matrix()
        self.get_degree_matrix()
        self.get_laplacian_matrix()
        if normalize:
            self.get_normalized_laplacian_matrix()
        self.get_eigenvectors()
        kmeans = Kmeans(k=self.k, data=self.eigvecs[:, 1][..., np.newaxis])
        kmeans.run()
        return kmeans