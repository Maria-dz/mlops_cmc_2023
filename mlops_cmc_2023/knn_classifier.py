import euclidean_distance, cosine_distance from distances.py
import numpy as np
import pandas as pd

class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.tbs = test_block_size
        self.X_train = None
        self.y_train = None
        self.model = None

    def fit(self, X, y):
        if self.strategy != "my_own":
            self.model = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy,
                                                            metric=self.metric)
            self.model.fit(X)
            self.y_train = y
        else:
            self.X_train = X
            self.y_train = y


    def create_block(self, X, count, tbs):
        start = tbs * count
        end = start + tbs
        if end > np.shape(X)[0]:
            end = np.shape(X)[0]
        return X[start:end]

    def find_kneighbors(self, X, return_distance):
        if self.strategy != "my_own":
            return self.model.kneighbors(X, n_neighbors=self.k, return_distance=return_distance)

        else:
            if self.metric == "euclidean":
                distances = euclidean_distance(X, self.X_train)
            else:
                distances = cosine_distance(X, self.X_train)
            indx = np.argsort(distances)[:, :self.k]
            dist = np.zeros((np.shape(distances)[0], self.k))
            for i in range(np.shape(distances)[0]):
                dist[i] = distances[i][indx[i]]

            if return_distance:
                return dist, indx
            else:
                return indx

    def predict(self, X):
        def f(x):
            return 1/(x+0.00001)

        count = 0
        answ = np.zeros(np.shape(X)[0])
        X_block = self.create_block(X, count, self.tbs)
        while np.shape(X_block)[0] != 0:
            a, b = self.find_kneighbors(X_block, return_distance=True)
            distances = a
            y = self.y_train[b]
            if self.weights:
                w_dist = f(distances)

            for i in range(np.shape(X_block)[0]):
                if self.weights:
                    counter = np.zeros(np.shape(np.unique(self.y_train)))
                    for j in range(self.k):
                        counter[y[i][j]] += w_dist[i][j]
                    answ[i + count * self.tbs] = np.argmax(counter)
                else:
                    counter = np.bincount(y[i])
                    answ[i + count * self.tbs] = np.argmax(counter)

            count += 1
            X_block = self.create_block(X, count, self.tbs)
        return answ
