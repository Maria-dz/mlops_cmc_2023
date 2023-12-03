import numpy as np
import sklearn.neighbors
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(785, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(p=0.2)
        self.log_softmax = F.log_softmax

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.log_softmax(self.fc4(x), dim=1)
        return x


class KNNClassifier:
    def __init__(self, k, metric, strategy="auto", weights=None, tbs=None):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.tbs = tbs
        self.X_train = None
        self.y_train = None
        self.model = None

    def fit(self, X, y):
        if self.strategy != "my_own":
            self.model = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=self.k, algorithm=self.strategy, metric=self.metric
            )
            self.model.fit(X, y)
            self.y_train = y
        else:
            self.X_train = X
            self.y_train = y
        return self.model

    def create_block(self, X, count, tbs):
        start = tbs * count
        end = start + tbs
        if end > np.shape(X)[0]:
            end = np.shape(X)[0]
        return X[start:end]

    def find_kneighbors(self, X, return_distance):
        if self.strategy != "my_own":
            return self.model.kneighbors(
                X, n_neighbors=self.k, return_distance=return_distance
            )

        else:
            distances = np.linalg.norm(X - self.X_train)
            indx = np.argsort(distances)[:, : self.k]
            dist = np.zeros((np.shape(distances)[0], self.k))
            for i in range(np.shape(distances)[0]):
                dist[i] = distances[i][indx[i]]

            if return_distance:
                return dist, indx
            else:
                return indx

    def predict(self, X):
        def f(x):
            return 1 / (x + 0.00001)

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
