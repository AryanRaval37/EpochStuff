import numpy as np
from distance_metrics import euclidean

class KNN:
    # expects K value and the distance metric
    def __init__(self, k = 3, distance_metric=euclidean):
        self.k = k
        self.dist=distance_metric

    # no training happens in KNN anyways, just store data
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_one(self, X_test, weighted=False):
        if not weighted:
            dists = []
            # make dists array with all the distances and respective classes
            for i in range(len(self.X)):
                dists.append((self.dist(X_test, self.X[i]), self.y[i]))
            # sort based on distances
            dists.sort(key= lambda x: x[0])
            k_nearest_labels = [label for _, label in dists[:self.k]]
            # now choose the most common lab
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            max_index = np.argmax(counts)
            return values[max_index]
        else:
            dists = []
            # make dists array with all the distances and respective classes
            for i in range(len(self.X)):
                dists.append((self.dist(X_test, self.X[i]), self.y[i]))
            # sort based on distances
            dists.sort(key=lambda x: x[0])
            k_nearest = dists[:self.k]
            # compute Gaussian kernel weights: w = exp(-d^2 / (2 * sigma^2))
            weight_map = {}
            sigma = 1.0
            for dist, label in k_nearest:
                w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                weight_map[label] = weight_map.get(label, 0) + w
            # return label with highest accumulated weight
            return max(weight_map, key=weight_map.get)

    def predict(self, X_test, weighted=False):
        return [self.predict_one(X_test[i], weighted=weighted) for i in range(len(X_test))]