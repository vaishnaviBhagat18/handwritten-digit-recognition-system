# KNN works by calculating the distance between a test image and all training images. It selects the K nearest neighbors and predicts the digit based on majority voting.

# Distance Function
# measure similarity using Euclidean Distance
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Class Implementation
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            label = self._predict(x)
            predictions.append(label)
        return np.array(predictions)

    def _predict(self, x):

        # Compute distances
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]

        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]