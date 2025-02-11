# K-Nearest Neighbors Implementation
import numpy as np
from collections import Counter # helps count occurences of elements (used to find the most common class label)

class KNN:
    #Initializes a KNN object with k (The number of nearest neighbors)
    def __init__(self, k=3):
        """
        Initialize the KNN classifier with the number of neighbors (k).
        Default value is k=3.

        """
        self.k = k
    
    def fit(self, x_train, y_train):
        """ 
            Store the training data x_train and labels y_train
            KNN doesn't train a model in advance, instead it just stores the training data and makes predictions later by comparing new data points to it 
         """
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        """
        Predict the class labels for each test sample
        Calls the _predict function for each test sample and returns predictions as a NumPy array.
        """
        predictions = [self._predict(x) for x in x_test]
        return np.array(predictions)
    
    def _predict(self, x):
        """
        Predict the class label of a single test sample 'x' using the KNN algorithm.
        1. Compute distances to all training samples.
        2. Find the k nearest neighbors.
        3. Select the most common class label among them.
        """

        # Compute Euclidean distances between x and all training points
        distances = [np.linalg.norm(x - x_train) for x_train in self.x_train]
        # Sort distances and get the indices of k closest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Find the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]