import unittest
import numpy as np
from KMeansClustering import KMeansClustering

class TestKMeansClustering(unittest.TestCase):

    def test_fit(self):
        X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
        kmeans = KMeansClustering(k=2)
        y = kmeans.fit(X)
        self.assertEqual(len(y), len(X))  # Check if y has the same length as X

        # Check if centroids have been initialized
        self.assertIsNotNone(kmeans.centroids)
        self.assertEqual(kmeans.centroids.shape[0], 2)  # Check if correct number of centroids

    def test_euclidean_distance(self):
        data_point = np.array([1, 2])
        centroids = np.array([[2, 3], [4, 5], [6, 7]])
        distances = KMeansClustering.euclian_distance(data_point, centroids)
        expected_distances = np.array([np.sqrt(2), np.sqrt(20), np.sqrt(50)])
        np.testing.assert_allclose(distances, expected_distances, rtol=0.05, atol=0.05)

    def test_cluster_assignment(self):
        X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
        kmeans = KMeansClustering(k=2)
        kmeans.centroids = np.array([[1, 2], [8, 8]])
        y = kmeans.fit(X)
        self.assertIn(0, y)  # Check if both clusters are represented in y
        self.assertIn(1, y)

if __name__ == '__main__':
    unittest.main()
