from KMeansClustering import KMeansClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

def main():
	# X = np.random.randint(0, 100, (100, 2))

	# kmeans = KMeansClustering(k=3)
	# labels = kmeans.fit(X)

	# plt.scatter(X[:, 0], X[:, 1], c=labels)
	# plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='x', s=200, color='red')
	# plt.title("Data Points with Centroids")
	# plt.show()
	# plt.close()

	data = make_blobs(n_samples=100, n_features=2, centers=3)
	random_points = data[0]

	kmeans = KMeansClustering(k=3)
	labels = kmeans.fit(random_points)

	ars = adjusted_rand_score(labels, data[1])
	print("Adjusted Random Score: ", ars)

	plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
	plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker="*", s=200, c=range(len(kmeans.centroids)))
	plt.title("Data Points with Centroids")
	plt.show()

if __name__ == "__main__":
	main()

