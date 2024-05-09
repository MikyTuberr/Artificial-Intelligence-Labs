import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    return centroids


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = np.zeros((k, data.shape[1]))

    # Choose the first centroid randomly from the data
    centroids[0] = data[np.random.choice(data.shape[0])]

    # Choose the remaining centroids using k-means++ algorithm
    for i in range(1, k):
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            distances[:, j] = np.sqrt(((data - centroids[j])**2).sum(axis=1))

        # Choose the next centroid with probability proportional to the squared distance
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances**2 / np.sum(min_distances**2)
        next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
        centroids[i] = data[next_centroid_index]

    return centroids


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in centroids])
    assignments = np.argmin(distances, axis=0)
    return assignments


def update_centroids(data, assignments, k):
    # TODO find new centroids based on the assignments
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_data = data[assignments == i]
        new_centroids[i] = np.mean(cluster_data, axis=0)

    return new_centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, num_centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
