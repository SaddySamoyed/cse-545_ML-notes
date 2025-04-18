"""EECS545 HW5 Q1. K-means"""

import numpy as np
import sklearn.metrics


def hello():
    print('Hello from kmeans.py!')


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the pixel error between the data and compressed data.
    
    Arguments:
        x: A numpy array of shape (N*, d), where d is the data dimension.
        y: A numpy array of shape (N*, d), where d is the data dimension.
    Return:
        errors: A numpy array of shape (N*). Euclidean distances.
    """
    assert x.shape == y.shape
    error = np.sqrt(np.sum(np.power(x - y, 2), axis=-1))
    return error


def train_kmeans(train_data: np.ndarray, initial_centroids, *,
                 num_iterations: int = 50):
    """K-means clustering.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        initial_centroids: A numpy array of shape (K, d), where
            K is the number of clusters. Each data point means the initial
            centroid of cluster. You should NOT assume K = 16.
        num_iterations: Run K-means algorithm for this number of iterations.

    Returns:
        centroids: A numpy array of (K, d), the centroid of K-means clusters
            after convergence.
    """
    # Sanity check
    N, d = train_data.shape
    K, d2 = initial_centroids.shape
    if d != d2:
        raise ValueError(f"Invalid dimension: {d} != {d2}")

    # We assume train_data contains a real-valued vector, not integers.
    assert train_data.dtype.kind == 'f'

    centroids = initial_centroids.copy()
    for i in range(num_iterations):
        
        ###########################################################################
        # Implement K-means algorithm.
        ###########################################################################
    
        # E: Assign each data point to the closest centroid.
        # shape (N, K). Each row a one-hot vector, where the index of the
        # maximum value is the index of the closest centroid.
        distances = sklearn.metrics.pairwise_distances(train_data, centroids)  # shape (N, K)
        labels = np.argmin(distances, axis=1)  # shape (N,)

        # M: Compute the new centroids as the mean of the assigned data points.
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            assigned_points = train_data[labels == k]
            if len(assigned_points) > 0:
                new_centroids[k] = np.mean(assigned_points, axis=0)
            else:
                new_centroids[k] = centroids[k]  # Keep old if no points assigned

        centroids = new_centroids
        
        # monitor convergence
        assigned_centroids = centroids[labels]
        mean_error = np.mean(euclidean_distance(train_data, assigned_centroids))
        #######################################################################
        print(f'Iteration {i:2d}: mean error = {mean_error:2.2f}')

    # This should contain the centroid points after convergence.
    assert centroids.shape == (K, d)
    return centroids


def compress_image(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compress image by mapping each pixel to the closest centroid.

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        centroids: A numpy array of shape (K, 3), each row being the centroid
            of a cluster.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K, C2 = centroids.shape
    assert C == C2 == 3, "Invalid number of channels."
    assert image.dtype == np.uint8

    ###########################################################################
    # Implement K-means algorithm.
    ###########################################################################
    # Flatten image to (H*W, 3)
    flat_image = image.reshape(-1, 3).astype(np.float32)
    
    # Compute distances to centroids
    distances = sklearn.metrics.pairwise_distances(flat_image, centroids)  # (H*W, K)
    
    # Assign each pixel to the closest centroid
    labels = np.argmin(distances, axis=1)  # H*W,)
    compressed_image_flat = centroids[labels]  # (H*W, 3)
    
    # Reshape back to original image shape
    compressed_image = compressed_image_flat.reshape(H, W, C).astype(np.uint8)
    
    #######################################################################

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
