import numpy as np
import timeit


def generate_datapoints(n, dim=2, random_state=1):
    np.random.seed(random_state)
    return np.random.randn(n, dim)


def slow_knn(datapoints, k=3):
    """
    KNN implemented in python.
    Given datapoints and k as inputs, return a dictionary mapping the
    index of each datapoint to the index of its k nearest neighbors.
    """
    if k > len(datapoints):
        raise ValueError('k must be less than or equal to n')

    nearest_neighbors = {}
    for index, point in enumerate(datapoints):
        neighbors_and_distances = [
            (neighbor_index, _l2_distance(point, neighbor_point))
            for neighbor_index, neighbor_point in enumerate(datapoints) if neighbor_index != index
        ]
        neighbors_and_distances.sort(key=lambda item: item[1])
        closest_points = [neighbor_index for neighbor_index, neighbor_point in neighbors_and_distances[:k]]
        nearest_neighbors[index] = closest_points

    return nearest_neighbors


def fast_knn(datapoints, k=3):
    """ KNN implemented in numpy with vectorized operations """
    all_pairwise_distances = datapoints[:, np.newaxis] - datapoints[np.newaxis]
    l2_distance = np.sqrt((all_pairwise_distances ** 2).sum(axis=2))
    neighbors = l2_distance.argpartition(k+1, axis=1)[:, :k+1]  # Each point's nearest neighbor will be itself
    return {
        index: list(filter(lambda neighbor_index: neighbor_index != index, neighbors))
        for index, neighbors in enumerate(neighbors)
    }


def _l2_distance(point_1, point_2):
    return np.sqrt(np.sum((point_2 - point_1) ** 2))
