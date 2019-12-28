import numpy as np
import timeit


def generate_datapoints(n, dim=2, random_state=1):
    np.random.seed(random_state)
    return np.random.randn(n, dim)


def slow_knn(n, k=3):
    """ KNN implemented in python """
    if k > n:
        raise ValueError('k must be less than or equal to n')

    datapoints = generate_datapoints(n)
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


def _l2_distance(point_1, point_2):
    return np.sqrt(np.sum((point_2 - point_1) ** 2))
