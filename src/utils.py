import numpy as np


def compute_length(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    return total_length


def distance_euc(zi, zj):
    xi, xj = zi[0], zj[0]
    yi, yj = zi[1], zj[1]
    return round(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 0)
