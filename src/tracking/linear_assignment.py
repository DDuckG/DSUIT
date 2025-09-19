import numpy as np
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix):

    cost = np.array(cost_matrix, dtype = float)
    large_cost = 1000000
    bad = ~np.isfinite(cost)
    if bad.any():
        cost[bad] = large_cost

    cost[np.isnan(cost)] = large_cost
    cost[np.isinf(cost)] = large_cost

    if cost.size == 0:
        return np.array([], dtype = int), np.array([], dtype = int)

    if linear_sum_assignment is None:
        # naive fallback greedy
        assigned = []
        cost = cost_matrix.copy()
        while True:
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            i, j = idx
            if cost[i, j] >= large_cost:
                break
            assigned.append((i,j))
            cost[i,:] = large_cost
            cost[:,j] = large_cost
            if np.all(cost >= large_cost):
                break
        return np.array([a[0] for a in assigned], dtype = int), np.array([a[1] for a in assigned], dtype = int)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
