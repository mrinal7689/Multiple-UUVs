import numpy as np


def assign_targets_auction(hunters, targets):
    hunters_copy = hunters.copy()
    targets_copy = targets.copy()
    assignments = {}

    while hunters_copy and targets_copy:
        best_pair = None
        best_cost = float("inf")

        for h in hunters_copy:
            for t in targets_copy:
                cost = np.linalg.norm(h.pos - t.pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (h, t)

        if best_pair:
            h, t = best_pair
            assignments[h] = t
            hunters_copy.remove(h)
            targets_copy.remove(t)

    return assignments