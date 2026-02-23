import numpy as np


def nearest_target(world, hunter):
    dists = [np.linalg.norm(t.pos - hunter.pos) for t in world.targets]
    return world.targets[np.argmin(dists)]
