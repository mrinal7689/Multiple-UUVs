import numpy as np
from settings import SEPARATION_DISTANCE


def separation_force(world, hunter):

    force = np.zeros(2)

    for other in world.hunters:
        if other is hunter:
            continue

        dist = np.linalg.norm(hunter.pos - other.pos)

        if dist < SEPARATION_DISTANCE and dist > 0:
            force += (hunter.pos - other.pos) / dist

    return force
