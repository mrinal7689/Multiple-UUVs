import numpy as np
from settings import WIDTH, HEIGHT


def entity_move(entity):
    entity.pos += entity.vel

    # bounce off walls
    if entity.pos[0] <= 0 or entity.pos[0] >= WIDTH:
        entity.vel[0] *= -1
    if entity.pos[1] <= 0 or entity.pos[1] >= HEIGHT:
        entity.vel[1] *= -1

    entity.pos[0] = np.clip(entity.pos[0], 0, WIDTH)
    entity.pos[1] = np.clip(entity.pos[1], 0, HEIGHT)
