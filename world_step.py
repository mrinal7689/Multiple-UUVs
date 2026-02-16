import numpy as np
import random
from world_nearest_target import nearest_target
from world_separation_force import separation_force
from settings import TARGET_SPEED, HUNTER_SPEED, WIDTH, HEIGHT


def world_step(world):

    # ===== TARGETS MOVE SMOOTHLY =====
    for t in world.targets:

        # small random steering change
        t.vel += np.random.uniform(-0.2, 0.2, 2)

        speed = np.linalg.norm(t.vel)
        if speed > TARGET_SPEED:
            t.vel = (t.vel / speed) * TARGET_SPEED

        # call entity move (assumes Entity.move delegates to entity_move)
        t.move()

    # ===== HUNTERS CHASE + SEPARATE =====
    for h in world.hunters:

        target = nearest_target(world, h)

        pursue = target.pos - h.pos
        dist = np.linalg.norm(pursue)

        if dist > 0:
            pursue = pursue / dist

        sep = separation_force(world, h)

        direction = pursue + sep * 1.2

        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        h.vel = direction * HUNTER_SPEED
        h.move()

    # ===== CAPTURE =====
    for h in world.hunters:
        for t in world.targets:

            dist = np.linalg.norm(h.pos - t.pos)

            if dist < 15:
                t.pos = np.array(
                    [
                        random.randint(50, WIDTH - 50),
                        random.randint(50, HEIGHT - 50),
                    ],
                    dtype=np.float32,
                )
                t.vel = np.random.uniform(-1, 1, 2)
