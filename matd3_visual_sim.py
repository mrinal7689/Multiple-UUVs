import pygame
import numpy as np
import random

from settings import (
    WIDTH,
    HEIGHT,
    FPS,
    NUM_HUNTERS,
    NUM_TARGETS,
    HUNTER_COLOR,
    TARGET_COLOR,
    HUNTER_SIZE,
    TARGET_SIZE,
    HUNTER_SPEED,
    TARGET_SPEED,
    SEPARATION_DISTANCE,
)

from entity_move import entity_move
from entity_draw import entity_draw
from world_nearest_target import nearest_target as nearest_target_fn
from world_separation_force import separation_force as separation_force_fn
from world_step import world_step as world_step_fn
from world_draw import draw_world as draw_world_fn


# ===============================
# ENTITY
# ===============================

class Entity:
    def __init__(self, x, y, color, size):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.random.uniform(-1, 1, 2)
        self.color = color
        self.size = size
    def move(self):
        entity_move(self)

    def draw(self, screen):
        entity_draw(self, screen)


# ===============================
# WORLD
# ===============================

class MultiTargetWorld:

    def __init__(self):

        self.hunters = [
            Entity(
                random.randint(50, WIDTH - 50),
                random.randint(50, HEIGHT - 50),
                HUNTER_COLOR,
                HUNTER_SIZE,
            )
            for _ in range(NUM_HUNTERS)
        ]

        self.targets = [
            Entity(
                random.randint(50, WIDTH - 50),
                random.randint(50, HEIGHT - 50),
                TARGET_COLOR,
                TARGET_SIZE,
            )
            for _ in range(NUM_TARGETS)
        ]

    # ===============================
    # FIND NEAREST TARGET
    # ===============================
    def nearest_target(self, hunter):
        return nearest_target_fn(self, hunter)

    # ===============================
    # SEPARATION FORCE (avoid clustering)
    # ===============================
    def separation_force(self, hunter):
        return separation_force_fn(self, hunter)

    def step(self):
        world_step_fn(self)

    def draw(self, screen):
        draw_world_fn(self, screen)


# ===============================
# MAIN LOOP
# ===============================

def main():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Improved Hunters vs Targets Simulation")

    clock = pygame.time.Clock()

    world = MultiTargetWorld()

    running = True

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step()

        screen.fill((20, 20, 25))
        world.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
