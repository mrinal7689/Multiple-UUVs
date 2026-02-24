import pygame
import numpy as np
from settings import WIDTH, HEIGHT, INITIAL_VELOCITY_RANGE


class Entity:
    def __init__(self, x, y, color, size):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.random.uniform(
            INITIAL_VELOCITY_RANGE[0],
            INITIAL_VELOCITY_RANGE[1],
            2,
        )
        self.color = color
        self.size = size

    def move(self):
        self.pos += self.vel

        if self.pos[0] <= 0 or self.pos[0] >= WIDTH:
            self.vel[0] *= -1
        if self.pos[1] <= 0 or self.pos[1] >= HEIGHT:
            self.vel[1] *= -1

        self.pos[0] = np.clip(self.pos[0], 0, WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, HEIGHT)

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos[0]), int(self.pos[1])),
            self.size,
        )