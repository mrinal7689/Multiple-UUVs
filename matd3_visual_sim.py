import pygame
import numpy as np
import random

# ===============================
# SETTINGS
# ===============================

WIDTH, HEIGHT = 900, 600
FPS = 60

NUM_HUNTERS = 3
NUM_TARGETS = 2

HUNTER_COLOR = (0, 255, 0)
TARGET_COLOR = (255, 60, 60)

HUNTER_SIZE = 10
TARGET_SIZE = 8

HUNTER_SPEED = 3.5
TARGET_SPEED = 2.5

SEPARATION_DISTANCE = 40


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
        self.pos += self.vel

        # bounce off walls
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
        dists = [np.linalg.norm(t.pos - hunter.pos) for t in self.targets]
        return self.targets[np.argmin(dists)]

    # ===============================
    # SEPARATION FORCE (avoid clustering)
    # ===============================
    def separation_force(self, hunter):

        force = np.zeros(2)

        for other in self.hunters:
            if other is hunter:
                continue

            dist = np.linalg.norm(hunter.pos - other.pos)

            if dist < SEPARATION_DISTANCE and dist > 0:
                force += (hunter.pos - other.pos) / dist

        return force

    def step(self):

        # ===== TARGETS MOVE SMOOTHLY =====
        for t in self.targets:

            # small random steering change
            t.vel += np.random.uniform(-0.2, 0.2, 2)

            speed = np.linalg.norm(t.vel)
            if speed > TARGET_SPEED:
                t.vel = (t.vel / speed) * TARGET_SPEED

            t.move()

        # ===== HUNTERS CHASE + SEPARATE =====
        for h in self.hunters:

            target = self.nearest_target(h)

            pursue = target.pos - h.pos
            dist = np.linalg.norm(pursue)

            if dist > 0:
                pursue = pursue / dist

            sep = self.separation_force(h)

            direction = pursue + sep * 1.2

            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            h.vel = direction * HUNTER_SPEED
            h.move()

        # ===== CAPTURE =====
        for h in self.hunters:
            for t in self.targets:

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

    def draw(self, screen):

        for h in self.hunters:
            h.draw(screen)

        for t in self.targets:
            t.draw(screen)


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
