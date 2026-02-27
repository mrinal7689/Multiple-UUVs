import random
import numpy as np
import pygame

from entity import Entity
from settings import *
from auction import assign_targets_auction


class MultiTargetWorld:
    def __init__(self):
        self.time_step = 0
        self.spawn_counter = 0
        self.captures = 0

        self.hunters = [
            Entity(
                random.randint(SPAWN_MARGIN, WIDTH - SPAWN_MARGIN),
                random.randint(SPAWN_MARGIN, HEIGHT - SPAWN_MARGIN),
                HUNTER_COLOR,
                HUNTER_SIZE,
            )
            for _ in range(NUM_HUNTERS)
        ]

        self.targets = []

    def _spawn_target(self):
        spawn_count = random.randint(3, 5)
        for _ in range(spawn_count):
            if len(self.targets) < MAX_ACTIVE_TARGETS:
                t = Entity(
                    random.randint(SPAWN_MARGIN, WIDTH - SPAWN_MARGIN),
                    random.randint(SPAWN_MARGIN, HEIGHT - SPAWN_MARGIN),
                    TARGET_COLOR,
                    TARGET_SIZE,
                )
                self.targets.append(t)

    def _get_ocean_flow(self, x, y):
        t = self.time_step * OCEAN_FLOW_FREQUENCY

        flow_x = (
            np.sin(x / OCEAN_FLOW_SCALE + t) * 0.5 +
            np.sin(y / OCEAN_FLOW_SCALE * 0.5) * 0.3
        )
        flow_y = (
            np.cos(y / OCEAN_FLOW_SCALE + t) * 0.5 +
            np.cos(x / OCEAN_FLOW_SCALE * 0.5) * 0.3
        )

        if OCEAN_VORTEX_ENABLED:
            center_x = WIDTH / 2
            center_y = HEIGHT / 2
            dx = x - center_x
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2) + 1
            vortex_x = -dy / (dist**2) * 0.3
            vortex_y = dx / (dist**2) * 0.3
            flow_x += vortex_x
            flow_y += vortex_y

        return np.array([flow_x, flow_y]) * OCEAN_FLOW_STRENGTH

    def separation_force(self, hunter):
        force = np.zeros(2)
        for other in self.hunters:
            if other is hunter:
                continue
            dist = np.linalg.norm(hunter.pos - other.pos)
            if dist < SEPARATION_DISTANCE and dist > 0:
                force += ((hunter.pos - other.pos) / dist) * (1 / dist)
        return force

    def _update_targets(self):
        for t in self.targets:
            if random.random() < 0.08:
                angle = random.uniform(0, 2 * np.pi)
                burst_speed = random.uniform(1.5, TARGET_SPEED * 1.8)
                t.vel = np.array([
                    np.cos(angle) * burst_speed,
                    np.sin(angle) * burst_speed
                ])

            flow = self._get_ocean_flow(t.pos[0], t.pos[1])
            t.vel += flow * 0.2

            speed = np.linalg.norm(t.vel)
            if speed > TARGET_SPEED * 1.8:
                t.vel = (t.vel / speed) * TARGET_SPEED * 1.8

            t.move()

    def _update_hunters(self, assignments):
        for h, target in assignments.items():
            pursue = target.pos - h.pos
            dist = np.linalg.norm(pursue)
            if dist > 0:
                pursue /= dist

            sep = self.separation_force(h)
            direction = pursue + sep * SEPARATION_WEIGHT

            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm

            h.vel = direction * HUNTER_SPEED
            h.move()

    def _check_captures(self):
        for h in self.hunters:
            for t in self.targets[:]:
                if np.linalg.norm(h.pos - t.pos) < CAPTURE_DISTANCE:
                    self.targets.remove(t)
                    self.captures += 1

    def step(self):
        self.time_step += 1
        self.spawn_counter += 1

        if self.spawn_counter >= SPAWN_INTERVAL:
            self.spawn_counter = 0
            self._spawn_target()

        while len(self.targets) < MIN_ACTIVE_TARGETS:
            self._spawn_target()

        self._update_targets()

        if self.targets:
            assignments = assign_targets_auction(self.hunters, self.targets)
            self._update_hunters(assignments)

        self._check_captures()

        return self.captures < MAX_TARGETS

    def draw(self, screen):
        for y in range(HEIGHT):
            ratio = y / HEIGHT
            r = int(OCEAN_COLOR_TOP[0] * (1 - ratio) + OCEAN_COLOR_BOTTOM[0] * ratio)
            g = int(OCEAN_COLOR_TOP[1] * (1 - ratio) + OCEAN_COLOR_BOTTOM[1] * ratio)
            b = int(OCEAN_COLOR_TOP[2] * (1 - ratio) + OCEAN_COLOR_BOTTOM[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))

        for h in self.hunters:
            h.draw(screen)
        for t in self.targets:
            t.draw(screen)