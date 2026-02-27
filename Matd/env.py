import numpy as np
import random


class UUV:
    def __init__(self, width, height):
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)


class Target:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)

    def move(self):
        drift = np.random.uniform(-0.8, 0.8, size=2)
        self.pos += drift
        self.pos = np.clip(self.pos, [0, 0], [self.width, self.height])


class MultiAgentEnv:
    def __init__(self, width=900, height=600, n_uuv=3, n_targets=2):
        self.width = width
        self.height = height
        self.n_uuv = n_uuv
        self.n_targets = n_targets
        self.catch_radius = 25
        self.reset()

    def reset(self):
        self.uuvs = [UUV(self.width, self.height) for _ in range(self.n_uuv)]
        self.targets = [Target(self.width, self.height) for _ in range(self.n_targets)]

        # FIXED ASSIGNMENT PER EPISODE
        self.assignments = self.assign_targets()

        return self._get_obs()

    def assign_targets(self):
        assignments = {}
        taken = set()

        for i, uuv in enumerate(self.uuvs):
            distances = []
            for j, target in enumerate(self.targets):
                if j in taken:
                    distances.append(np.inf)
                else:
                    distances.append(np.linalg.norm(uuv.pos - target.pos))

            chosen = np.argmin(distances)
            assignments[i] = chosen
            taken.add(chosen)

        return assignments

    def step(self, actions):

        rewards = []
        done = False

        # Move targets
        for t in self.targets:
            t.move()

        for i, uuv in enumerate(self.uuvs):

            uuv.pos += actions[i] * 8.0
            uuv.pos = np.clip(uuv.pos, [0, 0], [self.width, self.height])

            target = self.targets[self.assignments[i]]
            dist = np.linalg.norm(uuv.pos - target.pos)

            max_distance = np.sqrt(self.width**2 + self.height**2)

            # CLEAN REWARD
            reward = 1.0 - (dist / max_distance)

            if dist < self.catch_radius:
                reward += 3.0
                done = True

            rewards.append(reward)

        return self._get_obs(), rewards, done

    def _get_obs(self):

        obs = []

        target_positions = []
        for t in self.targets:
            target_positions.extend(
                t.pos / np.array([self.width, self.height])
            )

        for uuv in self.uuvs:
            uuv_pos = uuv.pos / np.array([self.width, self.height])
            state = np.concatenate([uuv_pos, target_positions])
            obs.append(state)

        return obs