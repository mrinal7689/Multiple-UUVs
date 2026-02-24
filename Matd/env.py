import numpy as np
import random


class Hunter:
    def __init__(self, width, height):
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)


class Fish:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)

    def move(self):
        # Random ocean-like motion
        movement = np.random.uniform(-2, 2, size=2)
        self.pos += movement
        self.pos = np.clip(self.pos, [0, 0], [self.width, self.height])


class MultiAgentEnv:
    def __init__(self, width=900, height=600, n_agents=3, n_fish=2):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_fish = n_fish
        self.reset()

    def reset(self):
        self.hunters = [Hunter(self.width, self.height) for _ in range(self.n_agents)]
        self.fish = [Fish(self.width, self.height) for _ in range(self.n_fish)]
        return self._get_obs()

    # ===============================
    # AUCTION SYSTEM
    # ===============================
    def assign_targets(self):
        assignments = {}

        for i, hunter in enumerate(self.hunters):
            distances = [
                np.linalg.norm(hunter.pos - f.pos) for f in self.fish
            ]
            target_id = np.argmin(distances)
            assignments[i] = target_id

        return assignments

    def step(self, actions):

        rewards = []

        # Move fish first
        for f in self.fish:
            f.move()

        # Auction assignment
        assignments = self.assign_targets()

        for i, hunter in enumerate(self.hunters):

            # Apply RL movement
            hunter.pos += actions[i] * 5.0
            hunter.pos = np.clip(hunter.pos, [0, 0], [self.width, self.height])

            # Assigned fish
            target = self.fish[assignments[i]]

            dist = np.linalg.norm(hunter.pos - target.pos)

            reward = -dist

            # Capture bonus
            if dist < 15:
                reward += 100

            rewards.append(reward)

        return self._get_obs(), rewards, False

    def _get_obs(self):
        obs = []
        for hunter in self.hunters:
            obs.append(hunter.pos / np.array([self.width, self.height]))
        return obs