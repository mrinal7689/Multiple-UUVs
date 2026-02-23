import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


# ==============================
# ENVIRONMENT
# ==============================
class TargetHuntingParallelEnv(ParallelEnv):

    metadata = {
        "name": "target_hunting_v0",
        "render_modes": ["human"],
        "is_parallelizable": True,
    }

    def __init__(self, n_hunters=4, max_steps=200):
        super().__init__()

        self.n_hunters = n_hunters
        self.max_steps = max_steps

        self.possible_agents = [f"hunter_{i}" for i in range(n_hunters)] + ["target"]
        self.agents = []

        self._action_spaces = {}
        self._observation_spaces = {}

        hunter_obs_dim = 10
        target_obs_dim = 10

        for i in range(n_hunters):
            a = f"hunter_{i}"
            self._action_spaces[a] = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
            self._observation_spaces[a] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(hunter_obs_dim,), dtype=np.float32
            )

        self._action_spaces["target"] = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._observation_spaces["target"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_dim,), dtype=np.float32
        )

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.step_count = 0

        observations = {
            a: np.zeros(self.observation_space(a).shape, dtype=np.float32)
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.step_count += 1

        observations = {
            a: np.random.randn(*self.observation_space(a).shape).astype(np.float32)
            for a in self.agents
        }

        rewards = {a: np.random.rand() for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: self.step_count >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos


# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":

    env = TargetHuntingParallelEnv()

    obs, infos = env.reset()

    print("Environment running...\n")

    for step in range(5):
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        print(f"Step {step+1}")
        print("Rewards:", rewards)
        print("-" * 30)
