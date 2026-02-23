import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from pettingzoo import ParallelEnv
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================
# ENVIRONMENT (MULTIPLE HUNTERS + MULTIPLE TARGETS)
# ===================================================

class MultiTargetEnv(ParallelEnv):

    metadata = {"name": "multi_target_v0"}

    def __init__(self, n_hunters=3, n_targets=2, max_steps=100):
        super().__init__()

        self.n_hunters = n_hunters
        self.n_targets = n_targets
        self.max_steps = max_steps

        self.possible_agents = (
            [f"hunter_{i}" for i in range(n_hunters)] +
            [f"target_{i}" for i in range(n_targets)]
        )

        self.agents = []
        self.obs_dim = 12
        self.act_dim = 2

        self._obs_spaces = {
            a: spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
            for a in self.possible_agents
        }

        self._act_spaces = {
            a: spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
            for a in self.possible_agents
        }

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.step_count = 0

        obs = {
            a: np.random.randn(self.obs_dim).astype(np.float32)
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):

        self.step_count += 1

        obs = {
            a: np.random.randn(self.obs_dim).astype(np.float32)
            for a in self.agents
        }

        rewards = {}
        for a in self.agents:
            if "hunter" in a:
                rewards[a] = np.random.uniform(0, 1)
            else:
                rewards[a] = np.random.uniform(-1, 0)

        terms = {a: False for a in self.agents}
        trunc = {a: self.step_count >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, rewards, terms, trunc, infos


# ===================================================
# MATD3 NETWORKS
# ===================================================

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# ===================================================
# REPLAY BUFFER
# ===================================================

class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = []
        self.max_size = size

    def add(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ===================================================
# MATD3 AGENT
# ===================================================

class MATD3Agent:

    def __init__(self, obs_dim, act_dim):

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_target = Actor(obs_dim, act_dim).to(device)

        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)

        self.critic1_target = Critic(obs_dim, act_dim).to(device)
        self.critic2_target = Critic(obs_dim, act_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), 1e-3)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), 1e-3)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), 1e-3)

        self.gamma = 0.99
        self.tau = 0.005

        self.update_targets(1.0)

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        for target, src in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(tau * src.data + (1 - tau) * target.data)

        for target, src in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target.data.copy_(tau * src.data + (1 - tau) * target.data)

        for target, src in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target.data.copy_(tau * src.data + (1 - tau) * target.data)

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return action


# ===================================================
# TRAIN LOOP
# ===================================================

def train():

    env = MultiTargetEnv()
    buffer = ReplayBuffer()

    agents = {
        a: MATD3Agent(env.obs_dim, env.act_dim)
        for a in env.possible_agents
    }

    episodes = 20

    for ep in range(episodes):

        obs, _ = env.reset()
        total_reward = 0

        for step in range(env.max_steps):

            actions = {
                a: agents[a].act(obs[a])
                for a in env.agents
            }

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            for a in env.agents:
                buffer.add(
                    (obs[a], actions[a], rewards[a], next_obs[a])
                )

            obs = next_obs
            total_reward += sum(rewards.values())

        print(f"Episode {ep+1} | Total Reward: {total_reward:.2f}")

    print("\nTraining loop finished successfully ✅")


# ===================================================
# RUN
# ===================================================

if __name__ == "__main__":
    train()
