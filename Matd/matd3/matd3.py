import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Matd.matd3.actor import Actor
from Matd.matd3.critic import Critic


class MATD3:
    def __init__(self, num_agents, state_dim, action_dim):

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.total_state_dim = num_agents * state_dim
        self.total_action_dim = num_agents * action_dim

        self.max_action = 1.0

        self.actor = Actor(state_dim, action_dim, self.max_action)
        self.actor_target = Actor(state_dim, action_dim, self.max_action)

        self.critic1 = Critic(self.total_state_dim, self.total_action_dim)
        self.critic2 = Critic(self.total_state_dim, self.total_action_dim)

        self.critic1_target = Critic(self.total_state_dim, self.total_action_dim)
        self.critic2_target = Critic(self.total_state_dim, self.total_action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2

        self.total_it = 0
    def select_action(self, state, explore=False, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]

        if explore:
            noise = noise_scale * np.random.randn(self.action_dim)
            action = action + noise

        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=64):

        if len(replay_buffer.storage) < batch_size:
            return

        self.total_it += 1

        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        noise = (
            torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_actions = []

        for i in range(self.num_agents):
            agent_state = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
            next_action = self.actor_target(agent_state)
            next_actions.append(next_action)

        next_actions = torch.cat(next_actions, dim=1)
        next_actions = (next_actions + noise).clamp(-1, 1)

        target_Q1 = self.critic1_target(next_states, next_actions)
        target_Q2 = self.critic2_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + self.gamma * target_Q.detach()

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        loss_Q1 = nn.MSELoss()(current_Q1, target_Q)
        loss_Q2 = nn.MSELoss()(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_delay == 0:

            actor_actions = []

            for i in range(self.num_agents):
                agent_state = states[:, i*self.state_dim:(i+1)*self.state_dim]
                action = self.actor(agent_state)
                actor_actions.append(action)

            actor_actions = torch.cat(actor_actions, dim=1)

            actor_loss = -self.critic1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )