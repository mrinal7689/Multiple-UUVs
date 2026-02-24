import numpy as np
import pygame
from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3
from Matd.matd3.replay_buffer import ReplayBuffer


# ===============================
# CONFIG
# ===============================
NUM_AGENTS = 3
STATE_DIM = 2
ACTION_DIM = 2

EPISODES = 200
MAX_STEPS = 200
BATCH_SIZE = 64


# ===============================
# INIT
# ===============================
env = MultiAgentEnv(n_agents=NUM_AGENTS)
agent = MATD3(NUM_AGENTS, STATE_DIM, ACTION_DIM)
buffer = ReplayBuffer()

pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("MATD3 Multi-Agent Training")
clock = pygame.time.Clock()

print("Training with live visualization...\n")


# ===============================
# TRAIN LOOP
# ===============================
for ep in range(EPISODES):

    obs = env.reset()
    joint_state = np.concatenate(obs)

    episode_reward = 0

    for step in range(MAX_STEPS):

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # ---- Select actions ----
        actions = []
        for i in range(NUM_AGENTS):
            action = agent.select_action(obs[i])
            actions.append(action)

        # ---- Environment step ----
        next_obs, rewards, done = env.step(actions)

        joint_action = np.concatenate(actions)
        next_joint_state = np.concatenate(next_obs)

        reward = np.mean(rewards)
        episode_reward += reward

        buffer.add((joint_state, joint_action, reward, next_joint_state))
        agent.train(buffer, batch_size=BATCH_SIZE)

        obs = next_obs
        joint_state = next_joint_state
        # RENDER
        # ===============================
        screen.fill((10, 10, 30))

        # Draw fish (yellow)
        for f in env.fish:
            pygame.draw.circle(screen, (255, 200, 0), (int(f.pos[0]), int(f.pos[1])), 6)

        # Draw hunters (blue)
        for hunter in env.hunters:
            pygame.draw.circle(screen, (0, 200, 255), (int(hunter.pos[0]), int(hunter.pos[1])), 10)

pygame.display.flip()
clock.tick(60)