import numpy as np
import pygame
import torch

from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3


NUM_AGENTS = 3
STATE_DIM = 2
ACTION_DIM = 2

# Load environment
env = MultiAgentEnv(n_agents=NUM_AGENTS, n_fish=2)

# Load trained agent
agent = MATD3(NUM_AGENTS, STATE_DIM, ACTION_DIM)
agent.actor.load_state_dict(torch.load("matd3_actor.pth"))
agent.actor.eval()

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("UUV Fish Hunting Simulation")
clock = pygame.time.Clock()

obs = env.reset()
running = True

print("Simulation started...")

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    actions = []
    for i in range(NUM_AGENTS):
        action = agent.select_action(obs[i])
        actions.append(action)

    obs, rewards, done = env.step(actions)

    # -------- RENDER --------
    screen.fill((10, 10, 40))

    # Draw fish (yellow)
    for f in env.fish:
        pygame.draw.circle(screen, (255, 200, 0),
                           (int(f.pos[0]), int(f.pos[1])), 6)

    # Draw hunters (blue)
    for hunter in env.hunters:
        pygame.draw.circle(screen, (0, 180, 255),
                           (int(hunter.pos[0]), int(hunter.pos[1])), 10)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
print("Simulation Ended.")