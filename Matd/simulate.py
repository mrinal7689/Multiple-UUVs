import numpy as np
import pygame
import torch

from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3


# ===============================
# CONFIG
# ===============================
NUM_AGENTS = 3
N_TARGETS = 2
STATE_DIM = 2 + (2 * N_TARGETS)
ACTION_DIM = 2

WIDTH = 900
HEIGHT = 600


# ===============================
# LOAD ENV + MODEL
# ===============================
env = MultiAgentEnv(width=WIDTH, height=HEIGHT,
                    n_uuv=NUM_AGENTS,
                    n_targets=N_TARGETS)

agent = MATD3(NUM_AGENTS, STATE_DIM, ACTION_DIM)
agent.actor.load_state_dict(torch.load("matd3_actor.pth"))
agent.actor.eval()


# ===============================
# PYGAME SETUP
# ===============================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-UUV Hunting Simulation")
clock = pygame.time.Clock()

obs = env.reset()
running = True

print("Simulation started...")

while running:

    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # -------- POLICY (NO NOISE) --------
    actions = []
    for i in range(NUM_AGENTS):
        action = agent.select_action(obs[i], explore=False)
        actions.append(action)

    obs, rewards, done = env.step(actions)

    # -------- RENDER --------
    screen.fill((15, 20, 40))

    # Draw targets (red)
    for target in env.targets:
        pygame.draw.circle(screen, (255, 70, 70),
                           (int(target.pos[0]), int(target.pos[1])), 10)

    # Draw hunters (green)
    for uuv in env.uuvs:
        pygame.draw.circle(screen, (0, 255, 120),
                           (int(uuv.pos[0]), int(uuv.pos[1])), 12)

    pygame.display.update()

    # Reset episode when done
    if done:
        obs = env.reset()

pygame.quit()
print("Simulation ended.")