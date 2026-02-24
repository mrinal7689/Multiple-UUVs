import pygame
import torch
import numpy as np
from Matd.env import MultiUUVEnv
from Matd.matd3.actor import Actor

WIDTH = 900
HEIGHT = 600

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = MultiUUVEnv()
actor = Actor(env.state_dim, env.action_dim)
actor.load_state_dict(torch.load("matd3_actor.pth"))
actor.eval()

states = env.reset()

running = True
while running:
    clock.tick(60)
    screen.fill((20, 20, 40))

    actions = []
    for i in range(env.num_agents):
        state = torch.FloatTensor(states[i])
        action = actor(state).detach().numpy()
        actions.append(action)

    actions = np.array(actions)
    states, rewards, done = env.step(actions)

    # Draw hunters
    for pos in env.hunters:
        pygame.draw.circle(screen, (0, 255, 0), pos.astype(int), 8)

    # Draw targets
    for pos in env.targets:
        pygame.draw.circle(screen, (255, 80, 80), pos.astype(int), 6)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()