import numpy as np
import torch
from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3
from Matd.matd3.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

NUM_AGENTS = 3
N_TARGETS = 2

STATE_DIM = 2 + (2 * N_TARGETS)
ACTION_DIM = 2

EPISODES = 3000
MAX_STEPS = 200
BATCH_SIZE = 64
TRAIN_ITERATIONS = 2
WARMUP_STEPS = 10000

env = MultiAgentEnv(n_uuv=NUM_AGENTS, n_targets=N_TARGETS)
agent = MATD3(NUM_AGENTS, STATE_DIM, ACTION_DIM)
buffer = ReplayBuffer()

print("Training started...\n")

episode_rewards = []
total_steps = 0

for ep in range(EPISODES):

    obs = env.reset()
    joint_state = np.concatenate(obs)
    episode_reward = 0

    exploration_noise = max(0.05, 0.3 * (1 - ep / EPISODES))

    for step in range(MAX_STEPS):

        actions = []

        for i in range(NUM_AGENTS):

            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(-1, 1, ACTION_DIM)
            else:
                action = agent.select_action(
                    obs[i],
                    explore=True,
                    noise_scale=exploration_noise
                )

            actions.append(action)

        next_obs, rewards, done = env.step(actions)

        joint_action = np.concatenate(actions)
        next_joint_state = np.concatenate(next_obs)

        # CENTRALIZED REWARD
        reward = np.mean(rewards)

        buffer.add((joint_state, joint_action, reward, next_joint_state))

        episode_reward += reward
        total_steps += 1

        # TRAIN ONLY AFTER WARMUP
        if total_steps > WARMUP_STEPS:
            for _ in range(TRAIN_ITERATIONS):
                agent.train(buffer, batch_size=BATCH_SIZE)

        obs = next_obs
        joint_state = next_joint_state

        if done:
            break

    episode_rewards.append(episode_reward)

    if (ep + 1) % 5 == 0:
        avg_reward = np.mean(episode_rewards[-5:])
        print(
            f"Episode {ep+1}/{EPISODES} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg (last 5): {avg_reward:8.2f}"
        )

torch.save(agent.actor.state_dict(), "matd3_actor.pth")

print("\nTraining Completed and Model Saved.")
print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
print(f"Best Episode Reward: {max(episode_rewards):.2f}")
print(f"Average Last 5 Episodes: {np.mean(episode_rewards[-5:]):.2f}")

plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward per Episode")
plt.show()