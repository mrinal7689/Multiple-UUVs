import numpy as np
import torch
from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3
from Matd.matd3.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

NUM_AGENTS = 3
N_TARGETS = 2

# Enforce constraints
assert NUM_AGENTS >= 3, "Must have at least 3 hunters (UUVs)"
assert N_TARGETS >= 1, "Must have at least 1 target"

STATE_DIM = 2 + (2 * N_TARGETS)
ACTION_DIM = 2

EPISODES = 10
MAX_STEPS = 150
BATCH_SIZE = 64
TRAIN_ITERATIONS = 5
WARMUP_STEPS = 1500  # Proportional to 1000 episodes

env = MultiAgentEnv(n_uuv=NUM_AGENTS, n_targets=N_TARGETS)
agent = MATD3(NUM_AGENTS, STATE_DIM, ACTION_DIM)
buffer = ReplayBuffer()

print("Training started...\n")
print(f"Total Episodes: {EPISODES}")
print(f"Warmup Steps: {WARMUP_STEPS}")
print(f"Training Iterations per Step: {TRAIN_ITERATIONS}\n")

episode_rewards = []
eval_rewards = []  # store eval rewards for plotting
eval_episodes_idx = []  # store which episode each eval corresponds to
total_steps = 0
max_eval_reward = -np.inf  # Track best eval performance

EVAL_EPISODES = 5

for ep in range(EPISODES):

    obs = env.reset()
    joint_state = np.concatenate(obs)
    episode_reward = 0

    # Gradually decay exploration noise over time
    exploration_noise = max(0.01, 0.35 * (1 - ep / EPISODES))
    
    # Decay learning rate over time for fine-tuning
    if ep > WARMUP_STEPS / MAX_STEPS:  # After warmup
        lr_decay = 1.0 - (ep / EPISODES) * 0.5  # Decay to 50% of initial LR
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = 3e-4 * lr_decay
        for param_group in agent.critic1_optimizer.param_groups:
            param_group['lr'] = 3e-4 * lr_decay
        for param_group in agent.critic2_optimizer.param_groups:
            param_group['lr'] = 3e-4 * lr_decay

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

        # Store transition with terminal flag
        buffer.add((joint_state, joint_action, reward, next_joint_state, done))

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

    if (ep + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(
            f"Episode {ep+1:4d}/{EPISODES} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg (last 10): {avg_reward:8.2f}"
        )

    # Periodic evaluation without exploration noise (every 50 episodes)
    if (ep + 1) % 50 == 0:
        eval_returns = []
        for _ in range(EVAL_EPISODES):
            obs_eval = env.reset()
            episode_eval_reward = 0.0
            for _ in range(MAX_STEPS):
                actions_eval = []
                for i in range(NUM_AGENTS):
                    action_eval = agent.select_action(obs_eval[i], explore=False)
                    actions_eval.append(action_eval)
                next_obs_eval, rewards_eval, done_eval = env.step(actions_eval)
                episode_eval_reward += np.mean(rewards_eval)
                obs_eval = next_obs_eval
                if done_eval:
                    break
            eval_returns.append(episode_eval_reward)

        eval_mean = np.mean(eval_returns)
        eval_std = np.std(eval_returns)
        eval_rewards.append(eval_mean)
        eval_episodes_idx.append(ep + 1)
        
        # Track improvement
        improved = "↑" if eval_mean > max_eval_reward else "↓"
        max_eval_reward = max(max_eval_reward, eval_mean)

        print(
            f"[EVAL] Episode {ep+1:4d}/{EPISODES} | "
            f"Avg: {eval_mean:7.2f} ± {eval_std:5.2f} | "
            f"Best: {max_eval_reward:7.2f} {improved}"
        )

torch.save(agent.actor.state_dict(), "matd3_actor.pth")

print("\n" + "="*60)
print("Training Completed and Model Saved.")
print("="*60)
print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
print(f"Best Episode Reward: {max(episode_rewards):.2f}")
print(f"Average Last 10 Episodes: {np.mean(episode_rewards[-10:]):.2f}")
if eval_rewards:
    print(f"Best Eval Reward (no noise): {max(eval_rewards):.2f}")
    print(f"Final Eval Reward: {eval_rewards[-1]:.2f}")
print("="*60)

plt.figure()
plt.plot(episode_rewards, label="Train (with noise)")
if len(eval_rewards) > 0:
    plt.plot(eval_episodes_idx, eval_rewards, "ro-", label="Eval (no noise)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training and Eval Reward per Episode")
plt.legend()
plt.show()