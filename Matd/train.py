import argparse
import random

import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3
from Matd.matd3.replay_buffer import ReplayBuffer


def _parse_args():
    p = argparse.ArgumentParser(description="Train MATD3 for multi-UUV target catching (no visualization).")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--num-targets", type=int, default=4)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-iters", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-best", type=str, default="matd3_actor_best.pth")
    p.add_argument("--save-final", type=str, default="matd3_actor.pth")
    p.add_argument("--world-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uuv-speed", type=float, default=20.0)
    p.add_argument("--catch-radius", type=float, default=20.0)
    p.add_argument("--target-speed", type=float, default=3.5)
    p.add_argument("--target-min-speed", type=float, default=1.2)
    p.add_argument("--target-burst-prob", type=float, default=0.08)
    return p.parse_args()


def main():
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.num_agents >= 3, "Must have at least 3 hunters (UUVs)"
    assert args.num_targets >= 1, "Must have at least 1 target"

    state_dim = 2 + (2 * args.num_targets)
    action_dim = 2

    env = MultiAgentEnv(
        n_uuv=args.num_agents,
        n_targets=args.num_targets,
        world_mode=args.world_mode,
        respawn_on_capture=True,
        # Keep target count stable for a fixed-size observation
        spawn_interval=10**9,
        max_active_targets=args.num_targets,
        min_active_targets=args.num_targets,
        uuv_speed=args.uuv_speed,
        catch_radius=args.catch_radius,
        target_speed=args.target_speed,
        target_min_speed=args.target_min_speed,
        target_burst_prob=args.target_burst_prob,
    )
    agent = MATD3(args.num_agents, state_dim, action_dim)
    buffer = ReplayBuffer()

    print("Training started...\n", flush=True)
    print(
        f"Agents: {args.num_agents} | Targets: {args.num_targets} | Episodes: {args.episodes}",
        flush=True,
    )
    print(f"Warmup steps: {args.warmup_steps} | Max steps/ep: {args.max_steps}", flush=True)
    print(f"Batch: {args.batch_size} | Train iters/step: {args.train_iters}\n", flush=True)

    total_steps = 0
    best_eval = -np.inf

    for ep in range(args.episodes):
        obs = env.reset()
        joint_state = np.concatenate(obs)
        ep_return = 0.0

        # Exploration noise decay (keeps things moving early, stabilizes late)
        exploration_noise = max(0.02, 0.25 * (1 - ep / max(1, args.episodes)))

        for _ in range(args.max_steps):
            actions = []
            for i in range(args.num_agents):
                if total_steps < args.warmup_steps:
                    a = np.random.uniform(-1, 1, action_dim)
                else:
                    a = agent.select_action(obs[i], explore=True, noise_scale=exploration_noise)
                actions.append(a)

            next_obs, rewards, done = env.step(actions)
            joint_action = np.concatenate(actions)
            next_joint_state = np.concatenate(next_obs)

            r = float(np.mean(rewards))
            buffer.add((joint_state, joint_action, r, next_joint_state, done))

            ep_return += r
            total_steps += 1

            if total_steps > args.warmup_steps:
                for _ in range(args.train_iters):
                    agent.train(buffer, batch_size=args.batch_size)

            obs = next_obs
            joint_state = next_joint_state

            if done:
                break

        if (ep + 1) % 25 == 0:
            print(
                f"Episode {ep+1:5d}/{args.episodes} | Return: {ep_return:9.2f} | Total steps: {total_steps}",
                flush=True,
            )

        if args.eval_every > 0 and (ep + 1) % args.eval_every == 0:
            eval_returns = []
            for _ in range(args.eval_episodes):
                obs_e = env.reset()
                ret = 0.0
                for _ in range(args.max_steps):
                    acts = [agent.select_action(obs_e[i], explore=False) for i in range(args.num_agents)]
                    obs_e, rew_e, done_e = env.step(acts)
                    ret += float(np.mean(rew_e))
                    if done_e:
                        break
                eval_returns.append(ret)

            eval_mean = float(np.mean(eval_returns))
            eval_std = float(np.std(eval_returns))
            improved = eval_mean > best_eval
            if improved:
                best_eval = eval_mean
                torch.save(agent.actor.state_dict(), args.save_best)

            arrow = "↑" if improved else "·"
            print(
                f"[EVAL]{arrow} ep {ep+1:5d} | mean {eval_mean:8.2f} ± {eval_std:6.2f} | best {best_eval:8.2f}",
                flush=True,
            )

    torch.save(agent.actor.state_dict(), args.save_final)
    print("\nTraining completed.", flush=True)
    print(f"Saved final actor: {args.save_final}", flush=True)
    print(f"Saved best actor : {args.save_best}", flush=True)


if __name__ == "__main__":
    main()