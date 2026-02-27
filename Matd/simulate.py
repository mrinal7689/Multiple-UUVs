from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque

import numpy as np
import pygame
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Matd.env import MultiAgentEnv
from Matd.matd3.matd3 import MATD3


def _parse_args():
    p = argparse.ArgumentParser(description="Pygame simulation for trained MATD3 multi-UUV policy.")
    p.add_argument("--model", type=str, default="matd3_actor.pth", help="Path to saved actor .pth")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--num-targets", type=int, default=4)
    p.add_argument(
        "--obs-targets-k",
        type=int,
        default=None,
        help="How many targets to include in the policy observation. "
        "If set (e.g. 2), you can simulate with more world targets than the model was trained on.",
    )
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--uuv-speed", type=float, default=18.0, help="UUV speed in pixels/step (visualization)")
    p.add_argument("--target-drift", type=float, default=0.35, help="Target random drift per step (visualization)")
    p.add_argument("--catch-radius", type=float, default=50.0)
    p.add_argument(
        "--world-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="World-like motion (targets have velocity/bursts, hunters use separation) like `classical-sim`.",
    )
    p.add_argument("--target-speed", type=float, default=3.5, help="(world-mode) Target base speed in pixels/step.")
    p.add_argument("--target-min-speed", type=float, default=1.2, help="(world-mode) Ensure targets keep moving.")
    p.add_argument("--target-burst-prob", type=float, default=0.08, help="(world-mode) Probability of a burst per step.")
    p.add_argument(
        "--continuous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Respawn targets on capture (final-product mode). Use --no-continuous to disable.",
    )
    p.add_argument(
        "--force-speed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize policy actions so UUVs move at constant speed magnitude.",
    )
    p.add_argument(
        "--action-gain",
        type=float,
        default=1.0,
        help="Multiply actor output by this before clipping (useful if actions are too small).",
    )
    p.add_argument(
        "--assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, use a pursuit controller (good for final visualization). Toggle in sim with key 'A'.",
    )
    p.add_argument(
        "--dynamic-assign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute target assignments each step (looks better with many targets).",
    )
    p.add_argument("--no-trails", action="store_true")
    p.add_argument("--no-lines", action="store_true")
    return p.parse_args()


def _build_obs(env: MultiAgentEnv, k: int | None):
    if k is None:
        return env._get_obs()

    k = max(0, int(k))
    w = float(env.width)
    h = float(env.height)
    world_scale = np.array([w, h], dtype=np.float32)

    target_pos = (
        np.array([t.pos for t in env.targets], dtype=np.float32)
        if env.targets
        else np.zeros((0, 2), dtype=np.float32)
    )
    obs = []

    def _pick_target_indices_for_agent(agent_idx: int) -> list[int]:
        if k == 0 or len(target_pos) == 0:
            return []

        # Prefer the environment's assignment (mimics "which target should I chase?")
        assigned = None
        if hasattr(env, "assignments") and isinstance(getattr(env, "assignments"), dict):
            assigned = env.assignments.get(agent_idx, None)
            if assigned is not None:
                assigned = int(assigned) % len(target_pos)

        uuv_pos = env.uuvs[agent_idx].pos
        dists = np.linalg.norm(target_pos - uuv_pos[None, :], axis=1)
        order = [int(i) for i in np.argsort(dists)]

        chosen: list[int] = []
        if assigned is not None:
            chosen.append(assigned)

        for j in order:
            if j not in chosen:
                chosen.append(j)
            if len(chosen) >= k:
                break

        return chosen[:k]

    for uuv in env.uuvs:
        uuv_pos = (uuv.pos / world_scale).astype(np.float32)

        if k == 0 or len(target_pos) == 0:
            obs.append(uuv_pos.copy())
            continue

        agent_idx = len(obs)
        idx = _pick_target_indices_for_agent(agent_idx)
        sel_targets = target_pos[idx] if len(idx) else np.zeros((0, 2), dtype=np.float32)
        sel = (sel_targets / world_scale).reshape(-1).astype(np.float32)

        if sel.shape[0] < 2 * k:
            sel = np.pad(sel, (0, 2 * k - sel.shape[0]), mode="constant", constant_values=0.0)

        obs.append(np.concatenate([uuv_pos, sel], dtype=np.float32))

    return obs


def main():
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    action_dim = 2

    # Load model first so we can infer what observation size it expects.
    try:
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.model, map_location="cpu")

    # Infer state_dim from actor's first layer.
    inferred_state_dim = int(state_dict.get("l1.weight").shape[1])
    if (inferred_state_dim - 2) % 2 != 0 or inferred_state_dim < 2:
        raise ValueError(f"Unexpected actor input dim: {inferred_state_dim}")
    inferred_targets = int((inferred_state_dim - 2) // 2)

    # If user didn't force an obs mapping, keep env targets aligned to the model.
    if args.obs_targets_k is None:
        model_targets = inferred_targets
        env_targets = model_targets
        obs_k = None
        state_dim = inferred_state_dim
    else:
        obs_k = int(args.obs_targets_k)
        state_dim = 2 + (2 * obs_k)
        env_targets = int(args.num_targets)
        if state_dim != inferred_state_dim:
            raise ValueError(
                f"Model expects state_dim={inferred_state_dim} (targets={inferred_targets}) but you set "
                f"--obs-targets-k {obs_k} => state_dim={state_dim}. Retrain or change --obs-targets-k."
            )

    env = MultiAgentEnv(
        width=args.width,
        height=args.height,
        n_uuv=args.num_agents,
        n_targets=env_targets,
        catch_radius=args.catch_radius,
        uuv_speed=args.uuv_speed,
        target_drift=args.target_drift,
        respawn_on_capture=args.continuous,
        world_mode=args.world_mode,
        # Keep world-mode target count stable (so obs size stays consistent).
        spawn_interval=10**9,
        max_active_targets=env_targets,
        min_active_targets=env_targets,
        target_speed=args.target_speed,
        target_min_speed=args.target_min_speed,
        target_burst_prob=args.target_burst_prob,
    )

    agent = MATD3(args.num_agents, state_dim, action_dim)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Multi-UUV MATD3 Simulation (Pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Precompute ocean gradient background (like `classical-sim`)
    ocean_top = (10, 40, 90)
    ocean_bottom = (0, 10, 40)
    ocean = pygame.Surface((args.width, args.height))
    for y in range(args.height):
        ratio = y / max(1, args.height - 1)
        r = int(ocean_top[0] * (1 - ratio) + ocean_bottom[0] * ratio)
        g = int(ocean_top[1] * (1 - ratio) + ocean_bottom[1] * ratio)
        b = int(ocean_top[2] * (1 - ratio) + ocean_bottom[2] * ratio)
        pygame.draw.line(ocean, (r, g, b), (0, y), (args.width, y))

    uuv_colors = [
        (0, 180, 255),
        (0, 255, 140),
        (180, 120, 255),
        (255, 170, 0),
        (255, 80, 160),
        (120, 240, 255),
    ]

    show_trails = not args.no_trails
    show_lines = not args.no_lines
    paused = False
    step_once = False
    assist = bool(args.assist)

    episode = 1
    step_idx = 0
    last_reward = 0.0

    trails = [deque(maxlen=200) for _ in range(args.num_agents)]

    env.reset()
    obs = _build_obs(env, args.obs_targets_k)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    step_once = True
                elif event.key == pygame.K_r:
                    env.reset()
                    obs = _build_obs(env, args.obs_targets_k)
                    step_idx = 0
                    last_reward = 0.0
                    trails = [deque(maxlen=200) for _ in range(args.num_agents)]
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_l:
                    show_lines = not show_lines
                elif event.key == pygame.K_a:
                    assist = not assist
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    args.fps = min(240, args.fps + 10)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    args.fps = max(5, args.fps - 10)

        do_step = (not paused) or step_once
        if do_step:
            step_once = False

            if args.dynamic_assign and hasattr(env, "assign_targets"):
                env.assignments = env.assign_targets()

            if assist and len(getattr(env, "targets", [])) > 0:
                # Pursuit controller: chase assigned target (or nearest fallback).
                actions = []
                for i, uuv in enumerate(env.uuvs):
                    if hasattr(env, "assignments") and isinstance(getattr(env, "assignments"), dict):
                        tj = env.assignments.get(i, 0)
                    else:
                        tj = 0
                    tj = int(tj) % len(env.targets)
                    target = env.targets[tj]
                    v = (target.pos - uuv.pos).astype(np.float32)
                    n = float(np.linalg.norm(v))
                    a = (v / n) if n > 1e-6 else np.zeros(2, dtype=np.float32)
                    actions.append(np.clip(a, -1.0, 1.0))
            else:
                raw_actions = [agent.select_action(obs[i], explore=False) for i in range(args.num_agents)]
                actions = []
                for a in raw_actions:
                    a = np.asarray(a, dtype=np.float32)
                    a = a * float(args.action_gain)
                    if args.force_speed:
                        n = float(np.linalg.norm(a))
                        if n > 1e-6:
                            a = a / n
                        else:
                            a = np.zeros_like(a)
                    actions.append(np.clip(a, -1.0, 1.0))
            _, rewards, done = env.step(actions)
            obs = _build_obs(env, args.obs_targets_k)

            last_reward = float(np.mean(rewards)) if len(rewards) else 0.0
            step_idx += 1

            for i, uuv in enumerate(env.uuvs):
                trails[i].append((float(uuv.pos[0]), float(uuv.pos[1])))

            if done or step_idx >= args.max_steps:
                env.reset()
                obs = _build_obs(env, args.obs_targets_k)
                episode += 1
                step_idx = 0
                trails = [deque(maxlen=200) for _ in range(args.num_agents)]

        # ------- RENDER -------
        screen.blit(ocean, (0, 0))

        # Draw targets
        for t in env.targets:
            pygame.draw.circle(screen, (255, 90, 90), (int(t.pos[0]), int(t.pos[1])), 7)
            pygame.draw.circle(
                screen,
                (255, 90, 90),
                (int(t.pos[0]), int(t.pos[1])),
                int(env.catch_radius),
                1,
            )

        # Assignment lines
        if show_lines and hasattr(env, "assignments"):
            for i, uuv in enumerate(env.uuvs):
                tj = env.assignments.get(i, None)
                if tj is None or tj >= len(env.targets):
                    continue
                t = env.targets[tj]
                pygame.draw.line(
                    screen,
                    (120, 120, 140),
                    (int(uuv.pos[0]), int(uuv.pos[1])),
                    (int(t.pos[0]), int(t.pos[1])),
                    1,
                )

        # Trails
        if show_trails:
            for i in range(args.num_agents):
                pts = list(trails[i])
                if len(pts) >= 2:
                    pygame.draw.lines(
                        screen,
                        (70, 70, 90),
                        False,
                        [(int(x), int(y)) for x, y in pts],
                        1,
                    )

        # Draw UUVs
        for i, uuv in enumerate(env.uuvs):
            color = uuv_colors[i % len(uuv_colors)]
            pygame.draw.circle(screen, color, (int(uuv.pos[0]), int(uuv.pos[1])), 10)
            pygame.draw.circle(screen, (0, 0, 0), (int(uuv.pos[0]), int(uuv.pos[1])), 10, 2)

        hud_lines = [
            f"Episode: {episode}",
            f"Step: {step_idx}/{args.max_steps}",
            f"Reward(mean): {last_reward:+.3f}",
            f"Captures: {getattr(env, 'captures', 0)}",
            f"Assist: {assist}   (A to toggle)",
            f"FPS cap: {args.fps}   ( +/- to change )",
            f"Paused: {paused}   ( SPACE pause, N step )",
            f"Trails: {show_trails} (T)   Lines: {show_lines} (L)   Reset: (R)   Quit: (Q/Esc)",
        ]
        y = 10
        for line in hud_lines:
            surf = font.render(line, True, (235, 235, 240))
            screen.blit(surf, (10, y))
            y += 20

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()